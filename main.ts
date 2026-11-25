// Tiny mean-reversion bot for make.fun "Tiny Bot, Easy $2: Beat Buy-and-Hold on Solana"
// Harness: agent-fun/bars-replay/v1 (hourly Solana bars, 7 symbols, USDC quote).
//
// Idea (my own design, backtested offline on the dev window):
//   Crypto hourly prices mean-revert around their 20-bar average far more often
//   than they trend cleanly, and the dev window is a slow grind down, so a bot
//   that only chases trends bleeds fees and loses to buy-and-hold in flat
//   windows. This bot therefore buys *dips* and sells the reversion, and never
//   chases a rally:
//   - DIP entry: a symbol's close falls below its 20-bar mean by 2 standard
//     deviations (a Bollinger lower-band touch). Buy 15% of equity.
//     Exit: close recovers to the 20-bar mean, or after 6 bars (do not let a
//     position rot in a falling market; freeing the capital is what keeps the
//     trade count healthy and the drawdown small).
//   - SCALP entry: a symbol drops 1.5% or more over the last 6 bars without
//     quite touching the lower band. Buy 10% of equity. Exit: +1% take-profit,
//     -3% stop, or after 6 bars, whichever comes first.
//   - STAGGER: at most 2 new positions per bar. After a sharp market drop
//     every symbol touches its lower band at once; buying all seven in one
//     bar would deploy 100% of the capital into a still-falling market and
//     let a single bar's timing decide the whole run. Spreading entries over
//     successive bars (signals on hourly bars persist for several bars) cuts
//     that concentration risk without losing the trades.
//   Both entries only fire after a 24-bar warmup (the Bollinger window needs
//   20 bars of history plus a few bars to settle). No shorts, no leverage,
//   at most one open position per symbol, at most 15% of equity per position
//   (under the harness 25% cap), at most 14 orders per bar (under the 16 cap).
//
// Determinism: every decision is a pure function of the bars received so far.
// No clock, no randomness, no network, no files. Two runs on the same replay
// therefore produce byte-identical action logs.
//
// Protocol (from protocol.schema.json, implemented directly, no SDK):
//   stdin:  {"type":"hello",...} then {"type":"batch","seq":n,"bars":[...],
//           "fills":[...],"account":{...}} per bar, then {"type":"done",...}.
//   stdout: one {"type":"reply","seq":n,"orders":[...]} per batch.
//   Orders decided on a batch fill at the next bar's open (1-bar latency), so
//   signals are computed on the just-closed bar and fills are read back from
//   the next batch's "fills" array. Holdings are always taken from the
//   account snapshot the harness sends, never assumed from our own orders.

import * as readline from "node:readline";

// ---- strategy parameters (chosen by offline backtest; stable under +-20%) ----
const BB_N = 20; // Bollinger window (bars)
const BB_K = 2.0; // entry when close < mean - K * stdev
const DIP_WEIGHT_BPS = 1500; // 15% of equity per dip position
const DIP_MAX_HOLD = 6; // bars: exit a dip trade that has not reverted
const SCALP_LOOKBACK = 6; // bars for the sharp-drop test
const SCALP_DROP = 0.015; // enter a scalp when the 6-bar return <= -1.5%
const SCALP_WEIGHT_BPS = 1000; // 10% of equity per scalp position
const SCALP_TP = 0.01; // take profit +1%
const SCALP_STOP = 0.03; // stop loss -3%
const SCALP_MAX_HOLD = 6; // bars
const MAX_NEW_PER_BAR = 2; // stagger entries: at most 2 new positions per bar
const WARMUP_BARS = 24; // bars of history before the first trade
const MAX_SLIPPAGE_BPS = 100; // 1% cap per order; pool depth makes fills far tighter

type Mode = "dip" | "scalp";

interface SymState {
  priceScale: number;
  qtyScale: number;
  closes: number[];
}

interface Held {
  qty: string; // base units, integer string at the symbol's qty scale
  entry: number; // fill price in quote currency (USDC per base unit)
  age: number; // bars held
  mode: Mode;
}

const symbols = new Map<string, SymState>();
const held = new Map<string, Held>();
let maxWeightBps = 2500;
let minNotional = 1000000n; // micro-USDC (1 USDC), from the hello message
let quoteScale = 6;
let warmupBars = WARMUP_BARS;
let barCount = 0; // batches seen (one bar per symbol per batch in this harness)

function sma(values: number[], n: number): number {
  const w = values.slice(-n);
  let s = 0;
  for (const v of w) s += v;
  return s / w.length;
}

function stdev(values: number[], n: number, mean: number): number {
  const w = values.slice(-n);
  let s = 0;
  for (const v of w) {
    const d = v - mean;
    s += d * d;
  }
  return Math.sqrt(s / w.length);
}

function toPrice(raw: string, scale: number): number {
  return Number(BigInt(raw)) / 10 ** scale;
}

function decide(msg: any): any[] {
  const orders: any[] = [];
  const bars: any[] = msg.bars ?? [];
  const account: any = msg.account ?? {};
  const equity = BigInt(account.equity_quote_units ?? "0");
  let cash = BigInt(account.cash_quote_units ?? "0");

  // 1. Record the new bars.
  for (const bar of bars) {
    const st = symbols.get(bar.symbol);
    if (!st) continue;
    st.closes.push(toPrice(bar.c, st.priceScale));
  }
  barCount += 1;

  // 2. Read back fills: a filled buy sets the entry price and mode; a filled
  //    sell clears the position. Client ids are "d-<bar>-<SYM>" (dip),
  //    "s-<bar>-<SYM>" (scalp) and "x-<bar>-<SYM>" (exit).
  for (const fill of msg.fills ?? []) {
    if (fill.status !== "filled") continue;
    const id: string = fill.client_id ?? "";
    const kind = id.charAt(0);
    const st = symbols.get(fill.symbol);
    if (!st) continue;
    if (kind === "d" || kind === "s") {
      held.set(fill.symbol, {
        qty: fill.qty,
        entry: toPrice(fill.price, st.priceScale),
        age: 0,
        mode: kind === "d" ? "dip" : "scalp",
      });
    } else if (kind === "x") {
      held.delete(fill.symbol);
    }
  }

  // 3. Reconcile with the harness account snapshot (source of truth for what
  //    we actually hold). A position with no fill record yet (should not
  //    happen) gets the current close as a provisional entry price.
  const liveQty = new Map<string, string>();
  for (const p of account.positions ?? []) {
    if (BigInt(p.qty) > 0n) liveQty.set(p.symbol, p.qty);
  }
  for (const [sym, h] of [...held.entries()]) {
    if (!liveQty.has(sym)) held.delete(sym);
    else h.qty = liveQty.get(sym)!;
  }
  for (const [sym, qty] of liveQty.entries()) {
    if (!held.has(sym)) {
      const st = symbols.get(sym)!;
      const last = st.closes[st.closes.length - 1];
      held.set(sym, { qty, entry: last, age: 0, mode: "dip" });
    }
  }
  // Age the positions we hold.
  for (const h of held.values()) h.age += 1;

  // 4. Decide. No trading until the warmup bars have passed.
  if (barCount < warmupBars) return orders;

  const dipBps = Math.min(DIP_WEIGHT_BPS, maxWeightBps);
  const scalpBps = Math.min(SCALP_WEIGHT_BPS, maxWeightBps);
  let newThisBar = 0;

  for (const bar of bars) {
    const sym: string = bar.symbol;
    const st = symbols.get(sym);
    if (!st || st.closes.length < BB_N) continue;
    const closes = st.closes;
    const close = closes[closes.length - 1];
    const mean = sma(closes, BB_N);
    const sd = stdev(closes, BB_N, mean);
    const lower = mean - BB_K * sd;
    const pos = held.get(sym);

    if (!pos) {
      const isDip = close < lower;
      const sixAgo = closes[closes.length - 1 - SCALP_LOOKBACK];
      const isScalp =
        sixAgo !== undefined && close / sixAgo - 1 <= -SCALP_DROP;
      if ((!isDip && !isScalp) || newThisBar >= MAX_NEW_PER_BAR) continue;
      const bps = isDip ? dipBps : scalpBps;
      let notional = (equity * BigInt(bps)) / 10000n;
      if (notional > cash) notional = cash;
      if (notional < minNotional) continue;
      cash -= notional;
      newThisBar += 1;
      orders.push({
        side: "buy",
        client_id: `${isDip ? "d" : "s"}-${barCount}-${sym}`,
        symbol: sym,
        notional_quote_units: notional.toString(),
        max_slippage_bps: MAX_SLIPPAGE_BPS,
      });
    } else {
      let exit = false;
      if (pos.mode === "dip") {
        // Sell the reversion to the mean; do not sit in a falling market.
        exit = close >= mean || pos.age >= DIP_MAX_HOLD;
      } else {
        exit =
          close >= pos.entry * (1 + SCALP_TP) ||
          close <= pos.entry * (1 - SCALP_STOP) ||
          pos.age >= SCALP_MAX_HOLD;
      }
      if (exit) {
        orders.push({
          side: "sell",
          client_id: `x-${barCount}-${sym}`,
          symbol: sym,
          qty: pos.qty,
          max_slippage_bps: MAX_SLIPPAGE_BPS,
        });
      }
    }
    if (orders.length >= 16) break; // harness cap: max 16 orders per bar
  }
  return orders;
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

rl.on("line", (line: string) => {
  if (!line.trim()) return;
  let msg: any;
  try {
    msg = JSON.parse(line);
  } catch {
    return;
  }
  if (msg.type === "hello") {
    quoteScale = msg.capital?.quote_scale ?? 6;
    maxWeightBps = msg.caps?.max_weight_bps ?? 2500;
    if (msg.fees?.min_notional_quote_units) {
      minNotional = BigInt(msg.fees.min_notional_quote_units);
    }
    if (typeof msg.warmup_bars === "number" && msg.warmup_bars > warmupBars) {
      warmupBars = msg.warmup_bars;
    }
    for (const s of msg.symbols ?? []) {
      symbols.set(s.symbol, {
        priceScale: s.price_scale,
        qtyScale: s.qty_scale,
        closes: [],
      });
    }
    return;
  }
  if (msg.type === "batch") {
    const orders = decide(msg);
    process.stdout.write(
      JSON.stringify({ type: "reply", seq: msg.seq, orders }) + "\n",
    );
    return;
  }
  if (msg.type === "done") {
    rl.close();
    process.exit(0);
  }
});
