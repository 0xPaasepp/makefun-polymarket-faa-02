// polymarket-faa-attribution-02 entry (Node 22, stdlib only).
// Adapted from the polymarket-nvidia v3 model (practice pinball 0.2524 vs
// house baseline 0.3636), re-tuned for the FAA attribution family:
//   - NO mean reversion to x/sum: the four Yes prices sum to ~0.10, not
//     ~1.0 (the dominant outcome, "no attribution to any of these", has
//     no market), so x/sum would wrongly pull every price UP toward 0.2+.
//   - Adaptive decay: every market crashed 85-96% in its first ~4h
//     (Iran 0.495 -> 0.0175). The center decays toward 0 at the rate of
//     the last 120 minutes' log-trend, capped at 0.002/min, never upward:
//     after a jump the price falls back; a rise is a spike, not a trend.
//   - Asymmetry flipped vs nvidia (up 1.3x, down 0.8x): prices sit near
//     0, so down moves are bounded by the price itself while a news
//     break can spike a market +0.2 in a minute. Upside tail risk rules.
//   - SIGMA_MAX 0.008 -> 0.02: the family's opening jumps (0.08-0.215
//     in a minute) are far bigger than the NVIDIA event's; the cap must
//     not strangle the spread right after a jump.
// Kept from v3: per-market EWMA vol (halflife 8 + 1440 floor), point-mass
// mixture over a Student-t(3) jump distribution, h^0.41 horizon scaling,
// level vol floor 0.005*sqrt(x).
//
// Runs offline, no network, under the 8,000 ms per-case cap.
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
// Student-t(3) inverse CDF in SD units, tau = 0.005..0.995 step 0.005.
// Exact closed form: F(x) = 0.5 + (1/pi)*(atan(x) + x/(1+x^2)).
const T3_TABLE = [
  -3.3723,-2.6216,-2.2494,-2.0103,-1.8374,-1.7035,-1.595,-1.5042,-1.4265,-1.3587,
  -1.2987,-1.2449,-1.1962,-1.1518,-1.111,-1.0733,-1.0381,-1.0053,-0.9746,-0.9456,
  -0.9181,-0.8921,-0.8674,-0.8439,-0.8214,-0.7998,-0.7791,-0.7592,-0.74,-0.7216,
  -0.7037,-0.6864,-0.6697,-0.6535,-0.6377,-0.6224,-0.6075,-0.5929,-0.5788,-0.5649,
  -0.5514,-0.5382,-0.5253,-0.5126,-0.5002,-0.4881,-0.4761,-0.4644,-0.4529,-0.4416,
  -0.4305,-0.4195,-0.4088,-0.3981,-0.3877,-0.3774,-0.3672,-0.3571,-0.3472,-0.3374,
  -0.3277,-0.3181,-0.3087,-0.2993,-0.29,-0.2808,-0.2717,-0.2627,-0.2538,-0.2449,
  -0.2361,-0.2274,-0.2187,-0.2102,-0.2016,-0.1931,-0.1847,-0.1763,-0.168,-0.1597,
  -0.1515,-0.1433,-0.1351,-0.127,-0.1189,-0.1109,-0.1028,-0.0948,-0.0868,-0.0789,
  -0.0709,-0.063,-0.0551,-0.0472,-0.0393,-0.0314,-0.0236,-0.0157,-0.0079,-0.0,
  0.0079,0.0157,0.0236,0.0314,0.0393,0.0472,0.0551,0.063,0.0709,0.0789,
  0.0868,0.0948,0.1028,0.1109,0.1189,0.127,0.1351,0.1433,0.1515,0.1597,
  0.168,0.1763,0.1847,0.1931,0.2016,0.2102,0.2187,0.2274,0.2361,0.2449,
  0.2538,0.2627,0.2717,0.2808,0.29,0.2993,0.3087,0.3181,0.3277,0.3374,
  0.3472,0.3571,0.3672,0.3774,0.3877,0.3981,0.4088,0.4195,0.4305,0.4416,
  0.4529,0.4644,0.4761,0.4881,0.5002,0.5126,0.5253,0.5382,0.5514,0.5649,
  0.5788,0.5929,0.6075,0.6224,0.6377,0.6535,0.6697,0.6864,0.7037,0.7216,
  0.74,0.7592,0.7791,0.7998,0.8214,0.8439,0.8674,0.8921,0.9181,0.9456,
  0.9746,1.0053,1.0381,1.0733,1.111,1.1518,1.1962,1.2449,1.2987,1.3587,
  1.4265,1.5042,1.595,1.7035,1.8374,2.0103,2.2494,2.6216,3.3723,
];
const SIGMA_MIN = 0.0003;
const SIGMA_MAX = 0.02;
const HL_SHORT = 8; // minutes, EWMA halflife for recent vol
const HL_LONG = 1440; // minutes, EWMA halflife for jump-risk floor
const LONG_K = 0.3; // floor = LONG_K * long-term vol
const LOOKBACK = 360; // minutes of returns for both EWMA windows
const GAMMA = 0.41; // horizon scaling exponent: width ~ h^GAMMA
const SQRT_C = 0.005; // vol floor = SQRT_C * sqrt(x)
const PM_WIN = 120; // minutes for estimating p_move
const UP_SCALE = 1.3; // up-jump quantile scale (news-spike risk)
const DOWN_SCALE = 0.8; // down-jump quantile scale (bounded by price)
const DRIFT_WIN = 120; // minutes for estimating the log-decay rate
const DRIFT_MAX = 0.004; // max decay rate per minute (halving ~3h)
const DRIFT_MIN_MIN = 10; // min minutes of history for a drift estimate

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Inverse CDF of Student-t(3) in SD units, by linear interpolation
 * over T3_TABLE (tau = 0.005 + i*0.005). */
function t3Inv(tau: number): number {
  if (tau <= 0.005) return T3_TABLE[0]!;
  if (tau >= 0.995) return T3_TABLE[T3_TABLE.length - 1]!;
  const pos = tau / 0.005 - 1;
  const i = Math.floor(pos);
  const f = pos - i;
  return T3_TABLE[i]! * (1 - f) + T3_TABLE[i + 1]! * f;
}

/** EWMA std of 1-min returns over the last `lookback` minutes, halflife `hl`,
 * DETRENDED by the market's decay rate: a steady fall is a level shift,
 * not volatility, so the drift component (-drift * price per minute) is
 * removed before squaring. Without this, a crash inflates sigma and the
 * quantiles spread far too wide around a center that is already moving. */
function ewmaVol(prices: number[], halflife: number, lookback: number, drift: number): number {
  const n = prices.length;
  if (n < 2) return 0;
  const start = Math.max(0, n - lookback - 1);
  const alpha = 1 - Math.pow(0.5, 1 / halflife);
  let variance = 0;
  let wsum = 0;
  let w = 1;
  for (let i = n - 2; i >= start; i--) {
    const r = (prices[i + 1]! - prices[i]!) + drift * prices[i]!;
    variance += w * r * r;
    wsum += w;
    w *= 1 - alpha;
  }
  return wsum > 0 ? Math.sqrt(variance / wsum) : 0;
}

/** Per-market 1-min return volatility used for every horizon. */
function marketSigma(prices: number[], drift: number): number {
  const sigShort = ewmaVol(prices, HL_SHORT, LOOKBACK, drift);
  const sigLong = ewmaVol(prices, HL_LONG, LOOKBACK, drift);
  const x = prices[prices.length - 1]!;
  return Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, sigShort, LONG_K * sigLong, SQRT_C * Math.sqrt(x)));
}

/** Fraction of the last PM_WIN minutes with a nonzero price move. */
function moveProb(prices: number[]): number {
  const n = prices.length;
  if (n < 2) return 0.1;
  const start = Math.max(0, n - PM_WIN - 1);
  let moves = 0;
  let total = 0;
  for (let i = start; i < n - 1; i++) {
    total++;
    if (prices[i + 1] !== prices[i]) moves++;
  }
  if (total === 0) return 0.1;
  return Math.min(0.9, Math.max(0.01, moves / total));
}

/** Decay rate per minute toward 0, from the last DRIFT_WIN minutes'
 * log-trend. Only decay (never growth): a rise is a spike, not a trend.
 * Needs >= DRIFT_MIN_MIN minutes of history, else null (no estimate).
 * UNCAPPED: the center caps it (DRIFT_MAX), but the volatility detrend
 * uses the full rate — a 5%/min crash is a trend, and the noise around
 * it is what the spread should measure. */
function ownDecayRate(prices: number[]): number | null {
  const n = prices.length;
  if (n < DRIFT_MIN_MIN + 1) return null;
  const start = Math.max(0, n - DRIFT_WIN);
  const p0 = prices[start]!;
  const p1 = prices[n - 1]!;
  if (p0 <= 0 || p1 <= 0) return null;
  const span = n - 1 - start;
  if (span < DRIFT_MIN_MIN) return null;
  const rate = -Math.log(p1 / p0) / span;
  if (rate <= 0) return 0;
  return rate;
}

/** Family decay: the median of the markets' own decay rates. A market
 * with too little history borrows the family's signal — in a crash all
 * four fall together, so a 4-minute-old market still gets the decay.
 * The family's rate is also a floor for every market: a crash is a
 * family event, not a per-market quirk. */
function familyDecay(markets: CaseMarket[]): number {
  const rates = markets.map((m) => ownDecayRate(m.prices)).filter((r): r is number => r !== null);
  if (rates.length === 0) return 0;
  rates.sort((a, b) => a - b);
  return rates[Math.floor(rates.length / 2)]!;
}

function forecast(m: CaseMarket, horizon: number, famDrift: number): number[][] {
  const p = m.prices;
  if (p.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = p[p.length - 1]!;
  const rawDrift = Math.max(ownDecayRate(p) ?? 0, famDrift);
  const drift = Math.min(DRIFT_MAX, rawDrift);
  const sigma = marketSigma(p, rawDrift);
  const pm = moveProb(p);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const center = x * Math.exp(-drift * h);
    const s = sigma * Math.pow(h, GAMMA);
    // Point-mass mixture: with prob p0 the price is exactly `center`.
    const p0 = Math.pow(1 - pm, h);
    const tauLow = (1 - p0) * 0.5;
    const tauHigh = tauLow + p0;
    const row: number[] = [];
    for (const tau of QUANTILES) {
      let q: number;
      if (tau < tauLow) {
        q = center + t3Inv(tau / (1 - p0)) * s * DOWN_SCALE;
      } else if (tau <= tauHigh) {
        q = center;
      } else {
        q = center + t3Inv((tau - p0) / (1 - p0)) * s * UP_SCALE;
      }
      row.push(price6(q));
    }
    for (let j = 1; j < row.length; j++) {
      if (row[j]! < row[j - 1]!) row[j] = row[j - 1]!;
    }
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  const famDrift = familyDecay(c.markets);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes, famDrift) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
