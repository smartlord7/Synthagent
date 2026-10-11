/**
 * sim/engine.js
 * Responsibility: the Monte Carlo engine shared by every tab. A configuration is prepared once into a
 * context (sizes, graph, averaging operators, privacy, adversaries, regions); each replicate draws model (4),
 * releases privatized estimates, and computes every enabled method; an accumulator turns replicates into
 * per-hospital and aggregate metrics with Monte Carlo standard errors.
 * Methods: local, pooled (single global model), oracle HB, plug-in GHB (Algorithm 1), robust t prior,
 * opt-out, release screening and GMRF.
 */
import { makeRng } from '../math/random.js';
import { metropolisHastings, slem, connected, matPow, matMul, edgeList, failLinks, predictedRounds, zeros } from '../math/graph.js';
import { dpVar } from '../math/dp.js';
import {
  drawReplicate, avgOp, borrow, combineAnchor, shrinkWeights, oracleHB, pooledMean, robustPosteriorMean,
  optOut, robustScreenGHB, gmrfJacobi, looMSEiv, borrowedPatients
} from '../math/ghb.js';
import { resolveSizes, resolveGraph, resolvePrivacy, resolveAdversaries, resolveRegions } from '../config.js';

/** Method identifiers, labels and colour tokens; the same token colours a method everywhere. */
export const METHODS = [
  { id: 'local', label: 'Local only', tok: '--rd-local' },
  { id: 'pooled', label: 'Global pooled', tok: '--rd-ref' },
  { id: 'oracle', label: 'Oracle HB', tok: '--rd-oracle' },
  { id: 'ghb', label: 'Plug-in GHB', tok: '--rd-ghb' },
  { id: 'robust', label: 'Robust t prior', tok: '--rd-robust' },
  { id: 'optout', label: 'GHB + opt-out', tok: '--rd-noise' },
  { id: 'screen', label: 'Screened GHB', tok: '--rd-robust2' },
  { id: 'gmrf', label: 'GMRF', tok: '--rd-gmrf' }
];
export const METHOD = Object.fromEntries(METHODS.map((m) => [m.id, m]));

/** Bins of the standardized site effect |u_k|/tau (Simulation 9). */
export const U_BINS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 12];

/**
 * Prepare a configuration for simulation.
 * @param {Object} c configuration (config.js)
 * @returns {Object} context: n, K, A, W, edges, lambda, connected, rounds, avgs, priv, q, bad, regions, simA, methods, ...
 * @see Section 8.7 (simulation design), Proposition 8.4, Simulation 13
 */
export function prepare(c) {
  const n = resolveSizes(c), K = n.length, A = resolveGraph(c, K), W = metropolisHastings(A), edges = edgeList(A);
  const conn = connected(A), lambda = conn ? slem(W) : 1;
  const priv = resolvePrivacy(c);
  const tau2m = c.regional ? c.tauR2 + c.tauW2 : c.tau2;
  const q = { tau2: tau2m, sigma2: c.sigma2, eps: priv.eps, delta: priv.delta, C: priv.C, mu: c.mu };
  let avgs;
  if (c.exact) avgs = [avgOp(null, K)];
  else if (c.pFail > 0) {
    const r = makeRng(1300 + c.seed); avgs = [];
    for (let s = 0; s < 8; s++) {
      let P = null;
      for (let t = 0; t < c.T; t++) { const Wt = metropolisHastings(failLinks(edges, K, c.pFail, r.u).A); P = P ? matMul(Wt, P) : Wt; }
      avgs.push(avgOp(P, K));
    }
  } else avgs = [avgOp(matPow(W, c.T), K)];
  const regions = resolveRegions(c, K);
  let simA;
  if (c.regional) { simA = zeros(K); for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) if (i !== j && regions[i] === regions[j]) simA[i][j] = 1; }
  else simA = A;
  const bad = resolveAdversaries(c, n), isBad = new Uint8Array(K); bad.forEach((k) => { isBad[k] = 1; });
  const outIdx = c.outAt === 'smallest' ? 0 : c.outAt === 'largest' ? K - 1 : Math.floor(K / 2);
  const methods = ['local', 'pooled', 'oracle', 'ghb'];
  if (c.robustT) methods.push('robust');
  if (c.optOut) methods.push('optout');
  if (c.screen) methods.push('screen');
  if (c.gmrf) methods.push('gmrf');
  const v = n.map((x) => c.sigma2 / x), s2 = n.map((x) => dpVar(x, priv.eps, priv.delta, priv.C));
  const totals = n.map((_, k) => tau2m + v[k] + s2[k]);
  const Mth = n.map((_, k) => looMSEiv(totals, k)), Bth = Mth.map((M) => borrowedPatients(tau2m, M, c.sigma2));
  return {
    c, n, K, A, W, edges, lambda, connected: conn, rounds: conn ? predictedRounds(lambda, c.gammaTol) : Infinity,
    avgs, priv, q, tau2m, regions, simA, bad, isBad, outIdx, methods, v, s2, Mth, Bth
  };
}

/**
 * Site effects u_k for one replicate.
 * @param {Object} ctx prepared context
 * @param {{n:()=>number, t:(nu:number)=>number}} rng generator
 * @returns {Float64Array}
 * @see Eq. (4); Simulation 6 (regions); Simulation 9 (heavy tails, outliers)
 */
export function drawEffects(ctx, rng) {
  const { c, K, regions, outIdx } = ctx, base = c.regional ? c.tauW2 : c.tau2, sd = Math.sqrt(base);
  const reg = c.regional ? Array.from({ length: c.regions }, () => Math.sqrt(c.tauR2) * rng.n()) : null;
  const u = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    let w;
    if (c.effects === 't') w = sd * Math.sqrt((c.nu - 2) / c.nu) * rng.t(c.nu);
    else w = sd * rng.n();
    if (c.effects === 'outlier' && k === outIdx) w = c.outK * sd;
    u[k] = (reg ? reg[regions[k]] : 0) + w;
  }
  return u;
}

/**
 * One replicate: draw, release, borrow, and every enabled estimator.
 * @param {Object} ctx prepared context
 * @param {Object} rng generator
 * @param {number} rep replicate index (selects the link-failure realization)
 * @returns {{d:Object, u:Float64Array, b:Object, alpha:Float64Array, est:Object<string,Float64Array>, keep:boolean[]|null, mu0:number|null}}
 * @see Algorithm 1; Propositions 8.1, 8.2, 8.4; Simulations 9, 11, 12
 */
export function replicate(ctx, rng, rep) {
  const { c, n, K, q, bad, methods } = ctx;
  const u = drawEffects(ctx, rng);
  const d = drawReplicate(rng, n, q, (k) => u[k]);
  for (const k of bad) d.x[k] = c.mu + c.lie * Math.sqrt(ctx.tau2m);
  const avg = ctx.avgs[rep % ctx.avgs.length];
  let b = borrow(d.x, d.r, avg, { estimator: c.tauEst, tau2: ctx.tau2m, scale: c.cauchy, weights: c.weights });
  let mu0 = null;
  if (c.anchor) { mu0 = c.mu + c.anchorErr * rng.n(); b = combineAnchor(b, mu0, c.anchorSd ** 2, c.anchorCheck); }
  const alpha = shrinkWeights(d.v, b), est = {};
  const ghb = new Float64Array(K); for (let k = 0; k < K; k++) ghb[k] = alpha[k] * d.th[k] + (1 - alpha[k]) * b.mu[k];
  est.local = d.th; est.ghb = ghb;
  const pm = pooledMean(d.th, n); est.pooled = new Float64Array(K).fill(pm);
  est.oracle = oracleHB(d.th, d.x, d.v, d.s2, ctx.tau2m);
  if (methods.includes('robust')) { const e = new Float64Array(K); for (let k = 0; k < K; k++) e[k] = robustPosteriorMean(d.th[k], d.v[k], b.mu[k], b.t2[k] + b.M[k], c.robustNu); est.robust = e; }
  if (methods.includes('optout')) { const e = new Float64Array(K); for (let k = 0; k < K; k++) e[k] = optOut(d.th[k], ghb[k], d.v[k], b.mu[k], b.t2[k], b.M[k], c.optZ); est.optout = e; }
  let keep = null;
  if (methods.includes('screen')) { const s = robustScreenGHB(d.th, d.x, d.r, d.v, c.screenZ); est.screen = s.est; keep = s.keep; }
  if (methods.includes('gmrf')) est.gmrf = gmrfJacobi(d.th, d.x, d.v, d.r, ctx.simA, c.beta, c.gamma, mu0 ?? 0, 100);
  return { d, u, b, alpha, est, keep, mu0 };
}

/**
 * Empty accumulator for a context.
 * @param {Object} ctx prepared context
 * @returns {Object}
 */
export function makeAccumulator(ctx) {
  const K = ctx.K, nb = U_BINS.length - 1, per = () => ({ s: new Float64Array(K), ss: new Float64Array(K), m: 0, mm: 0, h: 0, hh: 0, bin: new Float64Array(nb) });
  return {
    R: 0, K, m: Object.fromEntries(ctx.methods.map((id) => [id, per()])), binN: new Float64Array(nb),
    t2: 0, M: new Float64Array(K), alpha: new Float64Array(K), anchorUsed: 0, keptBad: 0, droppedHonest: 0, example: null
  };
}

/**
 * Add one replicate to the accumulator (relative squared error (est - theta)^2 / v_k).
 * @param {Object} ctx context
 * @param {Object} acc accumulator
 * @param {Object} rr output of replicate()
 */
export function accumulate(ctx, acc, rr) {
  const { K, isBad, c } = ctx, { d, u, b, alpha } = rr, tau = Math.sqrt(c.regional ? c.tauW2 : c.tau2);
  const bins = new Int16Array(K);
  for (let k = 0; k < K; k++) { const z = Math.abs(u[k] - (c.regional ? 0 : 0)) / tau; let bi = 0; while (bi < U_BINS.length - 2 && z >= U_BINS[bi + 1]) bi++; bins[k] = bi; acc.binN[bi]++; }
  const honest = K - ctx.bad.length;
  for (const id of ctx.methods) {
    const e = rr.est[id], a = acc.m[id]; let m = 0, h = 0;
    for (let k = 0; k < K; k++) {
      const rel = (e[k] - d.theta[k]) ** 2 / d.v[k];
      a.s[k] += rel; a.ss[k] += rel * rel; m += rel; a.bin[bins[k]] += rel;
      if (!isBad[k]) h += rel;
    }
    m /= K; h /= Math.max(1, honest);
    a.m += m; a.mm += m * m; a.h += h; a.hh += h * h;
  }
  let t = 0; for (let k = 0; k < K; k++) { t += b.t2[k]; acc.M[k] += b.M[k]; acc.alpha[k] += alpha[k]; if (b.used) acc.anchorUsed += b.used[k]; }
  acc.t2 += t / K;
  if (rr.keep) for (let k = 0; k < K; k++) { if (isBad[k] && rr.keep[k]) acc.keptBad++; if (!isBad[k] && !rr.keep[k]) acc.droppedHonest++; }
  if (!acc.example) acc.example = rr;
  acc.R++;
}

const seOf = (s, ss, R) => (R > 1 ? Math.sqrt(Math.max(ss / R - (s / R) ** 2, 0) / (R - 1)) : NaN);

/**
 * Metrics from an accumulator: per hospital and aggregated, with Monte Carlo standard errors.
 * @param {Object} ctx context
 * @param {Object} acc accumulator
 * @returns {Object} { R, perHospital: [{k, n, v, s, x, mu, M, t2, alpha, B, neff, gain, rel:{id:{y,se}}}], agg:{id:{mean, meanSE, worst, worstSE, worstK, share, honest, honestSE}}, bins, diagnostics }
 * @see Proposition 8.1 (relative risk, n_eff = sigma^2/R_k), Eq. (5)
 */
export function summarize(ctx, acc) {
  const { K, n, c } = ctx, R = acc.R, agg = {}, ex = acc.example;
  for (const id of ctx.methods) {
    const a = acc.m[id], y = Array.from(a.s, (s) => s / R), se = Array.from(a.s, (s, k) => seOf(s, a.ss[k], R));
    let wk = 0; for (let k = 1; k < K; k++) if (y[k] > y[wk]) wk = k;
    agg[id] = {
      y, se, mean: a.m / R, meanSE: seOf(a.m, a.mm, R), worst: y[wk], worstSE: se[wk], worstK: wk,
      share: y.filter((x) => x < 1).length / K, honest: a.h / R, honestSE: seOf(a.h, a.hh, R),
      bins: Array.from(a.bin, (s, i) => (acc.binN[i] > 30 ? s / acc.binN[i] : NaN))
    };
  }
  const per = n.map((nk, k) => {
    const g = agg.ghb.y[k];
    return {
      k, n: nk, v: ctx.v[k], s: Math.sqrt(ctx.s2[k]), x: ex ? ex.d.x[k] : NaN, th: ex ? ex.d.th[k] : NaN, theta: ex ? ex.d.theta[k] : NaN,
      mu: ex ? ex.b.mu[k] : NaN, est: ex ? ex.est.ghb[k] : NaN, M: acc.M[k] / R, t2: acc.t2 / R, alpha: acc.alpha[k] / R,
      B: ctx.Bth[k], neffTheory: nk + ctx.Bth[k], neff: nk / g, gain: 1 / g, nsr: ctx.s2[k] / ctx.v[k], bad: !!ctx.isBad[k],
      rel: Object.fromEntries(ctx.methods.map((id) => [id, { y: agg[id].y[k], se: agg[id].se[k] }]))
    };
  });
  const nb = ctx.bad.length;
  return {
    R, perHospital: per, agg, binN: Array.from(acc.binN),
    diagnostics: {
      t2hat: acc.t2 / R, anchorsDropped: c.anchor ? 1 - acc.anchorUsed / (R * K) : null,
      keptBad: c.screen && nb ? acc.keptBad / (R * nb) : null, droppedHonest: c.screen ? acc.droppedHonest / (R * (K - nb)) : null,
      Bmean: ctx.Bth.reduce((a, b) => a + b, 0) / K, nsrMean: per.reduce((a, p) => a + p.nsr, 0) / K
    }
  };
}

/**
 * Monte Carlo run as a chunked generator (see sim/scheduler.js).
 * @param {Object} c configuration
 * @param {{R?:number, onProgress?:(summary:Object, done:boolean)=>void, every?:number}} [opt] replicate override and progress callback
 * @returns {Generator<void, Object>} returns the final summary
 */
export function* runMonteCarlo(c, opt = {}) {
  const ctx = prepare(c), acc = makeAccumulator(ctx), rng = makeRng(c.seed * 104729 + 1), R = opt.R ?? c.R;
  const chunk = Math.max(1, c.chunk), every = opt.every ?? Math.max(chunk, Math.round(R / 12));
  let next = every;
  while (acc.R < R) {
    for (let i = 0; i < chunk && acc.R < R; i++) accumulate(ctx, acc, replicate(ctx, rng, acc.R));
    if (opt.onProgress && (acc.R >= next || acc.R >= R)) { next += every; opt.onProgress(summarize(ctx, acc), acc.R >= R, ctx); }
    yield;
  }
  return { ctx, summary: summarize(ctx, acc) };
}
