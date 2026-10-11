/**
 * sim/sweep.js
 * Responsibility: parameter sweeps. Any sweepable parameter of config.js can be varied over a linear or log
 * grid; any metric below is recorded per grid point, with Monte Carlo standard errors for simulated metrics.
 * Closed-form metrics are computed without simulation. Also holds the catalogue of the paper's figures,
 * each expressed as a sweep (or closed-form curve family) on the current configuration.
 */
import { prepare, makeAccumulator, accumulate, replicate, summarize } from './engine.js';
import { makeRng } from '../math/random.js';
import { dpVar } from '../math/dp.js';
import { looMSE, borrowedPatients, releaseUnits, jointReleaseB, median } from '../math/ghb.js';

/**
 * Metrics that can be plotted against a swept parameter.
 * kind 'mc' needs simulation (per method, with standard errors); kind 'closed' is computed from the context.
 */
export const METRICS = [
  { key: 'mean', label: 'Mean relative risk R/v', unit: 'R_k / v_k', kind: 'mc', perMethod: true },
  { key: 'worst', label: 'Worst-hospital relative risk', unit: 'R_k / v_k', kind: 'mc', perMethod: true },
  { key: 'honest', label: 'Honest hospitals mean relative risk', unit: 'R_k / v_k', kind: 'mc', perMethod: true },
  { key: 'share', label: 'Share of hospitals that gain', unit: 'fraction', kind: 'mc', perMethod: true },
  { key: 'gainSmall', label: 'n_eff / n of the smallest hospital', unit: 'ratio', kind: 'mc', perMethod: true },
  { key: 't2hat', label: 'Mean estimated τ²', unit: 'τ²', kind: 'mc' },
  { key: 'Bmean', label: 'Borrowed patients B (Eq. 5, inverse variance)', unit: 'patients', kind: 'closed' },
  { key: 'Beq6', label: 'Borrowed patients B (Eq. 6, equal weights)', unit: 'patients', kind: 'closed' },
  { key: 'Bjoint', label: 'Borrowed patients B, joint release in groups of g', unit: 'patients', kind: 'closed' },
  { key: 'essRef', label: 'n_eff / n of the reference hospital', unit: 'ratio', kind: 'closed' },
  { key: 'noiseRef', label: 'Release noise sd s of the reference hospital', unit: 'sd', kind: 'closed' },
  { key: 'slem', label: 'SLEM λ', unit: 'λ', kind: 'closed' },
  { key: 'rounds', label: 'Rounds to tolerance γ', unit: 'rounds', kind: 'closed' },
  { key: 'lamT', label: 'Consensus error bound λ^T', unit: 'ratio', kind: 'closed' }
];
export const METRIC = Object.fromEntries(METRICS.map((m) => [m.key, m]));

/**
 * Grid of values for a sweep.
 * @param {number} start first value
 * @param {number} stop last value
 * @param {number} steps number of points (>= 2)
 * @param {boolean} log geometric spacing
 * @param {boolean} [int=false] round to integers and drop duplicates
 * @returns {number[]}
 */
export function grid(start, stop, steps, log, int = false) {
  const out = [];
  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? 0 : i / (steps - 1);
    let x = log ? Math.exp(Math.log(start) + t * (Math.log(stop) - Math.log(start))) : start + t * (stop - start);
    if (int) x = Math.round(x);
    if (!out.length || out[out.length - 1] !== x) out.push(x);
  }
  return out;
}

/**
 * A closed-form metric for a prepared context.
 * @param {Object} ctx context from engine.prepare
 * @param {string} key metric key
 * @returns {number}
 * @see Eq. (5), Eq. (6), Proposition 8.4
 */
export function closedMetric(ctx, key) {
  const { c, n, K } = ctx;
  switch (key) {
    case 'Bmean': return ctx.Bth.reduce((a, b) => a + b, 0) / K;
    case 'Beq6': { let b = 0; for (let k = 0; k < K; k++) b += borrowedPatients(ctx.tau2m, looMSE(n, k, { ...ctx.q }), c.sigma2); return b / K; }
    case 'Bjoint': return jointReleaseB(releaseUnits(n, c.group, median(n)), ctx.q);
    case 'essRef': { const refs = n.concat([c.refN]), M = looMSE(refs, refs.length - 1, ctx.q); return 1 + borrowedPatients(ctx.tau2m, M, c.sigma2) / c.refN; }
    case 'noiseRef': return Math.sqrt(dpVar(c.refN, ctx.priv.eps, ctx.priv.delta, ctx.priv.C));
    case 'slem': return ctx.lambda;
    case 'rounds': return ctx.rounds;
    case 'lamT': return ctx.lambda ** c.T;
    default: return NaN;
  }
}

/**
 * Run a sweep as a chunked generator.
 * @param {Object} c base configuration
 * @param {{param:string, values:number[], metrics:string[], methods:string[], R:number}} spec sweep definition
 * @param {(res:Object, done:boolean)=>void} [onProgress] called after each grid point
 * @returns {Generator<void, Object>} result { param, values, series: [{metric, method, pts:[[x, y, se]]}] }
 */
export function* runSweep(c, spec, onProgress) {
  const mcMetrics = spec.metrics.filter((m) => METRIC[m]?.kind === 'mc'), series = new Map();
  const key = (m, id) => m + '|' + (id || '');
  const res = { param: spec.param, values: spec.values, series: [] };
  for (const x of spec.values) {
    const cc = { ...c, [spec.param]: x };
    if (spec.param === 'K' && cc.sizeMode === 'paper') cc.sizeMode = 'lognormal';
    if (spec.param === 'K' && cc.sizeMode === 'table') cc.sizeMode = 'lognormal';
    if (spec.param === 'anchorErr' || spec.param === 'anchorSd') cc.anchor = true;
    if (spec.param === 'nBad' || spec.param === 'lie') cc.nBad = Math.max(cc.nBad, spec.param === 'nBad' ? x : 1);
    if (spec.param === 'tauR2' || spec.param === 'tauW2') cc.regional = true;
    for (const id of spec.methods) { if (id === 'robust') cc.robustT = true; if (id === 'optout') cc.optOut = true; if (id === 'screen') cc.screen = true; if (id === 'gmrf') cc.gmrf = true; }
    const ctx = prepare(cc);
    for (const m of spec.metrics.filter((k) => METRIC[k]?.kind === 'closed')) {
      const k2 = key(m); if (!series.has(k2)) series.set(k2, { metric: m, method: null, pts: [] });
      series.get(k2).pts.push([x, closedMetric(ctx, m), 0]);
    }
    if (mcMetrics.length) {
      const acc = makeAccumulator(ctx), rng = makeRng(cc.seed * 104729 + 1);
      while (acc.R < spec.R) { for (let i = 0; i < cc.chunk && acc.R < spec.R; i++) accumulate(ctx, acc, replicate(ctx, rng, acc.R)); yield; }
      const s = summarize(ctx, acc);
      for (const m of mcMetrics) {
        if (!METRIC[m].perMethod) {
          const k2 = key(m); if (!series.has(k2)) series.set(k2, { metric: m, method: null, pts: [] });
          series.get(k2).pts.push([x, s.diagnostics.t2hat, 0]); continue;
        }
        for (const id of spec.methods) {
          const a = s.agg[id]; if (!a) continue;
          const k2 = key(m, id); if (!series.has(k2)) series.set(k2, { metric: m, method: id, pts: [] });
          const val = m === 'mean' ? [a.mean, a.meanSE] : m === 'worst' ? [a.worst, a.worstSE] : m === 'honest' ? [a.honest, a.honestSE]
            : m === 'share' ? [a.share, 0] : [1 / a.y[0], a.se[0] / (a.y[0] ** 2)];
          series.get(k2).pts.push([x, val[0], val[1]]);
        }
      }
    } else yield;
    res.series = [...series.values()];
    onProgress?.(res, x === spec.values[spec.values.length - 1]);
  }
  return res;
}

/**
 * The paper's simulation figures, each as a sweep on the current configuration.
 * @see Section 8.7, Table 6 and the online supplement
 */
export const PAPER_FIGURES = [
  { id: 'tau2', title: 'Relative risk vs heterogeneity τ²', sim: 'Simulation 2', param: 'tau2', start: 1e-4, stop: 1, steps: 9, log: true, metrics: ['mean', 'worst'], methods: ['ghb', 'oracle', 'pooled'], R: 600, logY: true },
  { id: 'K', title: 'Relative risk vs number of hospitals K', sim: 'Simulation 3', param: 'K', start: 3, stop: 48, steps: 8, log: true, metrics: ['mean'], methods: ['ghb', 'oracle'], R: 600 },
  { id: 'T', title: 'Relative risk vs gossip rounds T', sim: 'Simulation 4', param: 'T', start: 1, stop: 160, steps: 9, log: true, metrics: ['mean'], methods: ['ghb'], R: 500 },
  { id: 'jointEps', title: 'Borrowed patients B vs ε', sim: 'Simulation 5', param: 'eps', start: 0.03, stop: 3, steps: 12, log: true, metrics: ['Bmean', 'Bjoint'], methods: [], R: 0, logY: true },
  { id: 'gmrf', title: 'GMRF vs exchangeable GHB vs τ_r²', sim: 'Simulation 6', param: 'tauR2', start: 3e-4, stop: 0.3, steps: 7, log: true, metrics: ['mean'], methods: ['ghb', 'gmrf'], R: 300 },
  { id: 'anchor', title: 'Relative risk vs anchor error', sim: 'Simulation 11 (Gaussian analogue)', param: 'anchorErr', start: 0, stop: 1, steps: 8, log: false, metrics: ['mean'], methods: ['ghb'], R: 500 },
  { id: 'lie', title: 'Honest hospitals risk vs lie size b', sim: 'Simulation 12', param: 'lie', start: 0, stop: 30, steps: 9, log: false, metrics: ['honest'], methods: ['ghb', 'screen'], R: 400 },
  { id: 'pFail', title: 'Relative risk vs link failure probability', sim: 'Simulation 13', param: 'pFail', start: 0, stop: 0.8, steps: 5, log: false, metrics: ['mean'], methods: ['ghb'], R: 500 },
  { id: 'refits', title: 'Borrowed patients B vs refits r', sim: 'Simulation 14', param: 'refits', start: 1, stop: 50, steps: 10, log: true, metrics: ['Bmean'], methods: [], R: 0, logY: true }
];
