/**
 * config.js
 * Responsibility: the single configuration object of the lab. Declares every parameter once (schema with
 * label, unit, range, scale and paper default), and resolves a configuration into the concrete objects the
 * simulations need (sizes, graph, privacy parameters). Controls, presets, sweeps and JSON export all read
 * this schema, so no parameter is hard-coded elsewhere.
 */
import { PAPER_SIZES, lognormalSizes, shuffle, makeRng } from './math/random.js';
import { topology, edgesToAdj, zeros } from './math/graph.js';
import { zcdpRho } from './math/dp.js';

const R = (key, group, label, unit, min, max, step, def, extra = {}) => ({ key, group, label, unit, type: 'range', min, max, step, def, ...extra });
const S = (key, group, label, options, def, extra = {}) => ({ key, group, label, type: 'select', options, def, ...extra });
const C = (key, group, label, def, extra = {}) => ({ key, group, label, type: 'check', def, ...extra });

/** Parameter groups in display order. */
export const GROUPS = [
  ['network', 'Network'], ['model', 'Model'], ['privacy', 'Privacy'], ['estimation', 'Estimation'],
  ['safeguards', 'Safeguards'], ['adversaries', 'Adversaries'], ['mc', 'Monte Carlo'], ['calc', 'Small-data calculator']
];

/**
 * Every parameter of the lab. `log: true` means the slider moves in log10 units; `sweep: true` marks
 * parameters offered on the Sweeps tab. Defaults are the paper's (Section 8.7, Example 2.4).
 */
export const PARAMS = [
  // network
  R('K', 'network', 'Hospitals K', '', 3, 48, 1, 24, { sweep: true, int: true }),
  S('sizeMode', 'network', 'Hospital sizes', [['paper', 'Paper profile (site_sizes)'], ['lognormal', 'Log-normal draw'], ['table', 'Editable table']], 'paper'),
  R('median', 'network', 'Median size', 'patients', 10, 1000, 5, 100, { sweep: true }),
  R('spread', 'network', 'Log-normal spread', 'sd log n', 0.1, 2, 0.05, 1, { sweep: true }),
  R('nMin', 'network', 'Smallest size', 'patients', 2, 200, 1, 15, { int: true }),
  R('nMax', 'network', 'Largest size', 'patients', 100, 5000, 50, 1500, { int: true }),
  R('sizeSeed', 'network', 'Size seed', '', 1, 99, 1, 11, { int: true }),
  S('topo', 'network', 'Topology', [['ring', 'Ring'], ['ring2', 'Ring + 2-hop'], ['regional', 'Regional clusters + bridges'], ['random', 'Sparse random'], ['custom', 'Custom (edit on graph)']], 'ring2'),
  R('hops', 'network', 'Ring hops', 'hops', 1, 6, 1, 2, { int: true }),
  R('clusters', 'network', 'Clusters', '', 2, 8, 1, 4, { int: true }),
  R('pIn', 'network', 'Within-cluster edge prob. p_in', '', 0.1, 1, 0.05, 0.8),
  R('pER', 'network', 'Random edge prob. p', '', 0.05, 0.6, 0.01, 0.15, { sweep: true }),
  R('graphSeed', 'network', 'Graph seed (0 = paper graph)', '', 0, 30, 1, 0, { int: true }),
  R('pFail', 'network', 'Link failure prob. per round', '', 0, 0.9, 0.05, 0, { sweep: true }),
  // model
  R('mu', 'model', 'Population mean μ', '', -1, 1, 0.05, 0),
  R('tau2', 'model', 'Heterogeneity τ²', '', -4, 0, 0.05, 0.02, { log: true, sweep: true }),
  R('sigma2', 'model', 'Per-patient variance σ²', '', 0.25, 4, 0.05, 1, { sweep: true }),
  S('effects', 'model', 'Site effects u_k', [['gauss', 'Gaussian'], ['t', 'Student-t (same variance)'], ['outlier', 'Gaussian + one outlier']], 'gauss'),
  R('nu', 'model', 'Student-t degrees of freedom ν', '', 2.5, 30, 0.5, 3, { sweep: true }),
  R('outK', 'model', 'Outlier at k·τ, k', 'τ', 1, 12, 0.5, 6, { sweep: true }),
  S('outAt', 'model', 'Outlier hospital', [['smallest', 'Smallest'], ['largest', 'Largest'], ['median', 'Median-sized']], 'smallest'),
  C('regional', 'model', 'Regional structure θ = μ + r_region + w', false),
  R('regions', 'model', 'Regions', '', 2, 8, 1, 4, { int: true }),
  R('tauR2', 'model', 'Between-region τ_r²', '', -3.5, -0.5, 0.05, 0.02, { log: true, sweep: true }),
  R('tauW2', 'model', 'Within-region τ_w²', '', -4, -1, 0.05, 0.002, { log: true, sweep: true }),
  // privacy
  C('noDP', 'privacy', 'No differential privacy', false),
  R('eps', 'privacy', 'Privacy budget ε', '', -1.5, 1, 0.05, 1, { log: true, sweep: true }),
  R('delta', 'privacy', 'δ', '', -8, -3, 0.5, 1e-5, { log: true }),
  R('C', 'privacy', 'Clip norm C', '', 0.1, 5, 0.1, 1, { sweep: true }),
  S('accounting', 'privacy', 'Accounting over refits', [['basic', 'Basic composition'], ['zcdp', 'Zero-concentrated DP']], 'basic'),
  R('refits', 'privacy', 'Refits r on the same patients', '', 1, 50, 1, 1, { int: true, sweep: true }),
  R('group', 'privacy', 'Joint-release group size g', '', 1, 8, 1, 1, { int: true, sweep: true }),
  // estimation
  S('tauEst', 'estimation', 'τ² estimator', [['dl', 'DerSimonian-Laird'], ['bayes', 'Half-Cauchy posterior'], ['oracle', 'Oracle (true τ²)']], 'dl'),
  R('cauchy', 'estimation', 'Half-Cauchy scale', '', 0.05, 2, 0.05, 0.5),
  S('weights', 'estimation', 'Leave-one-out weights', [['iv', 'Inverse variance'], ['equal', 'Equal']], 'iv'),
  C('exact', 'estimation', 'Exact averaging (server reference)', false),
  R('T', 'estimation', 'Gossip rounds T', 'rounds', 1, 200, 1, 60, { int: true, sweep: true }),
  R('gammaTol', 'estimation', 'Consensus tolerance γ', '', -6, -1, 0.5, 0.01, { log: true }),
  R('beta', 'estimation', 'GMRF smoothing β', '', 0, 4, 0.05, 100, { log: true, sweep: true }),
  R('gamma', 'estimation', 'GMRF anchor strength γ', '', 0, 1000, 1, 0, { sweep: true }),
  // safeguards
  C('optOut', 'safeguards', 'Opt-out rule', true),
  R('optZ', 'safeguards', 'Opt-out threshold', 'sd', 1, 6, 0.1, 3),
  C('robustT', 'safeguards', 'Robust Student-t prior', true),
  R('robustNu', 'safeguards', 'Robust prior ν', '', 2.5, 30, 0.5, 3),
  C('screen', 'safeguards', 'Release screening', true),
  R('screenZ', 'safeguards', 'Screening threshold', 'sd', 1, 6, 0.1, 3),
  C('gmrf', 'safeguards', 'GMRF prior (regional similarity)', false),
  C('anchor', 'safeguards', 'Knowledge anchor μ₀', false),
  R('anchorErr', 'safeguards', 'Anchor error (true sd)', '', 0, 1, 0.01, 0.1, { sweep: true }),
  R('anchorSd', 'safeguards', 'Anchor stated sd σ₀', '', 0.01, 1, 0.01, 0.1, { sweep: true }),
  C('anchorCheck', 'safeguards', 'Anchor conflict check', true),
  // adversaries
  R('nBad', 'adversaries', 'Dishonest hospitals f', '', 0, 8, 1, 0, { int: true, sweep: true }),
  S('badWhich', 'adversaries', 'Which hospitals lie', [['largest', 'The f largest'], ['smallest', 'The f smallest'], ['random', 'Random (seeded)']], 'largest'),
  R('lie', 'adversaries', 'Lie size b (release μ + b·τ)', 'τ', 0, 30, 0.5, 8, { sweep: true }),
  // Monte Carlo
  R('R', 'mc', 'Replicates', '', 200, 20000, 100, 3000, { int: true }),
  R('chunk', 'mc', 'Replicates per frame', '', 10, 500, 10, 60, { int: true }),
  R('seed', 'mc', 'Monte Carlo seed', '', 1, 999, 1, 7, { int: true }),
  R('refN', 'mc', 'Reference hospital size (closed forms)', 'patients', 5, 1000, 5, 50, { int: true }),
  // calculator (Section 2)
  R('cN', 'calc', 'Records n', 'records', 50, 20000, 50, 2000, { int: true }),
  R('cM', 'calc', 'Records per patient m̄', '', 1, 6, 0.1, 1.6),
  R('cICC', 'calc', 'Intra-class correlation', '', 0, 1, 0.01, 0.5),
  R('cRhoL', 'calc', 'Labeled fraction ρ_L', '', 0.05, 1, 0.01, 0.7),
  R('cEta', 'calc', 'Label noise η', '', 0, 0.45, 0.01, 0.1),
  R('cRhoS', 'calc', 'Kish shift efficiency ρ_S', '', 0.05, 1, 0.01, 0.6),
  R('cPrev', 'calc', 'Outcome prevalence π', '', 0.005, 0.5, 0.005, 0.08),
  R('cP', 'calc', 'Candidate parameters p', '', 2, 100, 1, 20, { int: true }),
  R('cR2', 'calc', 'R²_CS as share of max', '%', 0.02, 0.6, 0.01, 0.15),
  R('cS', 'calc', 'Target shrinkage S', '', 0.8, 0.98, 0.01, 0.9)
];

/** Lookup of parameters by key. */
export const PARAM = Object.fromEntries(PARAMS.map((p) => [p.key, p]));

/**
 * The paper defaults as a fresh configuration object.
 * @returns {Object} configuration with every key of PARAMS plus `sizesTable` and `customEdges`
 * @see Section 8.7 (defaults of the simulation study), Example 2.4
 */
export function defaultConfig() {
  const c = Object.fromEntries(PARAMS.map((p) => [p.key, p.def]));
  c.sizesTable = PAPER_SIZES.slice();
  c.customEdges = null;
  return c;
}

/**
 * Normalize an imported or edited configuration: unknown keys dropped, values clamped to their ranges.
 * @param {Object} raw candidate configuration (e.g. parsed JSON)
 * @returns {{config:Object, warnings:string[]}}
 */
export function normalizeConfig(raw) {
  const c = defaultConfig(), warnings = [];
  for (const [k, v] of Object.entries(raw || {})) {
    const p = PARAM[k];
    if (k === 'sizesTable' && Array.isArray(v)) { c.sizesTable = v.map(Number).filter((x) => x >= 2).map(Math.round); continue; }
    if (k === 'customEdges') { c.customEdges = Array.isArray(v) ? v.filter((e) => Array.isArray(e) && e.length === 2).map((e) => e.map(Number)) : null; continue; }
    if (!p) { warnings.push(`Ignored unknown key "${k}".`); continue; }
    if (p.type === 'check') c[k] = !!v;
    else if (p.type === 'select') { if (p.options.some((o) => o[0] === v)) c[k] = v; else warnings.push(`"${k}" must be one of ${p.options.map((o) => o[0]).join(', ')}.`); }
    else {
      let x = Number(v);
      if (!isFinite(x)) { warnings.push(`"${k}" is not a number.`); continue; }
      const lo = p.log ? 10 ** p.min : p.min, hi = p.log ? 10 ** p.max : p.max;
      if (x < lo || x > hi) { warnings.push(`"${k}" clamped to [${lo}, ${hi}].`); x = Math.min(hi, Math.max(lo, x)); }
      c[k] = p.int ? Math.round(x) : x;
    }
  }
  return { config: c, warnings };
}

/**
 * Hospital sizes for a configuration, ascending. In 'paper' mode K is fixed at 24.
 * @param {Object} c configuration
 * @returns {number[]}
 * @see simulations/ghb.py site_sizes()
 */
export function resolveSizes(c) {
  if (c.sizeMode === 'paper') return PAPER_SIZES.slice();
  if (c.sizeMode === 'table') return c.sizesTable.slice(0, Math.max(3, c.sizesTable.length)).map((x) => Math.max(2, Math.round(x)));
  return lognormalSizes(c.K, c.median, c.spread, c.sizeSeed, c.nMin, Math.max(c.nMin, c.nMax));
}

/**
 * Collaboration graph adjacency for a configuration. Ring hops override ring/ring2 when set.
 * @param {Object} c configuration
 * @param {number} K number of hospitals
 * @returns {Float64Array[]}
 * @see Section 8.4, Figure 9
 */
export function resolveGraph(c, K) {
  if (c.topo === 'custom') return c.customEdges ? edgesToAdj(K, c.customEdges.filter(([i, j]) => i < K && j < K && i !== j)) : topology('ring', K);
  if (c.topo === 'ring' || c.topo === 'ring2') {
    const h = c.topo === 'ring' ? 1 : c.hops;
    if (h <= 2) return topology(h === 1 ? 'ring' : 'ring2', K);
    const A = zeros(K);
    for (let i = 0; i < K; i++) for (let s = 1; s <= h; s++) { const j = (i + s) % K; if (j !== i) A[i][j] = A[j][i] = 1; }
    return A;
  }
  return topology(c.topo, K, c.pER, c.graphSeed, c.clusters, c.pIn);
}

/**
 * Region of each hospital (seeded permutation so sizes mix across regions, as in exp6_gmrf.py).
 * @param {Object} c configuration
 * @param {number} K hospitals
 * @returns {number[]} region index per hospital
 * @see Simulation 6
 */
export function resolveRegions(c, K) {
  const per = Math.max(1, Math.floor(K / c.regions)), order = shuffle(Array.from({ length: K }, (_, i) => i), 66), reg = new Array(K);
  order.forEach((k, pos) => { reg[k] = Math.min(c.regions - 1, Math.floor(pos / per)); });
  return reg;
}

/**
 * Privacy parameters of one release after accounting for r refits on the same patients.
 * Under zCDP, the per-release noise is converted to the classical calibration by an effective epsilon.
 * @param {Object} c configuration
 * @returns {{eps:number, delta:number, C:number, rho:number|null, perEps:number, perDelta:number, noiseScale:number}}
 *   eps/delta/C to pass to dpVar for one release (noiseScale multiplies the classical sd under zCDP)
 * @see Proposition 8.2, Simulation 14
 */
export function resolvePrivacy(c) {
  if (c.noDP) return { eps: Infinity, delta: c.delta, C: c.C, rho: null, perEps: Infinity, perDelta: 0, noiseScale: 0 };
  const r = c.refits;
  if (c.accounting === 'zcdp') {
    const rho = zcdpRho(c.eps, c.delta), rhoR = rho / r;
    // sd = Delta / sqrt(2 rho_r); classical sd at (e, delta) is Delta sqrt(2 ln(1.25/delta)) / e, so e = 2 sqrt(rho_r ln(1.25/delta))
    const effEps = 2 * Math.sqrt(rhoR * Math.log(1.25 / c.delta));
    return { eps: effEps, delta: c.delta, C: c.C, rho, perEps: c.eps / r, perDelta: c.delta, noiseScale: 1 };
  }
  return { eps: c.eps / r, delta: c.delta / r, C: c.C, rho: null, perEps: c.eps / r, perDelta: c.delta / r, noiseScale: 1 };
}

/**
 * Indices of the dishonest hospitals.
 * @param {Object} c configuration
 * @param {number[]} n sizes (ascending)
 * @returns {number[]}
 * @see Simulation 12
 */
export function resolveAdversaries(c, n) {
  const K = n.length, f = Math.min(c.nBad, K - 2);
  if (f <= 0) return [];
  if (c.badWhich === 'smallest') return Array.from({ length: f }, (_, i) => i);
  if (c.badWhich === 'random') return shuffle(Array.from({ length: K }, (_, i) => i), 4242).slice(0, f).sort((a, b) => a - b);
  return Array.from({ length: f }, (_, i) => K - f + i);
}

/**
 * Random configuration within plausible ranges (the "Randomize" button).
 * @param {Object} base configuration to start from
 * @param {number} seed generator seed
 * @returns {Object}
 */
export function randomConfig(base, seed) {
  const r = makeRng(seed), c = { ...base }, pick = (a) => a[Math.floor(r.u() * a.length)];
  c.sizeMode = 'lognormal'; c.K = 6 + Math.floor(r.u() * 37); c.median = Math.round(20 + r.u() * 300); c.spread = +(0.3 + r.u() * 1.4).toFixed(2);
  c.sizeSeed = 1 + Math.floor(r.u() * 98); c.topo = pick(['ring', 'ring2', 'regional', 'random']); c.pER = +(0.1 + r.u() * 0.3).toFixed(2);
  c.graphSeed = 1 + Math.floor(r.u() * 29); c.tau2 = 10 ** (-3 + r.u() * 2.5); c.eps = 10 ** (-1 + r.u() * 1.5);
  c.T = 10 + Math.floor(r.u() * 120); c.effects = pick(['gauss', 'gauss', 't', 'outlier']);
  return c;
}
