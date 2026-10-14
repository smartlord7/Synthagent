/**
 * sim/arena.js
 * Responsibility: the Literature arena, a Monte Carlo of the paper's logistic-regression world (Section 6.3,
 * Simulations 8, 10 and 15; simulations/logit_world.py and simulations/exp15_literature.py) in which GHB, GHB-lite and
 * every literature method run on the same hospitals at the same (epsilon, delta). Methods are tuned in one of three
 * ways: paper defaults (fixed configurations), oracle (each method's grid evaluated on separate tuning replicates
 * against the true loss, the best then reported on fresh replicates, as Simulation 15 does) or manual. DOM-free:
 * it runs in a module worker (sim/arena.worker.js), on the main thread as a fallback, and headless in the tests.
 * Every stochastic step draws from its own stream, seeded by (seed, replicate, step), so a method's numbers do not
 * depend on which other methods are selected.
 */
import { makeRng, siteSizes, lognormalSizes } from '../math/random.js';
import { topology, metropolisHastings, matPow } from '../math/graph.js';
import { avgOp } from '../math/ghb.js';
import { P, MU, N_TEST, features, scores, sigmoid, fit, testSet, excess, rhoTotal, Public } from '../math/logit.js';
import { ghbLite, ghbJoint, GHB_V4, GHB_V5 } from '../math/ghbLogit.js';
import * as LIT from '../math/literature.js';

const ROUNDS = [1, 2, 3, 5, 10, 30, 100], DROUNDS = [3, 10, 30, 100], ETAS = [0.2, 0.5, 2.0], FT_STEPS = [0, 1, 2, 3, 5, 10, 30, 100];
const cart = (o) => Object.entries(o).reduce((acc, [k, vs]) => acc.flatMap((a) => vs.map((v) => ({ ...a, [k]: v }))), [{}]);
const keyOf = (id, c) => id + (c ? '|' + Object.entries(c).map(([k, v]) => `${k}=${v}`).join('|') : '');

/**
 * Every method of the arena. `hyper` lists each tuning parameter with its grid (the grids of literature.py,
 * personalized.py, real_networks.py and exp15_literature.py) and `def` is the paper-default configuration: the
 * configuration Simulation 15 chose at tau^2 = 0.05, epsilon = 1 for the simulation-world baselines, and the
 * configuration frozen for the realistic arm on the development networks (dev/v4_realistic_configs.json, or the
 * leave-one-network-out choice of Section 7.2) for the personalized methods.
 * @see Section 6.3, Table 6 (Simulation 15 grids); Section 7.2
 */
export const ARENA_METHODS = [
  { id: 'local', label: 'Local', short: 'Local', family: 'reference', tok: '--rd-local', note: 'MAP with the published anchor as prior N(μ₀, I); no other hospital.' },
  { id: 'pooled', label: 'Pooled, no DP (reference)', short: 'Pooled', family: 'reference', tok: '--rd-risk', private: false, note: 'One model on all patients without privacy: a reference no private method can reach.' },
  { id: 'ghb', label: 'GHB', short: 'GHB', family: 'ghb', tok: '--rd-ghb', cite: 'this paper, Section 5.2', note: 'GHB as released (v5): two joint releases (score and information sums), leave-one-out prior with privacy covariance, τ integrated out against own evidence, no-borrowing components. The intercept is not borrowed (base rates differ across hospitals); the information release has the tight sensitivity √2 G²/4 and is diagonal only when a public rule says so.',
    settings: { 'quantiles Q': 12, 'half-Cauchy scale': 0.5, 'default scales s': '0.5, 1, 2.5', 'no-borrowing weight': '1/4', 'Newton rounds': 1, 'budget split': '1/2, 1/2', 'intercept': 'not borrowed, prior sd 2.5', 'information sensitivity': '√2 G²/4', 'diagonal release': 'when s_J √P > N G²/(4P)' } },
  { id: 'ghbV4', label: 'GHB v4 (earlier version)', short: 'GHB v4', family: 'ghb', tok: '--rd-gmrf', cite: 'this paper, Section 5.2', note: 'The first released configuration, kept for comparison: the intercept is borrowed like the slopes and the information release has sensitivity G²/2, always as a full matrix.',
    settings: { 'quantiles Q': 12, 'half-Cauchy scale': 0.5, 'default scales s': '0.5, 1, 2.5', 'no-borrowing weight': '1/4', 'Newton rounds': 1, 'budget split': '1/2, 1/2', 'intercept': 'borrowed', 'information sensitivity': 'G²/2', 'diagonal release': 'never' } },
  { id: 'lite', label: 'GHB-lite', short: 'GHB-lite', family: 'ghb', tok: '--rd-lite', cite: 'this paper, Section 5.3', note: 'Each hospital releases its mean score at the anchor; moment inversion, gossip (T = 60), τ from the releases, integrated out.',
    settings: { 'quantiles Q': 12, 'half-Cauchy scale': 0.5, 'default scale': 2.5, 'gossip rounds T': 60, 'public draws': 20000, 'conflict threshold': 3 } },
  { id: 'liteOracle', label: 'GHB-lite, true τ² (reference)', short: 'GHB-lite, τ² known', family: 'reference', tok: '--rd-oracle', note: 'GHB-lite told the true heterogeneity: a reference, not a method.' },
  { id: 'fedavg1', label: 'One-shot FedAvg', short: 'FedAvg 1-shot', cite: 'McMahan et al., 2017', family: 'oneshot', tok: '--rd-noise', note: 'Sample-size weighted mean of GHB-lite\'s releases.' },
  { id: 'meta', label: 'Random-effects meta-analysis', short: 'Meta-analysis', cite: 'DerSimonian & Laird, 1986', family: 'oneshot', tok: '--rd-ref2', note: 'DerSimonian-Laird pooled mean of the same releases.' },
  { id: 'dittoStyle', label: 'Ditto-style, one-shot', short: 'Ditto-style', cite: 'Li et al., 2021', family: 'oneshot', tok: '--rd-ref', note: 'Local fit shrunk toward the one-shot FedAvg with proximal weight λ.',
    hyper: [{ k: 'lam', label: 'λ', grid: [0.3, 1, 3, 10, 30, 100] }], def: { lam: 10 } },
  { id: 'dpfedavg', label: 'DP-FedAvg', short: 'DP-FedAvg', cite: 'McMahan et al., 2018', family: 'federated', tok: '--rd-m1',
    hyper: [{ k: 'R', label: 'rounds R', grid: ROUNDS, int: true }, { k: 'eta', label: 'step η', grid: ETAS }], def: { R: 10, eta: 0.5 } },
  { id: 'dpfedavgFT', label: 'DP-FedAvg + fine-tuning', short: 'DP-FedAvg + FT', cite: 'McMahan et al., 2018', family: 'federated', tok: '--rd-m2',
    hyper: [{ k: 'R', label: 'rounds R', grid: ROUNDS, int: true }, { k: 'eta', label: 'step η', grid: ETAS }, { k: 'E', label: 'fine-tuning steps E', grid: FT_STEPS, int: true }], def: { R: 10, eta: 0.5, E: 5 } },
  { id: 'dpfedavgSA', label: 'DP-FedAvg-SA + fine-tuning', short: 'DP-FedAvg-SA + FT', cite: 'McMahan et al., 2018; Bonawitz et al., 2017', family: 'federated', tok: '--rd-m3', note: 'Secure aggregation: noise on the summed gradient only.',
    hyper: [{ k: 'R', label: 'rounds R', grid: [3, 10, 30], int: true }, { k: 'eta', label: 'step η', grid: [0.5, 2.0] }, { k: 'E', label: 'fine-tuning steps E', grid: [0, 5, 30], int: true }], def: { R: 3, eta: 0.5, E: 5 } },
  { id: 'dec', label: 'Decentralized DP-SGD', short: 'Decentralized', cite: 'Lian et al., 2017', family: 'federated', tok: '--rd-m4', note: 'No server: local noisy steps, neighbour averaging on the agreement graph.',
    hyper: [{ k: 'R', label: 'rounds R', grid: DROUNDS, int: true }, { k: 'eta', label: 'step η', grid: ETAS }, { k: 'E', label: 'fine-tuning steps E', grid: FT_STEPS, int: true }], def: { R: 3, eta: 0.2, E: 5 } },
  { id: 'clu', label: 'Clustered DP-FedAvg', short: 'Clustered', cite: 'Ghosh et al., 2020', family: 'federated', tok: '--rd-m5', note: 'IFCA-style: clusters from one private release (half the budget), DP-FedAvg within each.',
    hyper: [{ k: 'C', label: 'clusters C', grid: [2, 3, 4], int: true }, { k: 'R', label: 'rounds R', grid: ROUNDS, int: true }, { k: 'eta', label: 'step η', grid: ETAS }, { k: 'E', label: 'fine-tuning steps E', grid: FT_STEPS, int: true }], def: { C: 2, R: 5, eta: 0.2, E: 5 } },
  { id: 'lgd', label: 'Local GD from anchor (control)', short: 'Local GD', family: 'reference', tok: '--rd-muted', note: 'E gradient steps on own data from the anchor: no other hospital.',
    hyper: [{ k: 'E', label: 'steps E', grid: [1, 2, 3, 5, 10, 30, 100], int: true }], def: { E: 5 } },
  { id: 'ditto', label: 'Ditto', short: 'Ditto', cite: 'Li et al., 2021', family: 'personalized', tok: '--rd-p1',
    hyper: [{ k: 'R', label: 'rounds R', grid: [3, 10, 30], int: true }, { k: 'eta', label: 'step η', grid: [0.5, 2.0] }, { k: 'lam', label: 'proximal λ', grid: [0.1, 1, 10] }], def: { R: 3, eta: 0.5, lam: 1 } },
  { id: 'pfedme', label: 'pFedMe', short: 'pFedMe', cite: 'T. Dinh et al., 2020', family: 'personalized', tok: '--rd-p2',
    hyper: [{ k: 'R', label: 'rounds R', grid: [3, 10, 30], int: true }, { k: 'eta', label: 'step η', grid: [0.5, 2.0] }, { k: 'lam', label: 'proximal λ', grid: [1, 10] }], def: { R: 3, eta: 2, lam: 10 } },
  { id: 'apfl', label: 'APFL', short: 'APFL', cite: 'Deng et al., 2020', family: 'personalized', tok: '--rd-p3',
    hyper: [{ k: 'R', label: 'rounds R', grid: [3, 10, 30], int: true }, { k: 'eta', label: 'step η', grid: [0.5, 2.0] }, { k: 'alpha', label: 'mixing α', grid: [0.25, 0.5, 0.75] }], def: { R: 3, eta: 0.5, alpha: 0.75 } },
  { id: 'perfedavg', label: 'Per-FedAvg', short: 'Per-FedAvg', cite: 'Fallah et al., 2020', family: 'personalized', tok: '--rd-p4',
    hyper: [{ k: 'R', label: 'rounds R', grid: [3, 10, 30], int: true }, { k: 'eta', label: 'step η', grid: [0.5, 2.0] }, { k: 'a', label: 'adaptation a', grid: [0.1, 0.5] }], def: { R: 3, eta: 0.5, a: 0.1 } },
  { id: 'fedrep', label: 'FedRep', short: 'FedRep', cite: 'Collins et al., 2021', family: 'personalized', tok: '--rd-p5',
    hyper: [{ k: 'R', label: 'rounds R', grid: [3, 10, 30], int: true }, { k: 'eta', label: 'step η', grid: [0.5, 2.0] }, { k: 'E', label: 'fine-tuning steps E', grid: [0, 5], int: true }], def: { R: 3, eta: 0.5, E: 5 } },
  { id: 'fedprox', label: 'FedProx', short: 'FedProx', cite: 'Li et al., 2020', family: 'personalized', tok: '--rd-p6',
    hyper: [{ k: 'R', label: 'rounds R', grid: [3, 10, 30], int: true }, { k: 'eta', label: 'step η', grid: [0.2, 0.5] }, { k: 'mu', label: 'proximal μ', grid: [0.1, 1] }, { k: 'E', label: 'fine-tuning steps E', grid: [0, 5, 30], int: true }], def: { R: 3, eta: 0.2, mu: 0.1, E: 30 } }
];
export const ARENA_METHOD = Object.fromEntries(ARENA_METHODS.map((m) => [m.id, m]));

/** Method families for the selection buttons. */
export const ARENA_FAMILIES = [
  ['ghb', 'GHB and GHB-lite'], ['oneshot', 'One-shot (Simulation 8)'], ['federated', 'Iterative federated (Simulation 15)'],
  ['personalized', 'Personalized FL (Section 7.2)'], ['reference', 'References and controls']
];

/**
 * Default arena specification: Simulation 15 at tau^2 = 0.05, epsilon = 1 with 24 hospitals.
 * @returns {Object}
 * @see Section 6.3, Simulation 15
 */
export function defaultArenaSpec() {
  return {
    K: 24, median: 100, tau2: 0.05, eps: 1, delta: 1e-5, shift: 0, regions: 0, tauR2: 0.2, anchorErr: 0.3, anchorSd: 0.3, check: true,
    reps: 10, tuneReps: 5, seed: 15, nTest: N_TEST, tuning: 'defaults', methods: ARENA_METHODS.map((m) => m.id), manual: {}
  };
}

/**
 * The configurations of a method under a tuning mode.
 * @param {Object} m method
 * @param {Object} spec arena specification
 * @param {boolean} full whole grid (oracle tuning replicates)
 * @returns {Array<Object|null>} configurations (null for an untuned method)
 */
export function methodConfigs(m, spec, full) {
  if (!m.hyper) return [null];
  if (full) return cart(Object.fromEntries(m.hyper.map((h) => [h.k, h.grid])));
  if (spec.tuning === 'manual' && spec.manual?.[m.id]) return [{ ...m.def, ...spec.manual[m.id] }];
  return [{ ...m.def }];
}

function hash(str) { let h = 2166136261; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

/**
 * Prepare the fixed parts of a specification: sizes, agreement graph and gossip operator.
 * @param {Object} spec arena specification
 * @returns {Object} context
 * @see Section 6.1 (24 hospitals, ring with two-hop links, T = 60)
 */
export function prepareArena(spec) {
  const K = spec.K, n = K <= 96 ? siteSizes(K, spec.median) : lognormalSizes(K, spec.median, 1, 11);
  const W = metropolisHastings(topology(K > 4 ? 'ring2' : 'ring', K)), avg = avgOp(matPow(W, 60), K);
  return { spec, K, n, W, avg, eps: spec.eps, delta: spec.delta, rho: rhoTotal(spec.eps, spec.delta) };
}

/**
 * Draw one world: coefficients theta_k = mu + sqrt(tau^2) z (plus regional effects), case-mix means, the published
 * anchor and every hospital's data.
 * @param {Object} ctx prepared context
 * @param {{n:()=>number, u:()=>number}} rng world stream
 * @returns {{theta:Float64Array[], means:Float64Array[]|null[], mu0:Float64Array, data:Array}}
 * @see Eq. (2) for heads; logit_world.world
 */
export function drawWorld(ctx, rng) {
  const { K, n, spec } = ctx, st = Math.sqrt(spec.tau2);
  const theta = Array.from({ length: K }, () => Float64Array.from(MU, (m) => m + st * rng.n()));
  if (spec.regions > 0) { const reg = Array.from({ length: spec.regions }, () => Float64Array.from({ length: P }, () => Math.sqrt(spec.tauR2) * rng.n())); theta.forEach((t, k) => { for (let a = 0; a < P; a++) t[a] += reg[k % spec.regions][a]; }); }
  const means = Array.from({ length: K }, () => Float64Array.from({ length: P - 1 }, () => spec.shift * rng.n()));
  const mu0 = Float64Array.from(MU, (m) => m + spec.anchorErr * rng.n());
  const data = theta.map((t, k) => {
    const X = features(rng, n[k], spec.shift > 0 ? means[k] : null), s = scores(X, t), y = new Float64Array(n[k]);
    for (let i = 0; i < n[k]; i++) y[i] = rng.u() < sigmoid(s[i]) ? 1 : 0;
    return { X, y, n: n[k] };
  });
  return { theta, means, mu0, data };
}

/**
 * Estimates of the requested configurations of every requested method, for one world.
 * @param {Object} ctx context
 * @param {Object} w world
 * @param {Map<string, Array<Object|null>>} want method id -> configurations
 * @param {(key:string)=>Object} R stream factory
 * @returns {Map<string, Float64Array[]>} candidate key -> per-hospital coefficients
 * @see Section 6.3 (every method on the same hospitals and budget)
 */
export function estimatesFor(ctx, w, want, R) {
  const { K, n, spec, avg, W, rho } = ctx, { data, mu0 } = w, out = new Map(), memo = new Map();
  const once = (k, f) => { if (!memo.has(k)) memo.set(k, f()); return memo.get(k); };
  const asd = spec.anchorSd > 0 ? spec.anchorSd : null;
  const lite = () => once('lite', () => ghbLite(R('lite'), data, n, new Public(features(R('public'), 20000)), mu0, avg,
    { eps: spec.eps, delta: spec.delta, asd, check: spec.check, oracleTau2: want.has('liteOracle') ? spec.tau2 + (spec.regions > 0 ? spec.tauR2 : 0) : null }));
  const groupFT = (id, cfgs, base, startFn) => {
    const groups = new Map();
    for (const c of cfgs) { const g = base(c); if (!groups.has(g)) groups.set(g, { c, Es: new Set() }); groups.get(g).Es.add(c.E ?? 0); }
    for (const [g, { c, Es }] of groups) { const snaps = LIT.fineTune(data, startFn(c, g), [...Es]); for (const cc of cfgs) if (base(cc) === g) out.set(keyOf(id, cc), snaps.get(cc.E ?? 0)); }
  };
  for (const [id, cfgs] of want) {
    switch (id) {
      case 'local': out.set('local', data.map((d) => fit(d, mu0, new Float64Array(P).fill(1)))); break;
      case 'pooled': {
        const N = n.reduce((a, b) => a + b, 0), X = new Float64Array(N * P), y = new Float64Array(N); let o = 0;
        for (const d of data) { X.set(d.X, o * P); y.set(d.y, o); o += d.n; }
        const t = fit({ X, y, n: N }, new Float64Array(P), new Float64Array(P).fill(1e-3)); out.set('pooled', data.map(() => t)); break;
      }
      case 'ghb': out.set('ghb', ghbJoint(R('ghb'), data, spec.eps, spec.delta, { ...GHB_V5, anchor: asd ? { mu0, sd: asd } : null }).est); break;
      case 'ghbV4': out.set('ghbV4', ghbJoint(R('ghbV4'), data, spec.eps, spec.delta, { ...GHB_V4, anchor: asd ? { mu0, sd: asd } : null }).est); break;
      case 'lite': out.set('lite', lite().est); break;
      case 'liteOracle': out.set('liteOracle', lite().oracle); break;
      case 'fedavg1': out.set('fedavg1', LIT.repeat(LIT.oneShotFedavg(lite().inv.th, n), K)); break;
      case 'meta': out.set('meta', LIT.repeat(LIT.metaAnalysis(lite().inv.th, lite().inv.r), K)); break;
      case 'dittoStyle': { const c0 = once('fa1', () => LIT.oneShotFedavg(lite().inv.th, n)); for (const c of cfgs) out.set(keyOf(id, c), LIT.dittoStyle(data, c0, c.lam)); break; }
      case 'dpfedavg': case 'dpfedavgFT':
        groupFT(id, cfgs.map((c) => (id === 'dpfedavg' ? { ...c, E: 0 } : c)), (c) => `fa|${c.R}|${c.eta}`, (c, g) => LIT.repeat(once(g, () => LIT.fedavg(R(g), data, n, mu0, rho, c.R, c.eta)), K));
        if (id === 'dpfedavg') for (const c of cfgs) { const kk = keyOf(id, { ...c, E: 0 }); out.set(keyOf(id, c), out.get(kk)); if (kk !== keyOf(id, c)) out.delete(kk); }
        break;
      case 'dpfedavgSA': groupFT(id, cfgs, (c) => `sa|${c.R}|${c.eta}`, (c, g) => LIT.repeat(once(g, () => LIT.fedavgSA(R(g), data, n, mu0, rho, c.R, c.eta)), K)); break;
      case 'dec': groupFT(id, cfgs, (c) => `dec|${c.R}|${c.eta}`, (c, g) => once(g, () => LIT.decentralized(R(g), data, n, mu0, rho, c.R, c.eta, W))); break;
      case 'clu': groupFT(id, cfgs, (c) => `clu|${c.C}|${c.R}|${c.eta}`, (c, g) => {
        const lab = once(`clulab|${c.C}`, () => LIT.privateClusters(R(`clulab|${c.C}`), data, n, rho / 2, c.C));
        return once(g, () => LIT.clusteredFedavg(R(g), data, n, mu0, rho / 2, lab, c.C, c.R, c.eta));
      }); break;
      case 'lgd': { const snaps = LIT.fineTune(data, LIT.repeat(mu0, K), cfgs.map((c) => c.E)); for (const c of cfgs) out.set(keyOf(id, c), snaps.get(c.E)); break; }
      case 'ditto': for (const c of cfgs) out.set(keyOf(id, c), LIT.ditto(R(keyOf(id, c)), data, n, rho, c.R, c.eta, c.lam)); break;
      case 'pfedme': for (const c of cfgs) out.set(keyOf(id, c), LIT.pfedme(R(keyOf(id, c)), data, n, rho, c.R, c.eta, c.lam)); break;
      case 'apfl': for (const c of cfgs) out.set(keyOf(id, c), LIT.apfl(R(keyOf(id, c)), data, n, rho, c.R, c.eta, c.alpha)); break;
      case 'perfedavg': for (const c of cfgs) out.set(keyOf(id, c), LIT.perFedavg(R(keyOf(id, c)), data, n, rho, c.R, c.eta, c.a)); break;
      case 'fedrep': { const by = new Map(); for (const c of cfgs) { const g = `rep|${c.R}|${c.eta}`; if (!by.has(g)) by.set(g, LIT.fedrep(R(g), data, n, rho, c.R, c.eta, [0, 5, ...cfgs.map((x) => x.E)].filter((v, i, a) => a.indexOf(v) === i))); out.set(keyOf(id, c), by.get(g).get(c.E)); } break; }
      case 'fedprox': { const by = new Map(); for (const c of cfgs) { const g = `prox|${c.R}|${c.eta}|${c.mu}`; if (!by.has(g)) by.set(g, LIT.fedprox(R(g), data, n, rho, c.R, c.eta, c.mu, 5, [0, 5, 30, ...cfgs.map((x) => x.E)].filter((v, i, a) => a.indexOf(v) === i))); out.set(keyOf(id, c), by.get(g).get(c.E)); } break; }
      default: throw new Error('unknown arena method ' + id);
    }
  }
  return out;
}

/**
 * One replicate: draw the world and the test sets, compute every requested candidate, evaluate excess log-loss
 * (and AUC) of each at each hospital.
 * @param {Object} ctx context
 * @param {number} rep replicate index (tuning replicates use their own range)
 * @param {Map<string, Array<Object|null>>} want method id -> configurations
 * @param {boolean} withAuc compute AUC
 * @returns {Map<string, {kl:Float64Array, auc:Float64Array}>} candidate key -> per-hospital metrics
 * @see logit_world.one_rep and evaluate
 */
export function arenaReplicate(ctx, rep, want, withAuc) {
  const seed = ctx.spec.seed, R = (key) => makeRng(hash(`${seed}|${rep}|${key}`));
  const w = drawWorld(ctx, R('world'));
  const est = estimatesFor(ctx, w, want, R);
  const tr = R('test'), tests = w.theta.map((t, k) => testSet(tr, t, ctx.spec.shift > 0 ? w.means[k] : null, ctx.spec.nTest));
  const res = new Map();
  for (const [key, e] of est) {
    const kl = new Float64Array(ctx.K), au = new Float64Array(ctx.K);
    for (let k = 0; k < ctx.K; k++) { const r = excess(tests[k], e[k], withAuc); kl[k] = r.kl; au[k] = r.auc; }
    res.set(key, { kl, auc: au });
  }
  return res;
}

/**
 * Summary of a method's replicates.
 * @param {Float64Array[]} KL per replicate, per hospital excess log-loss
 * @param {Float64Array[]} AU per replicate, per hospital AUC
 * @param {Float64Array} loc per-hospital mean excess log-loss of local estimation
 * @param {number[]} n sizes
 * @returns {Object} mean, se, auc, worst, share, small, large, perHospital
 * @see exp15_literature.setting (kl, kl_se, auc, worst_ratio, share_better, kl_small, kl_large)
 */
export function summarizeMethod(KL, AU, loc, n) {
  const Rn = KL.length, K = n.length, per = new Float64Array(K), perAuc = new Float64Array(K), repMean = KL.map((r) => r.reduce((a, b) => a + b, 0) / K);
  KL.forEach((r) => r.forEach((v, k) => { per[k] += v / Rn; }));
  AU.forEach((r) => r.forEach((v, k) => { perAuc[k] += v / Rn; }));
  const mean = repMean.reduce((a, b) => a + b, 0) / Rn, sd = Rn > 1 ? Math.sqrt(repMean.reduce((a, b) => a + (b - mean) ** 2, 0) / (Rn - 1)) : NaN;
  const order = n.map((_, k) => k).sort((a, b) => n[a] - n[b]), third = Math.floor(K / 3);
  const avgOver = (idx) => idx.reduce((a, k) => a + per[k], 0) / idx.length;
  const ratio = Array.from(per, (v, k) => v / loc[k]);
  return {
    mean, se: sd / Math.sqrt(Rn), auc: perAuc.reduce((a, b) => a + b, 0) / K, worst: Math.max(...ratio), worstK: ratio.indexOf(Math.max(...ratio)),
    share: ratio.filter((r) => r < 1).length / K, small: avgOver(order.slice(0, third)), large: avgOver(order.slice(K - third)), perHospital: Array.from(per), perAuc: Array.from(perAuc), ratio
  };
}

/**
 * Run the arena as a generator: optional oracle tuning on separate replicates, then evaluation. Yields progress
 * objects after every replicate; returns the full result.
 * @param {Object} spec arena specification (defaultArenaSpec)
 * @returns {Generator<{phase:string, done:number, total:number}, Object>}
 * @see Section 6.3 (Simulation 15 tuning protocol)
 */
export function* runArena(spec) {
  const ctx = prepareArena(spec), t0 = Date.now();
  const methods = ARENA_METHODS.filter((m) => spec.methods.includes(m.id) || m.id === 'local');
  const chosen = {}, tuneScores = {};
  const totalTune = spec.tuning === 'oracle' ? spec.tuneReps : 0, total = totalTune + spec.reps;
  if (spec.tuning === 'oracle') {
    const tunable = methods.filter((m) => m.hyper), want = new Map(tunable.map((m) => [m.id, methodConfigs(m, spec, true)])), acc = new Map();
    for (let t = 0; t < spec.tuneReps; t++) {
      const res = arenaReplicate(ctx, 1e6 + t, want, false);
      for (const [key, r] of res) acc.set(key, (acc.get(key) || 0) + r.kl.reduce((a, b) => a + b, 0) / ctx.K / spec.tuneReps);
      yield { phase: 'tuning', done: t + 1, total, candidates: res.size };
    }
    for (const m of tunable) {
      const cands = want.get(m.id).map((c) => [c, acc.get(keyOf(m.id, c))]);
      cands.sort((a, b) => a[1] - b[1]);
      chosen[m.id] = cands[0][0]; tuneScores[m.id] = cands.map(([c, v]) => ({ config: c, kl: v }));
    }
  }
  const want = new Map(methods.map((m) => [m.id, spec.tuning === 'oracle' && m.hyper ? [chosen[m.id]] : methodConfigs(m, spec, false)]));
  for (const m of methods) if (m.hyper && !chosen[m.id]) chosen[m.id] = want.get(m.id)[0];
  const KL = Object.fromEntries(methods.map((m) => [m.id, []])), AU = Object.fromEntries(methods.map((m) => [m.id, []]));
  for (let r = 0; r < spec.reps; r++) {
    const res = arenaReplicate(ctx, r, want, true);
    for (const m of methods) { const x = res.get(keyOf(m.id, want.get(m.id)[0])); KL[m.id].push(x.kl); AU[m.id].push(x.auc); }
    yield { phase: 'evaluation', done: totalTune + r + 1, total, partial: r + 1 };
  }
  const loc = Float64Array.from({ length: ctx.K }, (_, k) => KL.local.reduce((a, rr) => a + rr[k], 0) / spec.reps);
  const summary = Object.fromEntries(methods.map((m) => [m.id, { ...summarizeMethod(KL[m.id], AU[m.id], loc, ctx.n), config: chosen[m.id] || null }]));
  return { spec: { ...spec }, n: ctx.n, K: ctx.K, rho: ctx.rho, summary, tuneScores, seconds: (Date.now() - t0) / 1000 };
}

/**
 * Drive the generator to completion (headless use: tests, worker).
 * @param {Object} spec arena specification
 * @param {(p:Object)=>void} [onProgress] progress callback
 * @returns {Object} result
 */
export function runArenaSync(spec, onProgress) {
  const g = runArena(spec); let r = g.next();
  while (!r.done) { onProgress?.(r.value); r = g.next(); }
  return r.value;
}

export { keyOf };
