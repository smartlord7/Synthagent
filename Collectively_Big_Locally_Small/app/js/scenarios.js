/**
 * scenarios.js
 * Responsibility: preset scenarios. "Paper defaults" is the design of Section 6.1; Simulations 1 to 14
 * follow Table 2 of the paper; two presets compare GHB with GHB-lite; the arena presets reproduce the logistic
 * Simulations 8, 10 and 15. Each preset is a set of overrides on the defaults, the tab it opens and the
 * paper figure (sim/sweep.js PAPER_FIGURES) it reproduces. Simulations 7, 8, 10 and 11 use logistic heads in
 * the paper; here they run as Gaussian analogues and say so.
 */
import { defaultConfig } from './config.js';

/** @type {Array<{id:string, title:string, question:string, tab:string, figure?:string, overrides:Object, note?:string}>} */
export const SCENARIOS = [
  { id: 'paper', title: 'Paper defaults', question: '24 hospitals, τ² = 0.02, ε = 1, 60 gossip rounds on the ring with two-hop links.', tab: 'borrow', overrides: {} },
  { id: 'ghbVsLite', title: 'GHB vs GHB-lite: strong privacy, small hospitals', question: 'Which protocol borrows more when per-hospital releases are drowned in noise?', tab: 'borrow',
    overrides: { sizeMode: 'lognormal', median: 30, spread: 1, nMin: 5, eps: 0.1, protocol: 'both' }, note: 'At ε = 0.1 each small hospital\'s own release is mostly noise; GHB\'s noise is calibrated to the network sum (Section 5.2).' },
  { id: 'liteWins', title: 'GHB vs GHB-lite: large hospitals, weak privacy', question: 'When are per-hospital releases worth their noise?', tab: 'borrow',
    overrides: { sizeMode: 'lognormal', K: 48, median: 400, spread: 1, nMax: 5000, eps: 10, protocol: 'both' }, note: 'Many informative releases let GHB-lite learn τ from their spread, which GHB cannot (Section 5.3).' },
  { id: 'S1', title: 'Simulation 1: closed forms', question: 'Do Propositions 4.1 and 4.2 match simulation?', tab: 'borrow',
    overrides: { sizeMode: 'table', sizesTable: [15, 30, 60, 120, 200, 200, 200, 200, 200, 400, 800, 1600], K: 12, exact: true, tauEst: 'oracle', weights: 'equal', optOut: false, robustT: false, screen: false } },
  { id: 'S2', title: 'Simulation 2: heterogeneity', question: 'Is plug-in GHB-lite ever worse than local estimation?', tab: 'sweeps', figure: 'tau2', overrides: {} },
  { id: 'S3', title: 'Simulation 3: equity and network size', question: 'Who gains, and how many hospitals are needed?', tab: 'sweeps', figure: 'K', overrides: {} },
  { id: 'S4', title: 'Simulation 4: gossip rounds', question: 'How many rounds are needed?', tab: 'sweeps', figure: 'T', overrides: { topo: 'ring' } },
  { id: 'S5', title: 'Simulation 5: joint release', question: 'Does secure aggregation among small hospitals help?', tab: 'privacy', figure: 'jointEps',
    overrides: { sizeMode: 'table', sizesTable: [...Array(16).fill(20), ...Array(8).fill(200)], group: 4, eps: 0.1 } },
  { id: 'S6', title: 'Simulation 6: regional structure', question: 'When does the GMRF prior of Eq. (5) pay off?', tab: 'regions', figure: 'gmrf',
    overrides: { regional: true, regions: 4, tauR2: 0.02, tauW2: 0.002, gmrf: true, beta: 100 } },
  { id: 'S7', title: 'Simulation 7: logistic heads (Gaussian analogue)', question: 'Does it work beyond the Gaussian model?', tab: 'borrow',
    overrides: { tau2: 0.05, eps: 0.3 }, note: 'The paper fits logistic heads; this preset runs the Gaussian model at the same heterogeneity and budget. The logistic version is in the Literature arena.' },
  { id: 'S8', title: 'Simulation 8: federated baselines (Gaussian analogue)', question: 'Does it beat existing approaches at the same budget?', tab: 'borrow',
    overrides: { tau2: 0.05 }, note: 'Gaussian analogue: global pooled stands in for one-shot FedAvg. The logistic Simulation 8, with FedAvg, meta-analysis and Ditto-style, is a preset of the Literature arena.' },
  { id: 'S9', title: 'Simulation 9: atypical hospitals', question: 'What happens to a hospital unlike the others?', tab: 'stress',
    overrides: { effects: 't', nu: 3, robustT: true, optOut: true } },
  { id: 'S10', title: 'Simulation 10: case-mix shift (Gaussian analogue)', question: 'Does it survive hospitals with different patients?', tab: 'borrow',
    overrides: { tau2: 0.08 }, note: 'In the Gaussian model a case-mix shift acts as extra heterogeneity; the logistic Simulation 10 is a preset of the Literature arena.' },
  { id: 'S11', title: 'Simulation 11: wrong anchor (Gaussian analogue)', question: 'What if the published model is wrong?', tab: 'sweeps', figure: 'anchor',
    overrides: { anchor: true, anchorErr: 0.3, anchorSd: 0.05, anchorCheck: true }, note: 'Overconfident anchor (stated sd 0.05) with the conflict check on.' },
  { id: 'S12', title: 'Simulation 12: dishonest hospitals', question: 'What if hospitals lie?', tab: 'stress', figure: 'lie',
    overrides: { nBad: 3, badWhich: 'largest', lie: 8, screen: true } },
  { id: 'S13', title: 'Simulation 13: link failures', question: 'What if links fail during gossip?', tab: 'network', figure: 'pFail', overrides: { pFail: 0.8 } },
  { id: 'S14', title: 'Simulation 14: repeated releases', question: 'What do refits on the same patients cost?', tab: 'privacy', figure: 'refits',
    overrides: { refits: 10, accounting: 'zcdp' } }
];

/**
 * Configuration for a preset.
 * @param {string} id scenario id
 * @returns {Object} configuration
 */
export function scenarioConfig(id) {
  const s = SCENARIOS.find((x) => x.id === id) || SCENARIOS[0];
  const c = { ...defaultConfig(), ...s.overrides };
  if (s.overrides.sizesTable) c.K = s.overrides.sizesTable.length;
  return c;
}

const SIM15 = ['local', 'pooled', 'ghb', 'lite', 'liteOracle', 'dittoStyle', 'dpfedavg', 'dpfedavgFT', 'dec', 'clu', 'lgd'];
const SIM8 = ['local', 'pooled', 'ghb', 'lite', 'fedavg1', 'meta', 'dittoStyle'];

/**
 * Presets of the Literature arena: the settings of Simulations 8, 10 and 15 (Section 6.3; exp8_baselines.py,
 * exp15_literature.py) and a run of every method. `overrides` are arena keys of config.js; `methods` the method ids;
 * `ref` the paper's mean excess log-loss (10^-3 nats) for the methods it reports, from supplement/results.
 * @type {Array<{id:string, title:string, note:string, overrides:Object, methods:string[]|null, ref?:Object}>}
 * @see Section 6.3, Tables 3 and 4
 */
export const ARENA_PRESETS = [
  { id: 'all', title: 'Every method at the paper\'s main setting', note: 'τ² = 0.05, ε = 1, 24 hospitals; every method with its paper-default configuration.', overrides: { aTuning: 'defaults', aTau2: 0.05, aEps: 1 }, methods: null },
  { id: 's15-0.05', title: 'Simulation 15: τ² = 0.05, ε = 1', note: 'The main comparison; baselines oracle-tuned on separate replicates.', overrides: { aTuning: 'oracle', aTau2: 0.05, aEps: 1 }, methods: SIM15,
    ref: { local: 27.1, lgd: 19.0, ghb: 15.9, lite: 13.8, liteOracle: 12.4, dittoStyle: 15.4, dpfedavg: 27.4, dpfedavgFT: 15.3, dec: 19.1, clu: 21.4, pooled: 17.3 } },
  { id: 's15-0.01', title: 'Simulation 15: τ² = 0.01', note: 'Nearly identical hospitals.', overrides: { aTuning: 'oracle', aTau2: 0.01, aEps: 1 }, methods: SIM15,
    ref: { local: 28.4, lgd: 16.6, ghb: 11.8, lite: 10.1, liteOracle: 8.1, dittoStyle: 12.2, dpfedavg: 15.6, dpfedavgFT: 11.4, dec: 16.6, clu: 16.3, pooled: 4.3 } },
  { id: 's15-0.2', title: 'Simulation 15: τ² = 0.2', note: 'Strong heterogeneity.', overrides: { aTuning: 'oracle', aTau2: 0.2, aEps: 1 }, methods: SIM15,
    ref: { local: 26.7, lgd: 24.4, ghb: 24.1, lite: 21.7, liteOracle: 19.4, dittoStyle: 20.5, dpfedavg: 73.2, dpfedavgFT: 23.2, dec: 24.4, clu: 24.4, pooled: 61.0 } },
  { id: 's15-0.5', title: 'Simulation 15: τ² = 0.5', note: 'Hospitals differ so much that no private method gains.', overrides: { aTuning: 'oracle', aTau2: 0.5, aEps: 1 }, methods: SIM15,
    ref: { local: 26.0, lgd: 29.3, ghb: 28.8, lite: 27.5, liteOracle: 24.0, dittoStyle: 26.0, dpfedavg: 136.6, dpfedavgFT: 28.7, dec: 29.4, clu: 29.4, pooled: 124.4 } },
  { id: 's15-eps0.3', title: 'Simulation 15: ε = 0.3', note: 'Strict privacy budget, where GHB is more accurate than GHB-lite.', overrides: { aTuning: 'oracle', aTau2: 0.05, aEps: 0.3 }, methods: SIM15,
    ref: { local: 26.9, lgd: 20.5, ghb: 16.8, lite: 17.5, liteOracle: 15.8, dittoStyle: 30.6, dpfedavg: 45.2, dpfedavgFT: 19.6, dec: 24.8, clu: 20.6, pooled: 17.0 } },
  { id: 's15-nodp', title: 'Simulation 15: no privacy', note: 'Without privacy noise.', overrides: { aTuning: 'oracle', aTau2: 0.05, aNoDP: true }, methods: SIM15,
    ref: { local: 27.8, lgd: 19.0, ghb: 16.1, lite: 10.4, liteOracle: 9.7, dittoStyle: 10.1, dpfedavg: 16.7, dpfedavgFT: 11.3, dec: 11.8, clu: 13.8, pooled: 16.7 } },
  { id: 's15-regional', title: 'Simulation 15: regional world', note: '4 regions, between-region τ_r² = 0.2, within τ² = 0.02 (held out).', overrides: { aTuning: 'oracle', aTau2: 0.02, aEps: 1, aRegions: 4, aTauR2: 0.2 }, methods: SIM15,
    ref: { local: 28.7, lgd: 26.5, ghb: 23.4, lite: 20.9, liteOracle: 19.9, dittoStyle: 21.2, dpfedavg: 63.1, dpfedavgFT: 24.9, dec: 26.4, clu: 27.1, pooled: 48.5 } },
  { id: 's15-anchor', title: 'Simulation 15: wrong anchor', note: 'Published anchor wrong by N(0, 1) per coefficient, stated sd 0.3 (held out).', overrides: { aTuning: 'oracle', aTau2: 0.05, aEps: 1, aAnchorErr: 1 }, methods: SIM15,
    ref: { local: 34.6, lgd: 41.9, ghb: 16.7, lite: 16.2, liteOracle: 15.0, dittoStyle: 14.8, dpfedavg: 44.6, dpfedavgFT: 21.0, dec: 41.8, clu: 46.3, pooled: 16.9 } },
  { id: 's15-mod', title: 'Simulation 15: τ² = 0.1, ε = 0.5', note: 'Moderate heterogeneity and budget (held out).', overrides: { aTuning: 'oracle', aTau2: 0.1, aEps: 0.5 }, methods: SIM15,
    ref: { local: 26.9, lgd: 22.8, ghb: 19.6, lite: 18.4, liteOracle: 17.1, dittoStyle: 23.8, dpfedavg: 52.6, dpfedavgFT: 21.7, dec: 23.9, clu: 22.9, pooled: 32.7 } },
  { id: 's15-K48', title: 'Simulation 15: 48 hospitals', note: 'A larger network (held out).', overrides: { aTuning: 'oracle', aTau2: 0.05, aEps: 1, aK: 48 }, methods: SIM15,
    ref: { local: 32.4, lgd: 22.4, ghb: 17.7, lite: 14.4, liteOracle: 13.2, dittoStyle: 14.6, dpfedavg: 25.8, dpfedavgFT: 16.0, dec: 22.5, clu: 24.1, pooled: 17.2 } },
  { id: 's8-0.05', title: 'Simulation 8: one-shot baselines, τ² = 0.05', note: 'Every private method uses GHB-lite\'s releases; Ditto-style λ tuned.', overrides: { aTuning: 'oracle', aTau2: 0.05, aEps: 1 }, methods: SIM8,
    ref: { local: 27.4, lite: 13.4, ghb: 15.1, pooled: 16.7, fedavg1: 36.5, meta: 30.3, dittoStyle: 15.6 } },
  { id: 's8-0.5', title: 'Simulation 8: τ² = 0.5', note: 'Where only Ditto-style gains.', overrides: { aTuning: 'oracle', aTau2: 0.5, aEps: 1 }, methods: SIM8,
    ref: { local: 27.3, lite: 28.2, ghb: 30.7, pooled: 125.6, fedavg1: 141.1, meta: 138.8, dittoStyle: 26.5 } },
  { id: 's10-shift', title: 'Simulation 10: case-mix shift', note: 'Feature means differ by 1 sd between hospitals, τ² = 0.05.', overrides: { aTuning: 'oracle', aTau2: 0.05, aEps: 1, aShift: 1 }, methods: SIM8,
    ref: { local: 22.8, lite: 13.2, ghb: 14.2, pooled: 19.8, fedavg1: 46.3, meta: 43.1, dittoStyle: 14.0 } }
];

/**
 * Configuration for an arena preset: the arena keys reset to their defaults, then the preset's overrides.
 * @param {Object} base current configuration
 * @param {string} id preset id
 * @returns {Object} configuration
 */
export function arenaPresetConfig(base, id) {
  const p = ARENA_PRESETS.find((x) => x.id === id) || ARENA_PRESETS[0], d = defaultConfig(), c = { ...base };
  for (const k of Object.keys(d)) if (k.startsWith('a') && /^a[A-Z]/.test(k)) c[k] = d[k];
  return { ...c, ...p.overrides };
}
