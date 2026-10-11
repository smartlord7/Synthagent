/**
 * scenarios.js
 * Responsibility: preset scenarios. "Paper defaults" is the design of Section 8.7; Simulations 1 to 14
 * follow Table 6 of the paper. Each preset is a set of overrides on the defaults, the tab it opens and the
 * paper figure (sim/sweep.js PAPER_FIGURES) it reproduces. Simulations 7, 8, 10 and 11 use logistic heads in
 * the paper; here they run as Gaussian analogues and say so.
 */
import { defaultConfig } from './config.js';

/** @type {Array<{id:string, title:string, question:string, tab:string, figure?:string, overrides:Object, note?:string}>} */
export const SCENARIOS = [
  { id: 'paper', title: 'Paper defaults', question: '24 hospitals, τ² = 0.02, ε = 1, 60 gossip rounds on the ring with two-hop links.', tab: 'borrow', overrides: {} },
  { id: 'S1', title: 'Simulation 1: closed forms', question: 'Do Propositions 8.1 and 8.2 match simulation?', tab: 'borrow',
    overrides: { sizeMode: 'table', sizesTable: [15, 30, 60, 120, 200, 200, 200, 200, 200, 400, 800, 1600], K: 12, exact: true, tauEst: 'oracle', weights: 'equal', optOut: false, robustT: false, screen: false } },
  { id: 'S2', title: 'Simulation 2: heterogeneity', question: 'Is plug-in GHB ever worse than local estimation?', tab: 'sweeps', figure: 'tau2', overrides: {} },
  { id: 'S3', title: 'Simulation 3: equity and network size', question: 'Who gains, and how many hospitals are needed?', tab: 'sweeps', figure: 'K', overrides: {} },
  { id: 'S4', title: 'Simulation 4: gossip rounds', question: 'How many rounds are needed?', tab: 'sweeps', figure: 'T', overrides: { topo: 'ring' } },
  { id: 'S5', title: 'Simulation 5: joint release', question: 'Does secure aggregation among small hospitals help?', tab: 'privacy', figure: 'jointEps',
    overrides: { sizeMode: 'table', sizesTable: [...Array(16).fill(20), ...Array(8).fill(200)], group: 4, eps: 0.1 } },
  { id: 'S6', title: 'Simulation 6: regional structure', question: 'When does the GMRF prior of Eq. (7) pay off?', tab: 'regions', figure: 'gmrf',
    overrides: { regional: true, regions: 4, tauR2: 0.02, tauW2: 0.002, gmrf: true, beta: 100 } },
  { id: 'S7', title: 'Simulation 7: logistic heads (Gaussian analogue)', question: 'Does it work beyond the Gaussian model?', tab: 'borrow',
    overrides: { tau2: 0.05, eps: 0.3 }, note: 'The paper fits logistic heads; this preset runs the Gaussian model at the same heterogeneity and budget.' },
  { id: 'S8', title: 'Simulation 8: federated baselines (Gaussian analogue)', question: 'Does it beat existing approaches at the same budget?', tab: 'borrow',
    overrides: { tau2: 0.05 }, note: 'Global pooled stands in for one-shot FedAvg; Ditto and meta-analysis are not simulated here.' },
  { id: 'S9', title: 'Simulation 9: atypical hospitals', question: 'What happens to a hospital unlike the others?', tab: 'stress',
    overrides: { effects: 't', nu: 3, robustT: true, optOut: true } },
  { id: 'S10', title: 'Simulation 10: case-mix shift (Gaussian analogue)', question: 'Does it survive hospitals with different patients?', tab: 'borrow',
    overrides: { tau2: 0.08 }, note: 'In the Gaussian model a case-mix shift acts as extra heterogeneity; the paper simulates it with logistic heads.' },
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
