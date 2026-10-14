/**
 * sim/simpleStory.js
 * Responsibility: the data behind the animated network of Simple mode. One illustrative replicate of the current
 * Simple mode setting is computed with the engine's own building blocks, so every number and every movement in the
 * animation comes from the simulation, not from decoration. DOM-free: the page (ui/simpleNetwork.js) and the
 * headless test (tests/simple_story.test.mjs) both use it.
 *
 * What is computed, and from what
 * -------------------------------
 *   - Hospitals, sizes and agreements: engine.prepare(simpleConfig(choices)) (sizes from the paper's profile, the
 *     ring with two-hop links of the paper's design, Metropolis-Hastings gossip weights W).
 *   - True values and each hospital's own estimate: engine.drawEffects and ghb.drawReplicate, exactly as one
 *     replicate of the engine (the estimate of one illustrative number per hospital; its haze is the standard error
 *     sqrt(sigma^2 / n_k)).
 *   - Privacy noise: the network sum carries one Gaussian noise of sd s_J (dp.jointNoiseSd through prepare), shared
 *     out as K independent shares of sd s_J / sqrt(K), one added by each hospital; their sum is the xi passed to GHB.
 *   - Scrambling: pairwise masks, one random number per agreement, added by one end and subtracted by the other, so
 *     they cancel in the total (secure summation, Section 5.2).
 *   - Gossip: sim/gossip.js createGossip on the masked contributions with the setting's W; the first rounds are kept
 *     for the animation and the process runs the setting's T rounds; K times the consensus value is the noisy total.
 *   - Combined picture, trust and result: math/ghb.js jointGHB (as engine.replicate calls it): mu_k is the network's
 *     estimate without hospital k, alpha_k the weight on its own estimate, so trust in the network is 1 - alpha_k and
 *     the GHB estimate is alpha_k * own + (1 - alpha_k) * mu_k.
 *   - "Make Hospital X very different": the same replicate (same random draws) with hospital X's true value moved
 *     to ATYPICAL_SHIFT standard deviations of the between-hospital spread (the engine's 'outlier' effect).
 */
import { prepare, drawEffects } from './engine.js';
import { drawReplicate, jointGHB } from '../math/ghb.js';
import { makeRng } from '../math/random.js';
import { createGossip } from './gossip.js';
import { simpleConfig } from './simple.js';

/** Distance of an atypical hospital from the others, in standard deviations of the between-hospital spread. */
export const ATYPICAL_SHIFT = 8;
/** Gossip rounds shown one by one in the animation (the computation runs all T rounds of the setting). */
export const ROUNDS_SHOWN = 6;

/** The animated story: one id and title per step (captions with numbers come from stepCaption). */
export const STORY_STEPS = [
  { id: 'meet', title: 'Meet the hospitals' },
  { id: 'own', title: 'Each hospital looks only at its own patients' },
  { id: 'noise', title: 'Each hospital adds random noise' },
  { id: 'travel', title: 'Scrambled summaries travel along the agreements' },
  { id: 'combine', title: 'The scrambles cancel: everyone learns the combined picture' },
  { id: 'trust', title: 'Each hospital decides how much to trust the network' },
  { id: 'result', title: 'Result' }
];

/**
 * Data of one illustrative replicate for the animation.
 * @param {Object} ch Simple mode choices (sim/simple.js)
 * @param {{atypical?:number, seed?:number}} [o] atypical hospital index (-1 for none) and seed
 * @returns {Object} K, n, edges, theta, own, sd, noiseSd, shares, masks, contrib, frames, rounds, spread, total, N,
 *   network (mu), alpha, trust, ghb, centre, extent, atypical
 */
export function storyData(ch, o = {}) {
  const atypical = o.atypical ?? -1, c = simpleConfig(ch), ctx = prepare(c), { n, K, edges, W } = ctx;
  const rng = makeRng(o.seed ?? 2024), tau = Math.sqrt(ctx.tau2m);
  const u = drawEffects(ctx, rng);
  if (atypical >= 0 && atypical < K) u[atypical] = ATYPICAL_SHIFT * tau;
  const d = drawReplicate(rng, n, ctx.q, (k) => u[k]);
  const noiseSd = ctx.sJ / Math.sqrt(K), shares = Float64Array.from(n, () => noiseSd * rng.n());
  const xi = shares.reduce((a, b) => a + b, 0);
  let big = 1; for (let k = 0; k < K; k++) big = Math.max(big, Math.abs(n[k] * d.th[k]));
  const masks = new Float64Array(K);
  for (const [i, j] of edges) { const r = 3 * big * rng.n(); masks[i] += r; masks[j] -= r; }
  const contrib = Float64Array.from(n, (nk, k) => nk * d.th[k] + shares[k] + masks[k]);
  const g = createGossip({ K, edges, W }, contrib, 0), frames = [Float64Array.from(contrib)];
  for (let t = 0; t < c.T; t++) { g.step(); if (t < ROUNDS_SHOWN) frames.push(Float64Array.from(g.x)); }
  const total = contrib.reduce((a, b) => a + b, 0), xbar = total / K;
  let spread = 0; for (let k = 0; k < K; k++) spread = Math.max(spread, Math.abs(g.x[k] - xbar));
  const j = jointGHB(d.th, d.th, n, d.v, c.sigma2, ctx.sJ, xi, { scale: c.cauchy });
  const extent = 2.5 * Math.sqrt(ctx.tau2m + c.sigma2 / n[Math.floor(K / 2)]);
  return {
    K, n: n.slice(), edges: edges.map((e) => e.slice()), theta: Array.from(d.theta), own: Array.from(d.th), sd: Array.from(d.v, Math.sqrt),
    noiseSd, sJ: ctx.sJ, shares: Array.from(shares), masks: Array.from(masks), contrib: Array.from(contrib), frames: frames.map((f) => Array.from(f)),
    final: Array.from(g.x), rounds: c.T, spread: Math.abs(xbar) > 0 ? spread / Math.abs(xbar) : spread, total, xi, N: n.reduce((a, b) => a + b, 0),
    network: Array.from(j.mu), alpha: Array.from(j.alpha), trust: Array.from(j.alpha, (a) => 1 - a), ghb: Array.from(j.est),
    centre: c.mu, extent, atypical, eps: c.eps
  };
}

/**
 * Node radius in pixels: area proportional to patients (the largest hospital gets `room`).
 * @param {number[]} n sizes
 * @param {number} k hospital
 * @param {number} room radius of the largest hospital
 * @returns {number}
 */
export function nodeRadius(n, k, room) { return Math.max(6, Math.sqrt(n[k] / Math.max(...n)) * room); }

/**
 * Noise level for the drawing, in [0, 1]: grows with the noise sd of each hospital's share (log scale, 0 without
 * noise, about 1 at the strongest protection of Simple mode).
 * @param {number} noiseSd sd of one hospital's noise share
 * @returns {number}
 */
export function noiseLevel(noiseSd) { return Math.max(0, Math.min(1, Math.log10(1 + noiseSd) / 2)); }

const pc = (x) => Math.round(100 * x);
const hosp = (D, k) => `Hospital ${k + 1} (${D.n[k]} patients)`;

/**
 * Plain caption of a step, with numbers from the replicate (and from the Monte Carlo result for the last step).
 * @param {string} id step id
 * @param {Object} D storyData result
 * @param {Object|null} [res] digest of the Monte Carlo run (sim/simple.js), for the result step
 * @param {Object|null} [typical] storyData of the same setting without the atypical hospital (for comparison)
 * @returns {string[]} one or two sentences
 */
export function stepCaption(id, D, res = null, typical = null) {
  const K = D.K, s = 0, L = K - 1;
  switch (id) {
    case 'meet': return [
      `Each circle is a hospital: ${K} of them, from ${D.n[s]} to ${D.n[L]} patients, and a bigger circle means more patients.`,
      `Each line is an agreement between two hospitals to exchange summaries. There are ${D.edges.length} agreements, nobody talks to everyone, and there is no central server.`
    ];
    case 'own': return [
      'Each hospital looks only at its own patients and makes its own estimate. The patient records never leave the hospital.',
      `The haze shows how unsure each estimate is: wide for ${hosp(D, s)}, narrow for ${hosp(D, L)}.`
    ];
    case 'noise': {
      if (!(D.noiseSd > 0)) return ['No noise is added in this setting.'];
      const r = D.noiseSd / 2;
      return [
        'Each hospital adds random noise to its summary (the orange dots), so that no single patient can be recognized.',
        r >= 1 ? `With this privacy protection, each hospital's noise is about ${r >= 10 ? Math.round(r) : r.toFixed(1)} times bigger than the most a single patient can change a summary. Change the privacy setting to see it grow or shrink.`
          : `With this privacy protection, each hospital's noise is about ${pc(r)}% of the most a single patient can change a summary: the noise of all hospitals together is what hides each patient. Change the privacy setting to see it grow or shrink.`
      ];
    }
    case 'travel': return [
      'The summaries travel along the agreements, scrambled so nobody can read any one hospital\'s.',
      `Each hospital adds a secret random number shared with each neighbour, one adding it and the other subtracting it, and passes mixed totals to its neighbours, round after round (${D.rounds} rounds here).`
    ];
    case 'combine': return [
      'Added up, the scrambles cancel and every hospital learns the network\'s combined picture.',
      `After ${D.rounds} rounds all ${K} hospitals hold the same total${D.spread < 0.001 ? ', to better than 0.1%' : `, to within ${(100 * D.spread).toFixed(1)}%`}. Each one removes its own part to see what the others have seen (the blue marks).`
    ];
    case 'trust': {
      const out = [`Each hospital decides how much to trust the network, using its own data: ${hosp(D, s)} trusts it ${pc(D.trust[s])}%, ${hosp(D, L)} ${pc(D.trust[L])}%.`];
      const a = D.atypical;
      if (a >= 0 && typical) out.push(`${hosp(D, a)} is now very different from the others, and notices it: it trusts the network ${pc(D.trust[a])}%, against ${pc(typical.trust[a])}% when it was typical.`);
      else {
        let lo = 0; for (let k = 1; k < K; k++) if (D.trust[k] < D.trust[lo]) lo = k;
        out.push(`Small hospitals move a lot toward the network, large ones little. The least trusting is ${hosp(D, lo)}, at ${pc(D.trust[lo])}%.`);
      }
      return out;
    }
    case 'result': {
      if (!res) return ['Running the full simulation...'];
      const H = res.hospitals, better = H.filter((h) => h.ghb <= 1).length;
      return [
        `Over 2,000 repetitions, ${better} of ${H.length} hospitals end up with a smaller error than working alone (teal circles; red means worse off).`,
        `${hosp(D, s)} gets an error of ${pc(H[s].ghb)} instead of 100 alone, like having ${Math.round(H[s].likePatients)} patients.`
      ];
    }
    default: return [];
  }
}
