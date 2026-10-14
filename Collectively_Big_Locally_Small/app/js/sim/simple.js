/**
 * sim/simple.js
 * Responsibility: the Simple mode scenario. Turns a handful of plain-language choices into the configurations of the
 * existing simulations (no new mathematics), runs them, and writes plain sentences that are computed from the run.
 * DOM-free: the Simple mode page (ui/simple.js) and the headless test (tests/simple.test.mjs) both use it.
 *
 * How the Simple mode controls map to the simulation parameters
 * ------------------------------------------------------------
 * Main test: the Gaussian Monte Carlo engine of the expert tabs (sim/engine.js, config.js), paper defaults
 * (Section 6.1) for every parameter not listed here. Each hospital estimates one number from its patients; the error
 * of an estimate is its mean squared distance from the hospital's true value. "Error compared with working alone" is
 * the engine's per-hospital relative risk, MSE / (sigma^2 / n_k), so working alone is exactly 100.
 *   - Number of hospitals           -> K. Sizes are taken from the paper's size profile, siteSizes(K, median), as a
 *                                       per-hospital table (sizeMode 'table'), so both tests below use the same sizes.
 *   - Hospital sizes                -> median of that profile (log-normal spread 1, sizes kept within 15 to 1500):
 *                                       mostly small = 30, mixed = 100 (the paper's design), mostly large = 400.
 *   - How different the patients are -> heterogeneity tau^2 between hospitals (per-patient variance sigma^2 = 1):
 *                                       very similar = 0.005, somewhat different = 0.02 (paper default),
 *                                       very different = 0.2.
 *   - Privacy protection            -> privacy budget epsilon (delta = 1e-5, clip norm 1):
 *                                       low = 10, medium = 1 (paper default), high = 0.1.
 *   - "With GHB"                    -> method 'ghb' (GHB with one joint release of the network sum, Section 5.2).
 *   - "Each hospital alone"         -> method 'local' (exactly 100 on this scale).
 *   - "Everyone pooled"             -> method 'pooled' (one model on all patients, no privacy; not allowed in
 *                                       practice, shown as a reference).
 *   Monte Carlo: 2,000 replicates, seed 7 (so a setting always gives the same answer). The same setting is also run
 *   without privacy noise (noDP), same seed, to say in one sentence what privacy costs.
 * Optional second test, "a standard federated method": the Literature arena (sim/arena.js), the paper's logistic
 *   prediction world (Section 6.3), with methods local, ghb and dpfedavgFT (DP-FedAvg with fine-tuning, paper-default
 *   configuration), same K, same sizes (siteSizes(K, median)), same epsilon, and heterogeneity tau^2 per coefficient:
 *   very similar = 0.01, somewhat different = 0.05 (Simulation 15), very different = 0.5. 6 replicates, 1,000 test
 *   patients per hospital, seed 15. Error is excess log-loss, again shown relative to working alone.
 */
import { defaultConfig, normalizeConfig } from '../config.js';
import { siteSizes } from '../math/random.js';
import { runMonteCarlo } from './engine.js';
import { defaultArenaSpec, runArenaSync } from './arena.js';

/** The plain-language choices, their labels, one-line explanations and the parameters they set. */
export const CHOICES = {
  sizes: {
    label: 'Hospital sizes',
    help: 'How many patients each hospital has. A small hospital has a few dozen, a large one several hundred.',
    options: [
      { id: 'small', label: 'Mostly small', median: 30 },
      { id: 'mixed', label: 'Mixed', median: 100 },
      { id: 'large', label: 'Mostly large', median: 400 }
    ],
    def: 'mixed'
  },
  difference: {
    label: 'How different are the hospitals\' patients?',
    help: 'When patients differ a lot from one hospital to the next, what one hospital has seen says less about another.',
    options: [
      { id: 'similar', label: 'Very similar', tau2: 0.005, arenaTau2: 0.01 },
      { id: 'somewhat', label: 'Somewhat different', tau2: 0.02, arenaTau2: 0.05 },
      { id: 'very', label: 'Very different', tau2: 0.2, arenaTau2: 0.5 }
    ],
    def: 'somewhat'
  },
  privacy: {
    label: 'Privacy protection',
    help: 'How much random noise is added to what the hospitals share, so that no single patient can be identified. More noise protects patients better but blurs what can be learned.',
    options: [
      { id: 'low', label: 'Low', eps: 10 },
      { id: 'medium', label: 'Medium', eps: 1 },
      { id: 'high', label: 'High', eps: 0.1 }
    ],
    def: 'medium'
  }
};

/** Range of the "number of hospitals" slider. */
export const HOSPITALS = { min: 5, max: 30, def: 12, label: 'Number of hospitals', help: 'How many hospitals take part. Each one keeps its patients\' records at home.' };

/** Default choices. */
export function defaultChoices() {
  return { K: HOSPITALS.def, sizes: CHOICES.sizes.def, difference: CHOICES.difference.def, privacy: CHOICES.privacy.def, pooled: true, federated: false };
}

const opt = (group, id) => CHOICES[group].options.find((o) => o.id === id) || CHOICES[group].options.find((o) => o.id === CHOICES[group].def);
const clampK = (K) => Math.max(HOSPITALS.min, Math.min(HOSPITALS.max, Math.round(Number(K) || HOSPITALS.def)));

/**
 * Hospital sizes for a set of choices (ascending).
 * @param {Object} ch choices
 * @returns {number[]}
 */
export function simpleSizes(ch) { return siteSizes(clampK(ch.K), opt('sizes', ch.sizes).median); }

/**
 * Configuration of the Gaussian engine for a set of choices (see the mapping in the header).
 * @param {Object} ch choices
 * @returns {Object} engine configuration
 */
export function simpleConfig(ch) {
  const raw = {
    ...defaultConfig(), sizeMode: 'table', sizesTable: simpleSizes(ch), K: clampK(ch.K),
    tau2: opt('difference', ch.difference).tau2, eps: opt('privacy', ch.privacy).eps, noDP: false, R: 2000, chunk: 50, seed: 7
  };
  return normalizeConfig(raw).config;
}

/**
 * Literature arena specification for the optional federated comparison (see the mapping in the header).
 * @param {Object} ch choices
 * @returns {Object} arena specification
 */
export function simpleArenaSpec(ch) {
  return {
    ...defaultArenaSpec(), K: clampK(ch.K), median: opt('sizes', ch.sizes).median, tau2: opt('difference', ch.difference).arenaTau2,
    eps: opt('privacy', ch.privacy).eps, reps: 6, nTest: 1000, seed: 15, tuning: 'defaults', methods: ['local', 'ghb', 'dpfedavgFT']
  };
}

/**
 * Per-hospital results of the main test from the engine's summaries.
 * @param {Object} summary engine summary (sim/engine.js summarize) at the chosen privacy level
 * @param {Object|null} noDP engine summary of the same setting without privacy noise (same seed), or null
 * @param {Object} ch choices
 * @returns {Object} plain result: hospitals [{k, n, ghb, ghbSE, pooled, noPrivacy, weightOwn, likePatients}], choices
 *   (every error is relative to working alone: 1 = as good as alone, below 1 = better)
 */
export function digest(summary, noDP, ch) {
  const A = summary.agg, hospitals = summary.perHospital.map((p) => ({
    k: p.k, n: p.n, ghb: A.ghb.y[p.k], ghbSE: A.ghb.se[p.k], pooled: A.pooled.y[p.k], noPrivacy: noDP ? noDP.agg.ghb.y[p.k] : NaN,
    weightOwn: p.alphaJ, likePatients: p.n / A.ghb.y[p.k]
  }));
  return { choices: { ...ch, K: clampK(ch.K) }, hospitals, replicates: summary.R };
}

/**
 * The main test as a chunked generator (for sim/scheduler.js): the chosen setting, then the same setting without
 * privacy noise (to say what privacy costs). Yields progress in [0, 1]; returns the digest.
 * @param {Object} ch choices
 * @returns {Generator<number, Object>}
 */
export function* simpleJob(ch) {
  const c = simpleConfig(ch), sums = [];
  for (const [i, cfg] of [c, { ...c, noDP: true }].entries()) {
    const g = runMonteCarlo(cfg); let r = g.next(), steps = 0;
    while (!r.done) { steps++; yield (i + Math.min(1, steps * cfg.chunk / cfg.R)) / 2; r = g.next(); }
    sums.push(r.value.summary);
  }
  return digest(sums[0], sums[1], ch);
}

/**
 * Run the main test synchronously (headless use). The page runs the same generator in chunks.
 * @param {Object} ch choices
 * @returns {Object} digest (see digest)
 */
export function runSimpleSync(ch) {
  const g = simpleJob(ch); let r = g.next();
  while (!r.done) r = g.next();
  return r.value;
}

/**
 * Plain result of the federated comparison from an arena result.
 * @param {Object} res arena result (sim/arena.js runArena)
 * @returns {{K:number, ghb:{mean:number, worse:number}, fed:{mean:number, worse:number}, replicates:number}}
 *   mean: average over hospitals of the error relative to working alone; worse: hospitals above working alone
 */
export function digestArena(res) {
  const one = (id) => { const r = res.summary[id].ratio; return { mean: r.reduce((a, b) => a + b, 0) / r.length, worse: r.filter((x) => x > 1).length, ratio: Array.from(r) }; };
  return { K: res.K, ghb: one('ghb'), fed: one('dpfedavgFT'), replicates: res.spec.reps };
}

/**
 * Run the federated comparison synchronously (headless use; the page uses the arena worker).
 * @param {Object} ch choices
 * @param {Object} [over] overrides of the arena specification (tests use fewer replicates)
 * @returns {Object} digestArena result
 */
export function runFederatedSync(ch, over = {}) { return digestArena(runArenaSync({ ...simpleArenaSpec(ch), ...over })); }

const pct = (x) => Math.round(100 * x);
const times = (g) => (g >= 9.5 ? Math.round(g).toString() : g.toFixed(1).replace(/\.0$/, ''));
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

/**
 * Plain sentences describing a run. Every claim is computed from the numbers passed in.
 * @param {Object} r digest of the main test
 * @param {Object|null} [fed] digestArena result of the federated comparison
 * @returns {{main:{text:string, kind:string}[], compare:{text:string, kind:string}[]}} the main findings
 *   (kinds 'small', 'none-worse' or 'worse', 'weak' or 'large') and the comparisons ('privacy', 'pooled', 'federated')
 */
export function describe(r, fed = null) {
  const H = r.hospitals, K = H.length, main = [], compare = [], s = H[0], L = H[K - 1], ch = r.choices;
  const say = (list, kind, text) => list.push({ kind, text });
  // 1. the smallest hospital
  const g = 1 / s.ghb;
  if (g >= 1.1) say(main, 'small', `With GHB, the smallest hospital (${s.n} patients) does as well as if it had about ${times(g)} times as many patients (about ${Math.round(s.likePatients)} instead of ${s.n}).`);
  else if (s.ghb <= 1) say(main, 'small', `With GHB, the smallest hospital (${s.n} patients) gains only a little: its error drops by ${100 - pct(s.ghb)}%.`);
  else say(main, 'small', `With GHB, the smallest hospital (${s.n} patients) is slightly worse off than working alone: its error is ${Math.max(1, pct(s.ghb) - 100)}% higher.`);
  // 2. is anyone worse off?
  const worse = H.filter((h) => h.ghb > 1);
  if (!worse.length) say(main, 'none-worse', `No hospital is worse off than working alone: all ${K} have a lower error with GHB.`);
  else {
    const w = worse.reduce((a, b) => (b.ghb > a.ghb ? b : a)), clear = worse.filter((h) => h.ghb - 2 * h.ghbSE > 1).length;
    const which = `The most affected is Hospital ${w.k + 1}, with ${w.n} patients and ${Math.max(1, pct(w.ghb) - 100)}% more error than alone.`;
    if (!clear) say(main, 'worse', `${plural(worse.length, 'hospital is', 'hospitals are')} slightly worse off than working alone, by ${worse.length === 1 ? 'an amount' : 'amounts'} within the random variation of this simulation. ${which}`);
    else say(main, 'worse', `${plural(worse.length, 'hospital is', 'hospitals are')} worse off than working alone. ${which}`);
  }
  // 3. on average, and when GHB helps little, say so and why
  const avgErr = H.reduce((a, h) => a + h.ghb, 0) / K, avgCut = 1 - avgErr, own = H.reduce((a, h) => a + h.weightOwn, 0) / K;
  const avgText = pct(avgCut) >= 1 ? `cuts the error by ${pct(avgCut)}%` : pct(avgCut) <= -1 ? `raises the error by ${-pct(avgCut)}%` : 'barely changes the error';
  if (avgCut < 0.1) {
    const why = ch.difference === 'very' ? 'the hospitals are too different to help each other much'
      : ch.sizes === 'large' ? 'each hospital already has many patients, so there is little left to learn from the others'
        : ch.privacy === 'high' ? 'the strong privacy noise blurs what the hospitals can learn from each other'
          : 'the hospitals have little to add to each other';
    say(main, 'weak', `Here ${why}: on average GHB ${avgText}. GHB notices this and mostly relies on each hospital's own data (about ${pct(own)}% of the weight on average).`);
  } else if (L.ghb > 1 / 1.1) {
    say(main, 'large', `On average GHB ${avgText}. The largest hospital (${L.n} patients) changes little (error ${pct(L.ghb)}, alone = 100): it already has many patients and puts about ${pct(L.weightOwn)}% of the weight on its own data.`);
  } else say(main, 'large', `On average GHB ${avgText}, and even the largest hospital (${L.n} patients) gains (error ${pct(L.ghb)}, alone = 100).`);
  // 4. what privacy costs (same setting, same random draws, no noise)
  if (H.every((h) => Number.isFinite(h.noPrivacy))) {
    const np = H.reduce((a, h) => a + h.noPrivacy, 0) / K, d = pct(avgErr) - pct(np);
    if (d <= 0) say(compare, 'privacy', `Privacy costs nothing visible here: without any noise at all, GHB's average error would also be about ${pct(np)} (alone = 100).`);
    else if (d <= 2) say(compare, 'privacy', `Privacy costs little here: without any noise at all, GHB's average error would be ${pct(np)} instead of ${pct(avgErr)} (alone = 100).`);
    else say(compare, 'privacy', `Privacy has a cost here: without any noise, GHB's average error would be ${pct(np)} instead of ${pct(avgErr)} (alone = 100).`);
  }
  // 5. pooled reference
  if (ch.pooled) {
    const pw = H.filter((h) => h.pooled > 1), pAvg = H.reduce((a, h) => a + h.pooled, 0) / K;
    if (!pw.length) say(compare, 'pooled', `Putting everyone's patients in one place (not allowed in practice) would help every hospital here${pAvg < avgErr ? ', more than GHB does without moving any records' : ', but no more than GHB does without moving any records'}.`);
    else {
      const mw = pw.reduce((a, h) => a + h.n, 0) / pw.length, mo = H.reduce((a, h) => a + h.n, 0) / K;
      say(compare, 'pooled', `Putting everyone's patients in one place (not allowed in practice) would leave ${pw.length} of ${K} hospitals worse off than working alone${pw.length < K && mw > mo ? ', mostly the larger ones' : ''}, because one shared model ignores how the hospitals differ.`);
    }
  }
  // 6. federated comparison
  if (fed) {
    const cut = (m) => (pct(m) < 100 ? `cuts the average error by ${100 - pct(m)}%` : pct(m) > 100 ? `raises the average error by ${pct(m) - 100}%` : 'leaves the average error unchanged');
    say(compare, 'federated', `In the second test (a yes or no prediction, ${fed.replicates} repetitions, so rougher numbers), the standard federated method ${cut(fed.fed.mean)} and leaves ${fed.fed.worse} of ${fed.K} hospitals worse off than alone; GHB ${cut(fed.ghb.mean)} and leaves ${fed.ghb.worse} worse off.`);
  }
  return { main, compare };
}

/** The short story shown at the top of Simple mode. */
export const STORY = [
  'Many hospitals want to predict how their patients will do, but a small hospital has too few patients to train a good prediction model on its own.',
  'Hospitals cannot simply send each other their patients\' records: that would break patient privacy.',
  'GHB (Graph Hierarchical Borrowing) lets each hospital learn from what the others have seen, without receiving a single patient record.',
  'Each hospital decides for itself how much to trust the others, so nobody has to tune anything and there is no central server.'
];

/** The "What just happened?" steps. */
export const STEPS = [
  'Each hospital summarizes its own patients into a few numbers. The patient records never leave the hospital.',
  'The hospitals add up their summaries together, passing them only to neighbouring hospitals, with random noise mixed in, so nobody can see any single hospital\'s summary or any single patient.',
  'Each hospital takes the network total, removes its own part, and gets a picture of what the other hospitals have seen.',
  'Each hospital then decides, from its own data alone, how much to trust that picture: a small hospital leans on the network a lot, a large one or an unusual one mostly keeps its own answer.'
];
