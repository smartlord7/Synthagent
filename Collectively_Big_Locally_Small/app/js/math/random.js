/**
 * math/random.js
 * Responsibility: seeded pseudo-random numbers and the hospital size profiles.
 * Pure and side-effect free: every generator owns its state, nothing global is touched.
 */

/**
 * The 24 hospital sizes of the paper's simulation study: the output of `site_sizes()` in
 * simulations/ghb.py (log-normal, median 100, spread 1, clipped to [15, 1500], seed 11, sorted).
 * @see Section 8.7, "Unless stated otherwise, the network has 24 hospitals ... (16 to 479 patients)"
 */
export const PAPER_SIZES = Object.freeze([16, 22, 42, 51, 59, 60, 68, 74, 82, 86, 87, 91, 95, 103, 148, 159, 177, 197, 199, 211, 228, 340, 390, 479]);

/**
 * Create a seeded generator (mulberry32 for uniforms, Marsaglia polar method for normals).
 * @param {number} seed integer seed; equal seeds give equal streams
 * @returns {{u:()=>number, n:()=>number, t:(nu:number)=>number}} uniform on [0,1), standard normal, Student-t with nu degrees of freedom
 */
export function makeRng(seed) {
  let a = (seed >>> 0) || 1;
  const u = () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const n = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let x, y, s;
    do { x = 2 * u() - 1; y = 2 * u() - 1; s = x * x + y * y; } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    spare = y * m;
    return x * m;
  };
  const t = (nu) => { let c = 0; for (let i = 0; i < nu; i++) { const z = n(); c += z * z; } return n() / Math.sqrt(c / nu); };
  return { u, n, t };
}

/**
 * A skewed, sorted set of hospital sizes drawn from a clipped log-normal distribution.
 * Mirrors `site_sizes()` of simulations/ghb.py, but with this module's generator, so the draws differ from numpy's.
 * @param {number} K number of hospitals
 * @param {number} median median size
 * @param {number} spread standard deviation of log size
 * @param {number} seed generator seed
 * @param {number} [lo=15] smallest allowed size
 * @param {number} [hi=1500] largest allowed size
 * @returns {number[]} sizes, ascending
 */
export function lognormalSizes(K, median, spread, seed, lo = 15, hi = 1500) {
  const r = makeRng(seed * 7919 + 13);
  const a = [];
  for (let i = 0; i < K; i++) a.push(Math.min(hi, Math.max(lo, Math.round(Math.exp(Math.log(median) + spread * r.n())))));
  return a.sort((x, y) => x - y);
}

/**
 * Seeded Fisher-Yates permutation (returns a new array).
 * @param {Array} arr items
 * @param {number} seed generator seed
 * @returns {Array} permuted copy
 */
export function shuffle(arr, seed) {
  const r = makeRng(seed), a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
