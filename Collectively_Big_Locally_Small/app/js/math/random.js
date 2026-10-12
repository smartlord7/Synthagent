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
 * The first 96 standard normals of numpy.random.default_rng(11), the draws behind `site_sizes()` of
 * simulations/ghb.py, so that its size profiles are reproduced exactly for any K <= 96 and any median or spread.
 */
export const SITE_Z = Object.freeze([0.03419276725318417,1.3597475403099617,1.2247210785859324,-0.5103070767876675,-0.2979695111064471,-0.5273841930334252,0.5697263575719601,-0.056064439045617594,0.7468856162565439,-1.8473247989741095,1.5665487746995206,-0.09643216015562055,0.6803784532741461,-0.13656633397682774,-0.3790985670748533,0.46311015859758675,0.824513527530113,-0.20252987069345152,-0.15278617857019708,0.685698610809258,-0.8703406419471712,-1.5143835037313955,0.39498186274953,-0.6705658236878794,-1.9203405901180286,-0.8140536639453595,-0.467597558892747,-1.1932024774322612,-1.4924638840630338,0.03663782694480509,0.8972492567277476,-0.23313207796045685,-0.7435960295088448,0.3849938087479083,0.7172358071943838,-0.3000105984884774,0.5446678079208929,1.0428754765829538,-0.20695643620832396,-0.8135155419815723,0.3476505985155095,0.24754574096284754,1.0988127684144084,-1.284580778805345,-0.6616129303555477,-0.8381669607156745,-1.7340148462328515,0.1264345551969962,0.527804212495524,-0.7387900314758065,1.3856470744961586,0.8219243366604353,0.6273764788355353,0.4017070914409699,0.955669564448635,-1.3319798395431022,0.6139296582498643,0.6027768335334479,-1.7677185771429749,0.34703010205437973,-0.2504213467099684,0.7815226960616993,-0.4390621876376686,-0.01824085764910033,0.3428515173176555,-0.8762616887442077,0.5985966481803844,-0.10496318852366823,0.49248262367924284,-0.5218375063367878,1.0862015432775176,0.6052019784294742,-0.17802502471933673,0.6319571570936101,1.259755161358625,1.7911755134979888,-1.5735763704402195,0.8831318116225195,0.4650685085133813,-0.09386078018634399,-1.0066649349770713,1.2571886134731436,-1.2617379934445705,0.5669454657347489,1.3018679962026896,-1.5996692880514796,-0.30251784048326236,-1.3092168175162993,0.24405410803590055,1.5143751306746547,2.0235560291721977,-1.7781144428835152,-0.5749490057210601,0.7035450933309114,1.5793726520932216,0.4212106442782976]);

/**
 * Hospital sizes exactly as simulations/ghb.py site_sizes(K, median, spread, lo, hi, seed=11): log-normal, rounded,
 * clipped and sorted. K = 24 and median 100 give PAPER_SIZES.
 * @param {number} K number of hospitals (at most 96)
 * @param {number} [median=100] median size
 * @param {number} [spread=1] standard deviation of log size
 * @param {number} [lo=15] smallest size
 * @param {number} [hi=1500] largest size
 * @returns {number[]} sizes, ascending
 * @see Section 6.1 (design of the simulation study)
 */
export function siteSizes(K, median = 100, spread = 1, lo = 15, hi = 1500) {
  if (K > SITE_Z.length) throw new Error('siteSizes supports at most ' + SITE_Z.length + ' hospitals');
  const rint = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f; };
  return SITE_Z.slice(0, K).map((z) => Math.min(hi, Math.max(lo, rint(Math.exp(Math.log(median) + spread * z))))).sort((a, b) => a - b);
}

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
