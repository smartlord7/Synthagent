/**
 * math/ghb.js
 * Responsibility: the estimators of Graph Hierarchical Borrowing (Section 8): closed forms of
 * Propositions 8.1 and 8.2, the plug-in estimator of Algorithm 1 (DerSimonian-Laird tau^2,
 * inverse-variance leave-one-out mean through gossip, shrinkage, anchor), its robust variants
 * (Student-t prior, opt-out, screening of dishonest releases), joint release and the GMRF Jacobi update.
 * All functions are pure; random draws take an explicit generator.
 */
import { dpVar } from './dp.js';

const map = (K, f) => { const o = new Float64Array(K); for (let i = 0; i < K; i++) o[i] = f(i); return o; };

/* ---------- closed forms (blueprint.py) ---------- */

/**
 * Optimal risk of hospital k.
 * @param {number} n hospital size
 * @param {number} tau2 heterogeneity
 * @param {number} M mean squared error of the population estimate
 * @param {number} [sigma2=1] per-patient variance
 * @returns {number} R* = v (tau^2 + M) / (v + tau^2 + M), v = sigma^2 / n
 * @see Proposition 8.1
 */
export function riskHB(n, tau2, M, sigma2 = 1) { const v = sigma2 / n; return v * (tau2 + M) / (v + tau2 + M); }

/**
 * Optimal shrinkage weight on the local estimate.
 * @param {number} v local variance sigma^2 / n_k
 * @param {number} tau2 heterogeneity
 * @param {number} M population-estimate MSE
 * @returns {number} alpha* = (tau^2 + M) / (tau^2 + M + v)
 * @see Proposition 8.1
 */
export function alphaHB(v, tau2, M) { return (tau2 + M) / (tau2 + M + v); }

/**
 * Borrowed (virtual) patients.
 * @param {number} tau2 heterogeneity
 * @param {number} M population-estimate MSE
 * @param {number} [sigma2=1] per-patient variance
 * @returns {number} B = sigma^2 / (tau^2 + M)
 * @see Proposition 8.1, Eq. (5)
 */
export function borrowedPatients(tau2, M, sigma2 = 1) { return sigma2 / (tau2 + M); }

/**
 * MSE of the equal-weight leave-one-out mean of privatized releases.
 * @param {number[]} sizes all hospital sizes
 * @param {number} k excluded hospital
 * @param {{tau2:number, sigma2?:number, eps:number, delta:number, C:number}} q model and privacy parameters
 * @returns {number} M = (K-1)^{-2} sum_{j != k} (tau^2 + sigma^2/n_j + s_j^2)
 * @see Proposition 8.2, Eq. (6)
 */
export function looMSE(sizes, k, q) {
  const K = sizes.length, s2 = q.sigma2 ?? 1; let s = 0;
  for (let j = 0; j < K; j++) if (j !== k) s += q.tau2 + s2 / sizes[j] + dpVar(sizes[j], q.eps, q.delta, q.C);
  return s / ((K - 1) ** 2);
}

/**
 * MSE of the inverse-variance leave-one-out mean, given each other release's total variance.
 * @param {number[]} totals tau^2 + sigma^2/n_j + s_j^2 for every hospital
 * @param {number} k excluded hospital
 * @returns {number} M = 1 / sum_{j != k} 1/total_j
 * @see Proposition 8.2 (third consequence), Algorithm 1
 */
export function looMSEiv(totals, k) { let s = 0; for (let j = 0; j < totals.length; j++) if (j !== k) s += 1 / totals[j]; return 1 / s; }

/**
 * Risk of hospital k under the sample-size weighted pooled mean (a single global model).
 * @param {number[]} sizes all sizes
 * @param {number} k hospital
 * @param {number} tau2 heterogeneity
 * @param {number} [sigma2=1] per-patient variance
 * @returns {number} tau^2 ((1 - w_k)^2 + sum_{j!=k} w_j^2) + sigma^2 / N
 * @see blueprint.risk_full_pooling, Figure 8
 */
export function riskFullPooling(sizes, k, tau2, sigma2 = 1) {
  const N = sizes.reduce((a, b) => a + b, 0); let so = 0;
  sizes.forEach((n, j) => { if (j !== k) so += (n / N) ** 2; });
  return tau2 * ((1 - sizes[k] / N) ** 2 + so) + sigma2 / N;
}

/* ---------- model draws ---------- */

/**
 * One replicate of model (4) with Gaussian-mechanism releases.
 * @param {{n:()=>number}} rng generator
 * @param {number[]} n sizes
 * @param {{tau2:number, sigma2?:number, eps:number, delta:number, C:number, mu?:number}} q parameters
 * @param {(k:number)=>number} [uFn] optional site effects u_k (default N(0, tau^2))
 * @returns {{v:Float64Array, s2:Float64Array, theta:Float64Array, th:Float64Array, x:Float64Array, r:Float64Array}}
 *   local variances, noise variances, true parameters, local estimates, releases, public release variances
 * @see Eq. (4), Proposition 8.2
 */
export function drawReplicate(rng, n, q, uFn) {
  const K = n.length, sg = q.sigma2 ?? 1, mu = q.mu ?? 0;
  const v = map(K, (i) => sg / n[i]), s2 = map(K, (i) => dpVar(n[i], q.eps, q.delta, q.C));
  const theta = map(K, (i) => mu + (uFn ? uFn(i) : Math.sqrt(q.tau2) * rng.n()));
  const th = map(K, (i) => theta[i] + Math.sqrt(v[i]) * rng.n());
  const x = map(K, (i) => th[i] + Math.sqrt(s2[i]) * rng.n());
  return { v, s2, theta, th, x, r: map(K, (i) => v[i] + s2[i]) };
}

/* ---------- plug-in GHB (Algorithm 1) ---------- */

/**
 * Network averaging as seen by each hospital: exact (server reference) or through an averaging operator P = W^T.
 * @param {Float64Array[]|null} P averaging operator, null for the exact mean
 * @param {number} K hospitals
 * @returns {(y:ArrayLike<number>)=>Float64Array} y -> per-hospital estimate of mean(y)
 * @see Proposition 8.4
 */
export function avgOp(P, K) {
  if (!P) return (y) => { let s = 0; for (let i = 0; i < K; i++) s += y[i]; const o = new Float64Array(K); o.fill(s / K); return o; };
  return (y) => { const o = new Float64Array(K); for (let k = 0; k < K; k++) { let s = 0; const Pk = P[k]; for (let j = 0; j < K; j++) s += Pk[j] * y[j]; o[k] = s; } return o; };
}

/**
 * DerSimonian-Laird method-of-moments tau^2 from releases with known variances, using network averages only.
 * @param {ArrayLike<number>} x releases
 * @param {ArrayLike<number>} r public release variances
 * @param {(y:ArrayLike<number>)=>Float64Array} avg averaging operator
 * @returns {Float64Array} each hospital's estimate of tau^2
 * @see Remark 8.3, Algorithm 1 (first gossip)
 */
export function dersimonianLaird(x, r, avg) {
  const K = x.length, w = map(K, (i) => 1 / r[i]);
  const S1 = avg(w), Sx = avg(map(K, (i) => w[i] * x[i])), Sxx = avg(map(K, (i) => w[i] * x[i] * x[i])), S11 = avg(map(K, (i) => w[i] * w[i]));
  return map(K, (k) => {
    const s1 = Math.max(K * S1[k], 1e-12), Q = K * Sxx[k] - (K * Sx[k]) ** 2 / s1, den = Math.max(s1 - K * S11[k] / s1, 1e-12);
    return Math.max(0, (Q - (K - 1)) / den);
  });
}

const TAU_GRID = (() => { const g = [0]; for (let i = 0; i < 160; i++) g.push(10 ** (-3.5 + 4.5 * i / 159)); return g; })();

/**
 * Posterior mean of tau^2 under a half-Cauchy(scale) prior on tau, mu integrated out under a flat prior
 * (ghb.tau2_bayes). Needs every release, which hospitals can obtain by flooding; it is post-processing.
 * @param {ArrayLike<number>} x releases
 * @param {ArrayLike<number>} r public release variances
 * @param {number} [scale=0.5] half-Cauchy scale
 * @returns {number} E[tau^2 | releases]
 * @see Remark 8.3, Simulation 3
 */
export function tau2Bayes(x, r, scale = 0.5) {
  const K = x.length, G = TAU_GRID.length, lp = new Float64Array(G);
  let mx = -Infinity;
  for (let g = 0; g < G; g++) {
    const t = TAU_GRID[g]; let sw = 0, swx = 0, slv = 0;
    for (let i = 0; i < K; i++) { const V = t * t + r[i]; sw += 1 / V; swx += x[i] / V; slv += Math.log(V); }
    const mu = swx / sw; let Q = 0;
    for (let i = 0; i < K; i++) Q += (x[i] - mu) ** 2 / (t * t + r[i]);
    lp[g] = -0.5 * slv - 0.5 * Math.log(sw) - 0.5 * Q - Math.log1p((t / scale) ** 2);
    if (lp[g] > mx) mx = lp[g];
  }
  let a = 0, b = 0;
  for (let g = 0; g < G; g++) {
    const dt = g === 0 ? TAU_GRID[1] - TAU_GRID[0] : g === G - 1 ? TAU_GRID[G - 1] - TAU_GRID[G - 2] : (TAU_GRID[g + 1] - TAU_GRID[g - 1]) / 2;
    const p = Math.exp(lp[g] - mx) * dt; a += p * TAU_GRID[g] ** 2; b += p;
  }
  return a / b;
}

/**
 * What the network gives each hospital: tau^2, the leave-one-out population estimate and its MSE.
 * @param {ArrayLike<number>} x releases
 * @param {ArrayLike<number>} r public release variances
 * @param {(y:ArrayLike<number>)=>Float64Array} avg averaging operator
 * @param {{tau2?:number|null, estimator?:'dl'|'bayes'|'oracle', scale?:number, weights?:'iv'|'equal'}} [opt]
 *   tau^2 estimator (DerSimonian-Laird by default; 'oracle' uses opt.tau2) and leave-one-out weights
 * @returns {{t2:Float64Array, mu:Float64Array, M:Float64Array}}
 * @see Algorithm 1 (second gossip), Remark 8.3, Proposition 8.4 (iii)
 */
export function borrow(x, r, avg, opt = {}) {
  const K = x.length, est = opt.estimator || 'dl';
  let t2;
  if (est === 'oracle') { t2 = new Float64Array(K); t2.fill(opt.tau2); }
  else if (est === 'bayes') { t2 = new Float64Array(K); t2.fill(tau2Bayes(x, r, opt.scale ?? 0.5)); }
  else t2 = dersimonianLaird(x, r, avg);
  if (opt.weights === 'equal') {
    const Ax = avg(x), Atot = avg(map(K, (i) => t2[i] + r[i]));
    return { t2, mu: map(K, (k) => (K * Ax[k] - x[k]) / (K - 1)), M: map(K, (k) => Math.max((K * Atot[k] - (t2[k] + r[k])) / ((K - 1) ** 2), 1e-12)) };
  }
  const w = map(K, (i) => 1 / (t2[i] + r[i]));
  const Aw = avg(w), Awx = avg(map(K, (i) => w[i] * x[i]));
  const Sw = map(K, (k) => Math.max(K * Aw[k] - w[k], 1e-12));
  return { t2, mu: map(K, (k) => (K * Awx[k] - w[k] * x[k]) / Sw[k]), M: map(K, (k) => 1 / Sw[k]) };
}

/**
 * Combine the network estimate with a knowledge anchor by precision weighting.
 * @param {{t2:Float64Array, mu:Float64Array, M:Float64Array}} b output of borrow
 * @param {number} mu0 anchor value
 * @param {number} s0sq stated anchor variance sigma_0^2
 * @param {boolean} [check=false] drop the anchor where |mu0 - mu_k| / sqrt(s0^2 + M_k) > 3
 * @returns {{t2:Float64Array, mu:Float64Array, M:Float64Array, used:Uint8Array}}
 * @see Algorithm 1 (anchor step), Simulation 11
 */
export function combineAnchor(b, mu0, s0sq, check = false) {
  const K = b.mu.length, used = new Uint8Array(K), mu = new Float64Array(K), M = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    const ok = !check || Math.abs(mu0 - b.mu[k]) / Math.sqrt(s0sq + b.M[k]) <= 3;
    used[k] = ok ? 1 : 0;
    if (ok) { const P = 1 / b.M[k] + 1 / s0sq; mu[k] = (b.mu[k] / b.M[k] + mu0 / s0sq) / P; M[k] = 1 / P; } else { mu[k] = b.mu[k]; M[k] = b.M[k]; }
  }
  return { t2: b.t2, mu, M, used };
}

/**
 * Plug-in shrinkage weights alpha_k = (tau^2_k + M_k) / (tau^2_k + M_k + v_k).
 * @param {ArrayLike<number>} v local variances
 * @param {{t2:ArrayLike<number>, M:ArrayLike<number>}} b network quantities
 * @returns {Float64Array}
 * @see Proposition 8.1
 */
export function shrinkWeights(v, b) { return map(v.length, (k) => alphaHB(v[k], b.t2[k], b.M[k])); }

/**
 * GHB estimate: the noise-free local estimate shrunk toward the network estimate.
 * @param {ArrayLike<number>} th local estimates (never released)
 * @param {ArrayLike<number>} v local variances
 * @param {{t2:ArrayLike<number>, mu:ArrayLike<number>, M:ArrayLike<number>}} b network quantities
 * @returns {Float64Array} alpha_k th_k + (1 - alpha_k) mu_k
 * @see Proposition 8.1, Algorithm 1
 */
export function shrink(th, v, b) {
  const a = shrinkWeights(v, b);
  return map(th.length, (k) => a[k] * th[k] + (1 - a[k]) * b.mu[k]);
}

/**
 * Oracle hierarchical borrowing: true tau^2, equal-weight leave-one-out mean of releases, M of Eq. (6).
 * @param {ArrayLike<number>} th local estimates
 * @param {ArrayLike<number>} x releases
 * @param {ArrayLike<number>} v local variances
 * @param {ArrayLike<number>} s2 noise variances
 * @param {number} tau2 true heterogeneity
 * @returns {Float64Array}
 * @see Propositions 8.1 and 8.2 (ghb.oracle_hb)
 */
export function oracleHB(th, x, v, s2, tau2) {
  const K = th.length; let sx = 0, stot = 0;
  const tot = map(K, (i) => tau2 + v[i] + s2[i]);
  for (let i = 0; i < K; i++) { sx += x[i]; stot += tot[i]; }
  return map(K, (k) => { const mu = (sx - x[k]) / (K - 1), M = (stot - tot[k]) / ((K - 1) ** 2), a = alphaHB(v[k], tau2, M); return a * th[k] + (1 - a) * mu; });
}

/**
 * A single global model: the sample-size weighted mean of all noise-free local estimates.
 * @param {ArrayLike<number>} th local estimates
 * @param {ArrayLike<number>} n sizes
 * @returns {number}
 * @see ghb.full_pooling, Figure 8
 */
export function pooledMean(th, n) { let a = 0, b = 0; for (let i = 0; i < th.length; i++) { a += n[i] * th[i]; b += n[i]; } return a / b; }

/* ---------- atypical hospitals (Simulation 9) ---------- */

const ZG = (() => { const g = []; for (let i = 0; i <= 160; i++) g.push(-8 + i * 0.1); return g; })();

/**
 * Posterior mean under likelihood N(th; theta, v) and a Student-t prior centred at mu with variance varPrior.
 * @param {number} th local estimate
 * @param {number} v local variance
 * @param {number} mu prior centre (leave-one-out network estimate)
 * @param {number} varPrior prior variance tau^2 + M
 * @param {number} [nu=3] degrees of freedom
 * @returns {number} posterior mean by quadrature on 161 points over +-8 sd
 * @see Simulation 9 (robust GHB), Section 8.2 remark on atypical hospitals
 */
export function robustPosteriorMean(th, v, mu, varPrior, nu = 3) {
  const sv = Math.sqrt(v), sc = Math.sqrt(Math.max(varPrior, 1e-12) * (nu - 2) / nu), lw = new Float64Array(ZG.length);
  let mx = -Infinity;
  for (let i = 0; i < ZG.length; i++) { const q = (th + sv * ZG[i] - mu) / sc; lw[i] = -0.5 * ZG[i] * ZG[i] - (nu + 1) / 2 * Math.log1p(q * q / nu); if (lw[i] > mx) mx = lw[i]; }
  let a = 0, b = 0;
  for (let i = 0; i < ZG.length; i++) { const w = Math.exp(lw[i] - mx); a += w * (th + sv * ZG[i]); b += w; }
  return a / b;
}

/**
 * Opt-out rule: keep the local estimate when it conflicts with the network.
 * @param {number} th local estimate
 * @param {number} ghbEst GHB estimate
 * @param {number} v local variance
 * @param {number} mu network estimate
 * @param {number} t2 tau^2 estimate
 * @param {number} M network MSE
 * @param {number} [z=3] threshold
 * @returns {number} th if |th - mu| / sqrt(v + t2 + M) > z, otherwise ghbEst
 * @see Simulation 9
 */
export function optOut(th, ghbEst, v, mu, t2, M, z = 3) { return Math.abs(th - mu) / Math.sqrt(v + t2 + M) > z ? th : ghbEst; }

/* ---------- dishonest hospitals (Simulation 12) ---------- */

/** @returns {number} median of an array */
export function median(arr) { const s = Array.from(arr).sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

/**
 * Weighted median: first sorted value whose cumulative normalized weight reaches 1/2 (exp12.weighted_median).
 * @param {ArrayLike<number>} x values
 * @param {ArrayLike<number>} w weights
 * @returns {number}
 * @see Simulation 12
 */
export function weightedMedian(x, w) {
  const K = x.length, o = Array.from({ length: K }, (_, i) => i).sort((a, b) => x[a] - x[b]);
  let tw = 0; for (const i of o) tw += w[i];
  let c = 0, idx = 0; for (const i of o) { c += w[i]; if (c / tw < 0.5) idx++; }
  return x[o[Math.min(idx, K - 1)]];
}

/**
 * Robust screening then GHB: drop releases far from the weighted median (MAD spread), re-estimate on the rest.
 * @param {ArrayLike<number>} th local estimates
 * @param {ArrayLike<number>} x releases (some may be dishonest)
 * @param {ArrayLike<number>} r public release variances
 * @param {ArrayLike<number>} v local variances
 * @param {number} [z=3] screening threshold
 * @returns {{est: Float64Array, keep: boolean[]}}
 * @see Simulation 12 (robust GHB)
 */
export function robustScreenGHB(th, x, r, v, z = 3) {
  const K = x.length, m = weightedMedian(x, map(K, (i) => 1 / r[i]));
  const mad = median(map(K, (i) => Math.abs(x[i] - m))), t0 = Math.max((1.4826 * mad) ** 2 - median(r), 0);
  const keep = Array.from({ length: K }, (_, i) => Math.abs(x[i] - m) / Math.sqrt(r[i] + t0) <= z);
  const xk = [], rk = [];
  for (let i = 0; i < K; i++) if (keep[i]) { xk.push(x[i]); rk.push(r[i]); }
  const t2 = xk.length > 1 ? dersimonianLaird(xk, rk, avgOp(null, xk.length))[0] : 0;
  const w = map(K, (i) => (keep[i] ? 1 / (t2 + r[i]) : 0));
  let sw = 0, swx = 0; for (let i = 0; i < K; i++) { sw += w[i]; swx += w[i] * x[i]; }
  return {
    est: map(K, (k) => { const S = Math.max(sw - w[k], 1e-12), mu = (swx - w[k] * x[k]) / S, M = 1 / S, a = alphaHB(v[k], t2, M); return a * th[k] + (1 - a) * mu; }),
    keep
  };
}

/* ---------- joint release (Simulation 5) ---------- */

/**
 * Release units when hospitals no larger than `cut` release jointly in groups of g (secure aggregation);
 * the others release alone.
 * @param {number[]} sizes hospital sizes
 * @param {number} g group size
 * @param {number} cut largest size that joins a group
 * @returns {number[][]} member sizes of each release unit
 * @see Proposition 8.2 (second consequence), Simulation 5
 */
export function releaseUnits(sizes, g, cut) {
  const small = sizes.filter((n) => n <= cut).sort((a, b) => a - b), large = sizes.filter((n) => n > cut), units = [];
  for (let i = 0; i < small.length; i += g) units.push(small.slice(i, i + g));
  for (const n of large) units.push([n]);
  return units;
}

/**
 * Borrowed patients when the population estimate is the inverse-variance mean of unit releases; a unit's
 * release is the size-weighted mean of its members with noise calibrated to the unit's total size.
 * @param {number[][]} units member sizes per release unit
 * @param {{tau2:number, sigma2?:number, eps:number, delta:number, C:number}} q parameters
 * @returns {number} B = sigma^2 / (tau^2 + M), M = 1 / sum_units 1/Var(unit)
 * @see Proposition 8.2 (second consequence), Simulation 5 (exp5_joint_release.theory_M)
 */
export function jointReleaseB(units, q) {
  const sg = q.sigma2 ?? 1; let inv = 0;
  for (const u of units) {
    const N = u.reduce((a, b) => a + b, 0), s2n = u.reduce((a, b) => a + b * b, 0);
    inv += 1 / (q.tau2 * s2n / (N * N) + sg / N + dpVar(N, q.eps, q.delta, q.C));
  }
  return sg / (q.tau2 + 1 / inv);
}

/**
 * Mean borrowed patients over hospitals for a given release-noise variance function, with the inverse-variance leave-one-out mean.
 * @param {number[]} sizes sizes
 * @param {number} tau2 heterogeneity
 * @param {(n:number)=>number} s2fn release noise variance for a hospital of size n
 * @param {number} [sigma2=1] per-patient variance
 * @returns {number} mean_k sigma^2 / (tau^2 + M_k)
 * @see Eq. (5), Simulation 14
 */
export function meanBorrowed(sizes, tau2, s2fn, sigma2 = 1) {
  const tot = sizes.map((n) => tau2 + sigma2 / n + s2fn(n));
  let b = 0; for (let k = 0; k < sizes.length; k++) b += borrowedPatients(tau2, looMSEiv(tot, k), sigma2);
  return b / sizes.length;
}

/* ---------- GMRF prior (Eq. 7) ---------- */

/**
 * GMRF estimate by Jacobi iteration on privatized releases, then each hospital's local update with its own estimate.
 * @param {ArrayLike<number>} th local estimates (used only in the final local step)
 * @param {ArrayLike<number>} x releases
 * @param {ArrayLike<number>} v local variances
 * @param {ArrayLike<number>} r release variances v + s^2
 * @param {Float64Array[]} A similarity weights a_kl
 * @param {number} beta smoothing strength
 * @param {number} [gamma=0] anchor strength
 * @param {number} [mu0=0] anchor
 * @param {number} [sweeps=100] Jacobi sweeps
 * @returns {Float64Array}
 * @see Eq. (7) and the Jacobi update of Section 8.4; Simulation 6
 */
export function gmrfJacobi(th, x, v, r, A, beta, gamma = 0, mu0 = 0, sweeps = 100) {
  const K = x.length, d = A.map((row) => row.reduce((s, a) => s + a, 0)), nb = A.map((row) => { const l = []; row.forEach((a, j) => { if (a) l.push([j, a]); }); return l; });
  const nsum = (z, k) => { let s = 0; for (const [j, a] of nb[k]) s += a * z[j]; return s; };
  let z = Float64Array.from(x);
  for (let s = 0; s < sweeps; s++) z = map(K, (k) => (x[k] / r[k] + beta * nsum(z, k) + gamma * mu0) / (1 / r[k] + beta * d[k] + gamma));
  return map(K, (k) => (th[k] / v[k] + beta * nsum(z, k) + gamma * mu0) / (1 / v[k] + beta * d[k] + gamma));
}
