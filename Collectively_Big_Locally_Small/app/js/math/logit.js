/**
 * math/logit.js
 * Responsibility: the logistic-regression heads of the paper's simulations (Section 6.3; simulations/exp7_logistic.py
 * and simulations/logit_world.py): constants of the logistic world, bounded features, the MAP fit under a Gaussian
 * prior (diagonal or full precision), excess log-loss against the true model, AUC, the zCDP budget, the public
 * feature moments used by GHB-lite, and the Laplace-evidence mixtures of Remark 4.4. A data set is {X, y, n} with
 * X a row-major Float64Array of n rows and P columns (intercept first).
 */
import { solve, inv, logdet } from './linalg.js';
import { zcdpRho } from './dp.js';

/** Coefficients per head (intercept and four features). @see exp7_logistic.P */
export const P = 5;
/** Bound on ||x||: features are clipped so that ||x|| <= sqrt(5). @see exp7_logistic.R_X */
export const R_X = Math.sqrt(5);
/** Population mean of the coefficients, mu. @see exp7_logistic.MU */
export const MU = Object.freeze([-1.0, 0.8, -0.6, 0.4, 0.2]);
/** Ridge of the legacy release, kept for the private cluster assignment. @see exp7_logistic.LAM */
export const LAM = 0.2;
/** Fresh test patients per hospital and replicate. @see exp7_logistic.N_TEST */
export const N_TEST = 4000;

/** @returns {number} logistic sigmoid */
export const sigmoid = (s) => (s >= 0 ? 1 / (1 + Math.exp(-s)) : Math.exp(s) / (1 + Math.exp(s)));
/** @returns {number} log(1 + e^s), stable (numpy.logaddexp(0, s)) */
export const softplus = (s) => (s > 0 ? s + Math.log1p(Math.exp(-s)) : Math.log1p(Math.exp(s)));

/**
 * Total zero-concentrated budget whose (epsilon, delta) conversion rho + 2 sqrt(rho ln(1/delta)) equals epsilon.
 * Infinity when there is no privacy (eps = Infinity).
 * @param {number} eps epsilon
 * @param {number} delta delta
 * @returns {number} rho
 * @see Proposition 5.1; ghb_logit.rho_total and literature.rho_total (brentq root, here in closed form)
 */
export function rhoTotal(eps, delta) { return isFinite(eps) ? zcdpRho(eps, delta) : Infinity; }

/**
 * Bounded features: an intercept and P - 1 standard normals (shifted by a hospital's case-mix mean), scaled so that
 * the four features have norm at most 2, hence ||x|| <= sqrt(5).
 * @param {{n:()=>number}} rng generator
 * @param {number} m rows
 * @param {ArrayLike<number>|null} [mean] case-mix mean of the four features
 * @returns {Float64Array} m x P row-major
 * @see logit_world.features
 */
export function features(rng, m, mean = null) {
  const X = new Float64Array(m * P);
  for (let i = 0; i < m; i++) {
    const o = i * P; let s = 0;
    X[o] = 1;
    for (let d = 1; d < P; d++) { const z = rng.n() + (mean ? mean[d - 1] : 0); X[o + d] = z; s += z * z; }
    const f = Math.min(1, 2 / Math.sqrt(s));
    if (f < 1) for (let d = 1; d < P; d++) X[o + d] *= f;
  }
  return X;
}

/**
 * Scores s = X theta.
 * @param {Float64Array} X design
 * @param {ArrayLike<number>} th coefficients
 * @returns {Float64Array}
 */
export function scores(X, th) {
  const m = X.length / P, s = new Float64Array(m);
  for (let i = 0; i < m; i++) { const o = i * P; s[i] = X[o] * th[0] + X[o + 1] * th[1] + X[o + 2] * th[2] + X[o + 3] * th[3] + X[o + 4] * th[4]; }
  return s;
}

/**
 * Mean gradient of the logistic loss, X^T (sigmoid(X theta) - y) / n.
 * @param {{X:Float64Array, y:Float64Array, n:number}} d data
 * @param {ArrayLike<number>} th point
 * @returns {Float64Array}
 * @see literature.grad
 */
export function grad(d, th) {
  const { X, y, n } = d, g = new Float64Array(P);
  for (let i = 0; i < n; i++) {
    const o = i * P, r = sigmoid(X[o] * th[0] + X[o + 1] * th[1] + X[o + 2] * th[2] + X[o + 3] * th[3] + X[o + 4] * th[4]) - y[i];
    for (let a = 0; a < P; a++) g[a] += X[o + a] * r;
  }
  for (let a = 0; a < P; a++) g[a] /= n;
  return g;
}

/**
 * Summed score at a point, X^T (y - sigmoid(X c)) (not divided by n).
 * @param {{X:Float64Array, y:Float64Array, n:number}} d data
 * @param {ArrayLike<number>} c point
 * @returns {Float64Array}
 * @see Section 5.2 (psi_i at the origin), Section 5.3 (g_k at the anchor)
 */
export function scoreSum(d, c) { const g = grad(d, c); for (let a = 0; a < P; a++) g[a] *= -d.n; return g; }

/**
 * Information matrix sum X^T X / 4 (the information at the origin).
 * @param {{X:Float64Array, n:number}} d data
 * @returns {Float64Array} P x P
 * @see Section 5.2, phi_i = x_i x_i^T / 4
 */
export function infoAtZero(d) {
  const { X, n } = d, J = new Float64Array(P * P);
  for (let i = 0; i < n; i++) { const o = i * P; for (let a = 0; a < P; a++) { const xa = X[o + a] / 4; for (let b = a; b < P; b++) J[a * P + b] += xa * X[o + b]; } }
  for (let a = 0; a < P; a++) for (let b = 0; b < a; b++) J[a * P + b] = J[b * P + a];
  return J;
}

function objective(X, y, n, t, m, prec, full) {
  let f = 0;
  for (let i = 0; i < n; i++) { const o = i * P, s = X[o] * t[0] + X[o + 1] * t[1] + X[o + 2] * t[2] + X[o + 3] * t[3] + X[o + 4] * t[4]; f += softplus(s) - y[i] * s; }
  if (full) { for (let a = 0; a < P; a++) { let r = 0; for (let b = 0; b < P; b++) r += prec[a * P + b] * (t[b] - m[b]); f += 0.5 * (t[a] - m[a]) * r; } }
  else for (let a = 0; a < P; a++) f += 0.5 * prec[a] * (t[a] - m[a]) ** 2;
  return f;
}

function newton(d, m, prec, full, iters, tol) {
  const { X, y, n } = d, th = Float64Array.from(m), H = new Float64Array(P * P), g = new Float64Array(P), cand = new Float64Array(P);
  let f = objective(X, y, n, th, m, prec, full);
  for (let it = 0; it < iters; it++) {
    H.fill(0); g.fill(0);
    for (let i = 0; i < n; i++) {
      const o = i * P, p = sigmoid(X[o] * th[0] + X[o + 1] * th[1] + X[o + 2] * th[2] + X[o + 3] * th[3] + X[o + 4] * th[4]), r = p - y[i], w = p * (1 - p);
      for (let a = 0; a < P; a++) { const xa = X[o + a]; g[a] += xa * r; const wa = w * xa; for (let b = a; b < P; b++) H[a * P + b] += wa * X[o + b]; }
    }
    for (let a = 0; a < P; a++) for (let b = 0; b < a; b++) H[a * P + b] = H[b * P + a];
    if (full) { for (let a = 0; a < P; a++) { let r = 0; for (let b = 0; b < P; b++) { r += prec[a * P + b] * (th[b] - m[b]); H[a * P + b] += prec[a * P + b]; } g[a] += r; } }
    else for (let a = 0; a < P; a++) { g[a] += prec[a] * (th[a] - m[a]); H[a * P + a] += prec[a]; }
    const step = solve(H, g, P);
    let gs = 0; for (let a = 0; a < P; a++) gs += g[a] * step[a];
    let al = 1, fNew = f;
    while (al > 1e-8) {
      for (let a = 0; a < P; a++) cand[a] = th[a] - al * step[a];
      fNew = objective(X, y, n, cand, m, prec, full);
      if (fNew <= f - 1e-4 * al * gs) break;
      al /= 2;
    }
    let big = 0;
    for (let a = 0; a < P; a++) { const dlt = al * step[a]; th[a] -= dlt; big = Math.max(big, Math.abs(dlt)); }
    f = fNew;
    if (big < tol) break;
  }
  return th;
}

/**
 * MAP of logistic regression under the prior N(m, diag(1/prec)), by damped Newton with Armijo backtracking.
 * @param {{X:Float64Array, y:Float64Array, n:number}} d data
 * @param {ArrayLike<number>} m prior mean
 * @param {ArrayLike<number>} prec prior precision per coefficient
 * @param {number} [iters=100] Newton iterations
 * @returns {Float64Array}
 * @see exp7_logistic.fit; Proposition 4.1 for heads (the posterior mode given a borrowed prior)
 */
export function fit(d, m, prec, iters = 100) { return newton(d, m, prec, false, iters, 1e-9); }

/**
 * MAP of logistic regression under N(m, Prec^{-1}) with a full precision matrix.
 * @param {{X:Float64Array, y:Float64Array, n:number}} d data
 * @param {ArrayLike<number>} m prior mean
 * @param {Float64Array} Prec P x P precision
 * @param {number} [iters=60] Newton iterations
 * @returns {Float64Array}
 * @see ghb_v3.fit_mat; Section 5.2 (fit under N(mu_-k, diag(tau_q^2) + V_-k))
 */
export function fitMat(d, m, Prec, iters = 60) { return newton(d, m, Prec, true, iters, 1e-8); }

function loglikHess(d, t, H) {
  const { X, y, n } = d; let ll = 0;
  for (let i = 0; i < n; i++) {
    const o = i * P, s = X[o] * t[0] + X[o + 1] * t[1] + X[o + 2] * t[2] + X[o + 3] * t[3] + X[o + 4] * t[4], p = sigmoid(s), w = p * (1 - p);
    ll += y[i] * s - softplus(s);
    for (let a = 0; a < P; a++) { const wa = w * X[o + a]; for (let b = a; b < P; b++) H[a * P + b] += wa * X[o + b]; }
  }
  for (let a = 0; a < P; a++) for (let b = 0; b < a; b++) H[a * P + b] = H[b * P + a];
  return ll;
}

/**
 * Posterior mean under a mixture of diagonal Gaussian priors, by the Laplace approximation of each component's
 * marginal likelihood. Each (mean, variance) is multiplied by the default prior N(0, sd^2) for every sd in priorSd.
 * @param {{X:Float64Array, y:Float64Array, n:number}} d hospital data
 * @param {ArrayLike<number>[]} means component means
 * @param {ArrayLike<number>[]} variances component variances per coefficient
 * @param {number|number[]} [priorSd=2.5] default prior scale(s)
 * @returns {Float64Array}
 * @see Remark 4.4 (tau integrated out against the hospital's own evidence); ghb_logit.combine
 */
export function combineDiag(d, means, variances, priorSd = 2.5) {
  const sds = Array.isArray(priorSd) ? priorSd : [priorSd], fits = [], lw = [];
  for (let c = 0; c < means.length; c++) for (const sd of sds) {
    const pr = new Float64Array(P), mm = new Float64Array(P);
    for (let a = 0; a < P; a++) { const v = Math.max(variances[c][a], 1e-8); pr[a] = 1 / v + 1 / sd ** 2; mm[a] = (means[c][a] / v) / pr[a]; }
    const t = fit(d, mm, pr), H = new Float64Array(P * P);
    let l = loglikHess(d, t, H), q = 0, lp = 0;
    for (let a = 0; a < P; a++) { H[a * P + a] += pr[a]; q += pr[a] * (t[a] - mm[a]) ** 2; lp += Math.log(pr[a]); }
    l += -0.5 * q + 0.5 * lp - 0.5 * logdet(H, P);
    fits.push(t); lw.push(l);
  }
  return mixture(fits, lw);
}

/**
 * Posterior mean under a mixture of full-covariance Gaussian priors with prior weights (Laplace evidence).
 * @param {{X:Float64Array, y:Float64Array, n:number}} d hospital data
 * @param {Array<{logw:number, m:ArrayLike<number>, C:Float64Array}>} comps components
 * @returns {Float64Array}
 * @see Remark 4.4, Section 5.2 (mixture over tau_q, default scales and non-borrowing components); ghb_v3.combine
 */
export function combineFull(d, comps) {
  const fits = [], lw = [];
  for (const { logw, m, C } of comps) {
    const Prec = inv(C, P), t = fitMat(d, m, Prec), H = new Float64Array(P * P);
    let l = loglikHess(d, t, H), q = 0;
    for (let a = 0; a < P; a++) { let r = 0; for (let b = 0; b < P; b++) { r += Prec[a * P + b] * (t[b] - m[b]); H[a * P + b] += Prec[a * P + b]; } q += (t[a] - m[a]) * r; }
    lw.push(logw + l - 0.5 * q + 0.5 * logdet(Prec, P) - 0.5 * logdet(H, P));
    fits.push(t);
  }
  return mixture(fits, lw);
}

function mixture(fits, lw) {
  const mx = Math.max(...lw), out = new Float64Array(P); let sw = 0;
  for (let c = 0; c < fits.length; c++) { const w = Math.exp(lw[c] - mx); sw += w; for (let a = 0; a < P; a++) out[a] += w * fits[c][a]; }
  for (let a = 0; a < P; a++) out[a] /= sw;
  return out;
}

/**
 * Area under the ROC curve from ranks (ties broken by order, as exp7_logistic.auc).
 * @param {ArrayLike<number>} s scores
 * @param {ArrayLike<number>} y labels in {0, 1}
 * @returns {number}
 * @see Section 6.3 (AUC column of the comparisons)
 */
export function auc(s, y) {
  const m = s.length, idx = new Uint32Array(m); for (let i = 0; i < m; i++) idx[i] = i;
  idx.sort((a, b) => s[a] - s[b]);
  let r1 = 0, n1 = 0;
  for (let r = 0; r < m; r++) if (y[idx[r]] === 1) { r1 += r + 1; n1++; }
  const n0 = m - n1;
  return (r1 - n1 * (n1 + 1) / 2) / (n1 * n0);
}

/**
 * Test set of one hospital: features, true probabilities, labels and the entropy of the true model (the Bayes
 * log-loss), so that the excess log-loss of any coefficient vector is exact in expectation over the labels.
 * @param {{n:()=>number, u:()=>number}} rng generator
 * @param {ArrayLike<number>} theta true coefficients
 * @param {ArrayLike<number>|null} mean case-mix mean
 * @param {number} [m=N_TEST] patients
 * @returns {{X:Float64Array, pt:Float64Array, y:Uint8Array, bayes:number}}
 * @see logit_world.evaluate
 */
export function testSet(rng, theta, mean, m = N_TEST) {
  const X = features(rng, m, mean), s = scores(X, theta), pt = new Float64Array(m), y = new Uint8Array(m);
  let b = 0;
  for (let i = 0; i < m; i++) { const p = sigmoid(s[i]); pt[i] = p; y[i] = rng.u() < p ? 1 : 0; b += -(p * Math.log(p) + (1 - p) * Math.log1p(-p)); }
  return { X, pt, y, bayes: b / m };
}

/**
 * Excess log-loss of coefficients th on a test set: E[log-loss | x] under the true model minus its entropy.
 * @param {{X:Float64Array, pt:Float64Array, bayes:number}} t test set
 * @param {ArrayLike<number>} th coefficients
 * @param {boolean} [withAuc=false] also return the AUC on the sampled labels
 * @returns {{kl:number, auc:number}}
 * @see logit_world.evaluate; Section 6.3 (mean excess log-loss)
 */
export function excess(t, th, withAuc = false) {
  const s = scores(t.X, th), m = s.length; let e = 0;
  for (let i = 0; i < m; i++) e += softplus(s[i]) - t.pt[i] * s[i];
  return { kl: e / m - t.bayes, auc: withAuc ? auc(s, t.y) : NaN };
}

/** Moments of the public feature distribution, from m draws (ghb_logit.Public). */
export class Public {
  /** @param {Float64Array} X public draws (m x P) */
  constructor(X) { this.X = X; this.m = X.length / P; }

  stats(th, c = null) {
    const { X, m } = this, mo = new Float64Array(P), I = new Float64Array(P * P), E = c ? new Float64Array(P * P) : null, mc = c ? new Float64Array(P) : null;
    for (let i = 0; i < m; i++) {
      const o = i * P, pt = sigmoid(X[o] * th[0] + X[o + 1] * th[1] + X[o + 2] * th[2] + X[o + 3] * th[3] + X[o + 4] * th[4]), w = pt * (1 - pt);
      let we = 0;
      if (c) { const pc = sigmoid(X[o] * c[0] + X[o + 1] * c[1] + X[o + 2] * c[2] + X[o + 3] * c[3] + X[o + 4] * c[4]); we = w + (pt - pc) ** 2; for (let a = 0; a < P; a++) mc[a] += X[o + a] * pc; }
      for (let a = 0; a < P; a++) {
        const xa = X[o + a]; mo[a] += xa * pt;
        for (let b = a; b < P; b++) { const xx = xa * X[o + b]; I[a * P + b] += w * xx; if (c) E[a * P + b] += we * xx; }
      }
    }
    for (let a = 0; a < P; a++) { mo[a] /= m; if (c) mc[a] /= m; for (let b = a; b < P; b++) { I[a * P + b] /= m; I[b * P + a] = I[a * P + b]; if (c) { E[a * P + b] /= m; E[b * P + a] = E[a * P + b]; } } }
    if (c) for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) E[a * P + b] -= (mo[a] - mc[a]) * (mo[b] - mc[b]);
    return { m: mo, info: I, cov: E };
  }

  /**
   * Moment map m(theta) = E[x sigmoid(x' theta)].
   * @param {ArrayLike<number>} th point
   * @returns {Float64Array}
   * @see Section 5.3 (E g_k = m(theta_k) - m(mu_0))
   */
  mean(th) { return this.stats(th).m; }

  /**
   * Inverse moment map: solve m(theta) + ridge theta = target by damped Newton steps.
   * @param {ArrayLike<number>} target target moment
   * @param {ArrayLike<number>} start starting point
   * @param {number} [ridge=0] Gaussian prior precision at 0
   * @param {number} [iters=60] iterations
   * @returns {Float64Array}
   * @see Section 5.3 (pooled moment estimate); ghb_logit.Public.invert
   */
  invert(target, start, ridge = 0, iters = 60) {
    const th = Float64Array.from(start);
    for (let it = 0; it < iters; it++) {
      const { m, info } = this.stats(th), A = Float64Array.from(info), rhs = new Float64Array(P);
      for (let a = 0; a < P; a++) { A[a * P + a] += ridge; rhs[a] = target[a] - m[a] - ridge * th[a]; }
      const step = solve(A, rhs, P);
      let big = 0; for (let a = 0; a < P; a++) big = Math.max(big, Math.abs(step[a]));
      const f = big > 1 ? 1 / big : 1;
      for (let a = 0; a < P; a++) th[a] = Math.min(8, Math.max(-8, th[a] + f * step[a]));
      if (big < 1e-10) break;
    }
    return th;
  }
}

