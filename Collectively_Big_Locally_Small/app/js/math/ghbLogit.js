/**
 * math/ghbLogit.js
 * Responsibility: GHB and GHB-lite for logistic-regression heads (Section 5). GHB (Section 5.2, Algorithm 1) makes
 * two joint releases, the privatized sums of every patient's score and information at the origin; each hospital
 * turns them locally into a leave-one-out prior with the privacy covariance propagated and integrates the unknown
 * heterogeneity out against its own evidence (simulations/ghb_v4.py with the frozen configuration f = 1, R1 = 1,
 * no-borrowing components, published anchor). GHB-lite (Section 5.3) has every hospital release its own mean score
 * at the anchor, inverted under the public feature distribution (simulations/ghb_logit.py). Arrays of per-hospital
 * vectors are arrays of Float64Array(P).
 */
import { P, Public, scoreSum, infoAtZero, combineDiag, combineFull, rhoTotal } from './logit.js';
import { inv, mv, mul, add, eye, diag, diagOf, psd } from './linalg.js';
import { borrow, GRID, DGRID, gridQuantiles, priorQuantiles } from './ghb.js';

export { GRID, gridQuantiles, priorQuantiles };

/** Bound G on ||x|| (per-patient score norm). @see Section 5.2 */
export const G = Math.sqrt(5);
/** Quantiles of tau used as mixture components. @see Remark 4.4, Section 5.2 (Q = 12) */
export const Q = 12;
/** Half-Cauchy scale of the prior on tau. @see Remark 4.4 */
export const SCALE = 0.5;
/** Weakly informative default prior scale s0 for logistic coefficients. @see Section 5.2 (s0 = 2.5) */
export const S0 = 2.5;
/** Default-prior scales integrated out by GHB. @see Section 5.2, s in {0.5, 1, 2.5} */
export const SDS = Object.freeze([0.5, 1.0, 2.5]);
/** Anchor conflict threshold. @see Section 5.2, |mu_0 - w| > 3 sqrt(sigma_0^2 + V) */
export const Z_CONFLICT = 3;

/**
 * Posterior quantiles of tau^2 given releases x with variances r, half-Cauchy prior, mu integrated out (flat prior).
 * @param {ArrayLike<number>} x releases
 * @param {ArrayLike<number>} r release variances
 * @param {number} [q=Q] quantiles
 * @returns {Float64Array} tau_q^2
 * @see Remark 4.4; ghb_logit.tau2_quantiles
 */
export function posteriorQuantiles(x, r, q = Q) {
  const K = x.length, lp = new Float64Array(GRID.length); let mx = -Infinity;
  for (let g = 0; g < GRID.length; g++) {
    const t = GRID[g]; let sw = 0, swx = 0, slv = 0;
    for (let i = 0; i < K; i++) { const V = t * t + r[i]; sw += 1 / V; swx += x[i] / V; slv += Math.log(V); }
    const mu = swx / sw; let Qs = 0;
    for (let i = 0; i < K; i++) Qs += (x[i] - mu) ** 2 / (t * t + r[i]);
    lp[g] = -0.5 * slv - 0.5 * Math.log(sw) - 0.5 * Qs - Math.log1p((t / SCALE) ** 2);
    if (lp[g] > mx) mx = lp[g];
  }
  const p = Float64Array.from(lp, (l, g) => Math.exp(l - mx) * DGRID[g]), s = p.reduce((a, b) => a + b, 0);
  return gridQuantiles(p.map((v) => v / s), q);
}

const col = (A, d) => Float64Array.from(A, (v) => v[d]);

/* ================= GHB-lite: one release per hospital (Section 5.3) ================= */

/**
 * GHB-lite release: each hospital's mean score at the anchor plus Gaussian noise, s_k = 2G / (n_k sqrt(2 rho)).
 * @param {{n:()=>number}} rng generator
 * @param {Array<{X:Float64Array,y:Float64Array,n:number}>} data hospitals
 * @param {ArrayLike<number>} mu0 anchor
 * @param {number} eps epsilon (Infinity: no privacy)
 * @param {number} [delta=1e-5] delta
 * @returns {{g:Float64Array[], sd:Float64Array}}
 * @see Section 5.3 (g_k), Proposition 4.2 scope (d); ghb_logit.release
 */
export function liteRelease(rng, data, mu0, eps, delta = 1e-5) {
  const rho = rhoTotal(eps, delta), sd = Float64Array.from(data, (d) => (isFinite(rho) ? 2 * G / d.n / Math.sqrt(2 * rho) : 0));
  const g = data.map((d, k) => { const s = scoreSum(d, mu0); for (let a = 0; a < P; a++) s[a] = s[a] / d.n + sd[k] * rng.n(); return s; });
  return { g, sd };
}

/**
 * Linearized inversion theta~_j = a_j + I(a_j)^{-1}(g_j + m(mu_0) - m(a_j)) and its delta-method variances.
 * @param {Public} pub public moments
 * @param {Float64Array[]} g releases
 * @param {ArrayLike<number>} sd noise sds
 * @param {ArrayLike<number>} n sizes
 * @param {ArrayLike<number>} mu0 anchor
 * @param {ArrayLike<number>[]} points linearization point per hospital (the same object repeated is computed once)
 * @param {number|null} [priorSd=null] information under the default prior N(0, priorSd^2) when given
 * @returns {{th:Float64Array[], r:Float64Array[]}}
 * @see Section 5.3; ghb_logit.linearize
 */
export function liteLinearize(pub, g, sd, n, mu0, points, priorSd = null) {
  const m0 = pub.mean(mu0), th = [], r = [], cache = new Map();
  for (let j = 0; j < n.length; j++) {
    const a = points[j];
    if (!cache.has(a)) cache.set(a, pub.stats(a, mu0));
    const st = cache.get(a), extra = priorSd == null ? 0 : 1 / (n[j] * priorSd ** 2);
    const Ii = inv(add(st.info, eye(P, extra)), P), rhs = new Float64Array(P);
    for (let b = 0; b < P; b++) rhs[b] = g[j][b] + m0[b] - st.m[b];
    const step = mv(Ii, rhs, P);
    th.push(Float64Array.from(a, (x, b) => x + step[b]));
    const Sig = add(st.cov.map((v) => v / n[j]), eye(P, sd[j] ** 2));
    r.push(diagOf(mul(mul(Ii, Sig, P), Ii, P), P)); // Ii symmetric, so Ii Sig Ii^T = Ii Sig Ii
  }
  return { th, r };
}

/**
 * Coordinatewise borrowing at given tau^2 per coefficient: inverse-variance leave-one-out mean through the gossip
 * operator, and its error M.
 * @param {Float64Array[]} th releases on the parameter scale
 * @param {Float64Array[]} r their variances
 * @param {(y:ArrayLike<number>)=>Float64Array} avg averaging operator (T gossip rounds)
 * @param {ArrayLike<number>} tau2 tau^2 per coefficient
 * @returns {{mu:Float64Array[], M:Float64Array[]}}
 * @see Proposition 4.2 (third consequence), Proposition 4.3 (iii); ghb_logit.network
 */
export function liteNetwork(th, r, avg, tau2) {
  const K = th.length, mu = th.map(() => new Float64Array(P)), M = th.map(() => new Float64Array(P));
  for (let d = 0; d < P; d++) {
    const b = borrow(col(th, d), col(r, d), avg, { estimator: 'oracle', tau2: tau2[d] });
    for (let k = 0; k < K; k++) { mu[k][d] = b.mu[k]; M[k][d] = b.M[k]; }
  }
  return { mu, M };
}

/**
 * Precision-weighted combination with the anchor, except where the conflict decision dropped it.
 * @param {{mu:Float64Array[], M:Float64Array[]}} b network estimates
 * @param {ArrayLike<number>} mu0 anchor
 * @param {number|null} asd stated anchor sd (null: no anchor)
 * @param {boolean[]} drop per-hospital decisions
 * @returns {{mu:Float64Array[], M:Float64Array[]}}
 * @see Section 5.1 (layer L3, knowledge anchor); ghb_logit.anchor
 */
export function liteAnchor(b, mu0, asd, drop) {
  if (asd == null) return b;
  const mu = [], M = [];
  for (let k = 0; k < b.mu.length; k++) {
    if (drop[k]) { mu.push(b.mu[k]); M.push(b.M[k]); continue; }
    const m = new Float64Array(P), v = new Float64Array(P);
    for (let d = 0; d < P; d++) { const pr = 1 / b.M[k][d] + 1 / asd ** 2; m[d] = (b.mu[k][d] / b.M[k][d] + mu0[d] / asd ** 2) / pr; v[d] = 1 / pr; }
    mu.push(m); M.push(v);
  }
  return { mu, M };
}

/**
 * Hospitals for which the anchor conflicts with the anchor-free network estimate (|z| > 3 in some coefficient).
 * @param {{mu:Float64Array[], M:Float64Array[]}} b network estimates
 * @param {ArrayLike<number>} mu0 anchor
 * @param {number|null} asd stated anchor sd
 * @param {boolean} check whether the check is on
 * @returns {boolean[]}
 * @see Section 5.2 and Simulation 11 (conflict check); ghb_logit.conflicts
 */
export function liteConflicts(b, mu0, asd, check) {
  return b.mu.map((m, k) => {
    if (asd == null || !check) return false;
    let z = 0; for (let d = 0; d < P; d++) z = Math.max(z, Math.abs(mu0[d] - m[d]) / Math.sqrt(asd ** 2 + b.M[k][d]));
    return z > Z_CONFLICT;
  });
}

/**
 * Releases mapped to the parameter scale: pooled moment inversion, first linearization, anchor decision on the
 * anchor-free leave-one-out estimate, second linearization at each hospital's (anchored) network estimate.
 * @param {Public} pub public moments
 * @param {Float64Array[]} g releases
 * @param {ArrayLike<number>} sd noise sds
 * @param {ArrayLike<number>} n sizes
 * @param {ArrayLike<number>} mu0 anchor
 * @param {Function} avg gossip operator
 * @param {number|null} [asd=0.3] stated anchor sd
 * @param {boolean} [check=true] conflict check
 * @returns {{th:Float64Array[], r:Float64Array[], drop:boolean[]}}
 * @see Section 5.3; ghb_logit.invert
 */
export function liteInvert(pub, g, sd, n, mu0, avg, asd = 0.3, check = true) {
  const N = Array.from(n).reduce((a, b) => a + b, 0), target = pub.mean(mu0);
  for (let j = 0; j < n.length; j++) for (let a = 0; a < P; a++) target[a] += n[j] * g[j][a] / N;
  const a0 = pub.invert(target, new Float64Array(P));
  const first = liteLinearize(pub, g, sd, n, mu0, Array(n.length).fill(a0));
  const t2 = posteriorQuantilesPer(first.th, first.r), med = Float64Array.from({ length: P }, (_, d) => t2[Math.floor(Q / 2)][d]);
  const net = liteNetwork(first.th, first.r, avg, med);
  const drop = liteConflicts(net, mu0, asd, check);
  const pts = liteAnchor(net, mu0, asd, drop).mu;
  const { th, r } = liteLinearize(pub, g, sd, n, mu0, pts);
  return { th, r, drop };
}

function posteriorQuantilesPer(th, r) {
  const out = Array.from({ length: Q }, () => new Float64Array(P));
  for (let d = 0; d < P; d++) { const q = posteriorQuantiles(col(th, d), col(r, d)); for (let j = 0; j < Q; j++) out[j][d] = q[j]; }
  return out;
}

/**
 * GHB-lite estimates: for each posterior quantile of tau^2, borrowing and anchoring give a prior
 * N(mu~_k, M_k + tau_q^2); each hospital fits its data under every component (times the default prior) and averages
 * the fits by their Laplace evidence. With oracleTau2, tau^2 is known (a single component, a reference).
 * @param {Array<{X:Float64Array,y:Float64Array,n:number}>} data hospitals
 * @param {{th:Float64Array[], r:Float64Array[], drop:boolean[]}} inv inverted releases
 * @param {ArrayLike<number>} mu0 anchor
 * @param {Function} avg gossip operator
 * @param {number|null} [asd=0.3] stated anchor sd
 * @param {number|null} [oracleTau2=null] known tau^2
 * @param {number|number[]} [priorSd=S0] default prior scale(s)
 * @returns {Float64Array[]}
 * @see Section 5.3, Remark 4.4; ghb_logit.estimate
 */
export function liteEstimate(data, inv_, mu0, avg, asd = 0.3, oracleTau2 = null, priorSd = S0) {
  const { th, r, drop } = inv_;
  const t2 = oracleTau2 == null ? posteriorQuantilesPer(th, r) : [new Float64Array(P).fill(oracleTau2)];
  const comps = t2.map((tq) => { const b = liteAnchor(liteNetwork(th, r, avg, tq), mu0, asd, drop); return { ...b, tq }; });
  return data.map((d, k) => combineDiag(d, comps.map((c) => c.mu[k]), comps.map((c) => Float64Array.from(c.M[k], (v, a) => v + c.tq[a])), priorSd));
}

/**
 * GHB-lite end to end: release, inversion and estimates (and the inverted releases, reused by the one-shot baselines).
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {ArrayLike<number>} n sizes
 * @param {Public} pub public moments
 * @param {ArrayLike<number>} mu0 anchor
 * @param {Function} avg gossip operator
 * @param {{eps:number, delta:number, asd:number|null, check:boolean, oracleTau2?:number|null}} o settings
 * @returns {{est:Float64Array[], inv:{th:Float64Array[], r:Float64Array[], drop:boolean[]}, oracle:Float64Array[]|null}}
 * @see Section 5.3
 */
export function ghbLite(rng, data, n, pub, mu0, avg, o) {
  const { g, sd } = liteRelease(rng, data, mu0, o.eps, o.delta);
  const iv = liteInvert(pub, g, sd, n, mu0, avg, o.asd, o.check);
  const est = liteEstimate(data, iv, mu0, avg, o.asd);
  const oracle = o.oracleTau2 != null ? liteEstimate(data, iv, mu0, avg, o.asd, o.oracleTau2) : null;
  return { est, inv: iv, oracle };
}

/* ================= GHB: two joint releases (Section 5.2) ================= */

/**
 * Stage 1: the two joint releases and the network estimate. The information sum (sensitivity G^2/2, Frobenius) and
 * R1 gradient sums (sensitivity 2G) each receive an equal share of rho1; w is R1 damped Newton steps from the origin.
 * @param {{n:()=>number}} rng generator
 * @param {Array<{X:Float64Array,y:Float64Array,n:number}>} data hospitals
 * @param {number} rho1 zCDP budget of the stage
 * @param {number} [R1=1] gradient rounds
 * @returns {{w:Float64Array, H:Float64Array, Vdp:Float64Array, sS:number, sJ:number}}
 * @see Section 5.2 (S~, J~, w = A S~, V_dp = s_S^2 A A^T); Algorithm 1; ghb_v4.stage1
 */
export function jointStage1(rng, data, rho1, R1 = 1) {
  const N = data.reduce((s, d) => s + d.n, 0), hs = 1 / (R1 + 1), rg = rho1 * (1 - hs) / R1, rh = rho1 * hs;
  const sg = isFinite(rg) ? 2 * G / Math.sqrt(2 * rg) : 0, sh = isFinite(rh) ? (G ** 2 / 2) / Math.sqrt(2 * rh) : 0;
  let H = new Float64Array(P * P);
  for (const d of data) H = add(H, infoAtZero(d));
  const E = new Float64Array(P * P);
  for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) { const z = rng.n(); if (b >= a) E[a * P + b] = z * sh; }
  for (let a = 0; a < P; a++) for (let b = a + 1; b < P; b++) E[b * P + a] = E[a * P + b];
  const H0 = psd(add(H, E), P, 1e-2 * N), A = inv(add(H0, eye(P, 1 / S0 ** 2)), P);
  let w = new Float64Array(P);
  for (let it = 0; it < R1; it++) {
    const g = new Float64Array(P);
    for (const d of data) { const s = scoreSum(d, w); for (let a = 0; a < P; a++) g[a] += s[a]; }
    for (let a = 0; a < P; a++) g[a] += sg * rng.n() - w[a] / S0 ** 2;
    const st = mv(A, g, P);
    w = Float64Array.from(w, (x, a) => Math.min(8, Math.max(-8, x + st[a])));
  }
  const Vdp = mul(A, A, P).map((v) => v * sg ** 2 * R1);
  return { w, H: H0, Vdp, sS: sg, sJ: sh };
}

/**
 * GHB estimates for one network (frozen configuration of the paper: the whole budget on the two joint releases, one
 * Newton step, no-borrowing components with a quarter of the prior weight, published anchor dropped on conflict).
 * @param {{n:()=>number}} rng generator (privacy noise)
 * @param {Array<{X:Float64Array,y:Float64Array,n:number}>} data hospitals
 * @param {number} eps epsilon (Infinity: no privacy)
 * @param {number} delta delta
 * @param {{anchor?:{mu0:ArrayLike<number>, sd:number}|null, safety?:boolean, safetyW?:number, scale?:number, loo?:boolean, vdp?:boolean}} [cfg]
 * @returns {{est:Float64Array[], w:Float64Array, anchorUsed:boolean, tq:Float64Array}}
 * @see Section 5.2, Algorithm 1, Proposition 5.1; ghb_v4.estimate with V4_CFG = (f=1, safety, R1=1)
 */
export function ghbJoint(rng, data, eps, delta, cfg = {}) {
  const c = { safety: true, safetyW: 1 / 3, scale: SCALE, loo: true, vdp: true, anchor: null, ...cfg };
  const N = data.reduce((s, d) => s + d.n, 0), rho = rhoTotal(eps, delta);
  const st = jointStage1(rng, data, rho, 1);
  const tq = priorQuantiles(c.scale);
  let { w, Vdp } = st; let anchorUsed = false;
  if (c.anchor) {
    const Vs = diagOf(inv(add(st.H, eye(P, 1 / S0 ** 2)), P), P), Vt = Float64Array.from(Vs, (v, a) => v + Vdp[a * P + a]), { mu0, sd } = c.anchor;
    let z = 0; for (let a = 0; a < P; a++) z = Math.max(z, Math.abs(mu0[a] - w[a]) / Math.sqrt(sd ** 2 + Vt[a]));
    if (z <= Z_CONFLICT) {
      anchorUsed = true;
      const nw = new Float64Array(P), nv = new Float64Array(P);
      for (let a = 0; a < P; a++) { const pr = 1 / Vt[a] + 1 / sd ** 2; nw[a] = (w[a] / Vt[a] + mu0[a] / sd ** 2) / pr; nv[a] = Math.max(1 / pr - Vs[a], 0); }
      w = nw; Vdp = diag(nv);
    }
  }
  const est = data.map((d) => {
    const sj = scoreSum(d, w), Hm = add(psd(add(st.H, infoAtZero(d), -1), P, 1e-2 * Math.max(N - d.n, 1)), eye(P, 1 / S0 ** 2));
    let V = inv(Hm, P);
    const step = mv(V, sj, P), mu = c.loo ? Float64Array.from(w, (x, a) => x - step[a]) : w;
    if (c.vdp) V = add(V, Vdp);
    const comps = [], lq = Math.log(1 / (Q * SDS.length));
    for (let q = 0; q < Q; q++) {
      const Cn = add(diag(new Float64Array(P).fill(tq[q])), V), Cni = inv(Cn, P), Cnim = mv(Cni, mu, P);
      for (const s of SDS) { const C = inv(add(Cni, eye(P, 1 / s ** 2)), P); comps.push({ logw: lq, m: mv(C, Cnim, P), C }); }
    }
    if (c.safety) for (const s of SDS) comps.push({ logw: Math.log(c.safetyW / SDS.length), m: new Float64Array(P), C: eye(P, s ** 2) });
    return combineFull(d, comps);
  });
  return { est, w, anchorUsed, tq };
}

export { Public };
