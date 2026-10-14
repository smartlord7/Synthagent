/**
 * math/literature.js
 * Responsibility: the literature methods the paper compares with GHB, for logistic heads, each spending the same
 * (epsilon, delta) per hospital under zero-concentrated accounting (Section 6.3, Simulations 8 and 15; Section 7.2).
 * One-shot baselines on GHB-lite's releases (FedAvg of releases, DerSimonian-Laird meta-analysis, Ditto-style),
 * iterative baselines (DP-FedAvg, fine-tuning, DP-FedAvg with secure aggregation, decentralized DP-SGD, clustered
 * DP-FedAvg with a private cluster assignment, local gradient descent from the anchor) and personalized federated
 * learning (Ditto, pFedMe, APFL, Per-FedAvg, FedRep, FedProx). Ports of simulations/literature.py,
 * simulations/personalized.py, simulations/ghb_sa.py and simulations/logit_world.py. Per-hospital estimates are
 * arrays of Float64Array(P).
 */
import { P, R_X, LAM, grad, fit, sigmoid } from './logit.js';
import { dersimonianLaird, avgOp } from './ghb.js';

/** Per-patient gradient norm bound. @see literature.G */
export const G = Math.sqrt(5);
/** Smoothness of the per-patient logistic loss, max ||x||^2 / 4. @see personalized.L_SMOOTH */
export const L_SMOOTH = 1.25;
/** Fine-tuning step size. @see literature.FT_ETA */
export const FT_ETA = 1.0;

const zerosK = (K) => Array.from({ length: K }, () => new Float64Array(P));
const repeat = (v, K) => Array.from({ length: K }, () => Float64Array.from(v));
const copyK = (A) => A.map((v) => Float64Array.from(v));

/** Noise sd of one round's mean gradient: factor 2G / (n sqrt(2 rho / R)). @see personalized._sd */
function sdRound(n, rho, R, factor = 1) { return Float64Array.from(n, (x) => (isFinite(rho) ? factor * 2 * G / x / Math.sqrt(2 * rho / R) : 0)); }

/** n-weighted average of privatized vectors. @see personalized._avg */
function noisyAvg(rng, vecs, n, sd) {
  const N = n.reduce((a, b) => a + b, 0), out = new Float64Array(P);
  vecs.forEach((v, k) => { for (let a = 0; a < P; a++) out[a] += n[k] / N * (v[a] + sd[k] * rng.n()); });
  return out;
}

/**
 * Snapshots of E plain local gradient steps (no privacy cost) from each hospital's start.
 * @param {Array<{X:Float64Array,y:Float64Array,n:number}>} data hospitals
 * @param {Float64Array[]} start per-hospital start
 * @param {number[]} steps snapshot step counts
 * @returns {Map<number, Float64Array[]>}
 * @see Section 6.3 (fine-tuning); literature.fine_tune
 */
export function fineTune(data, start, steps) {
  const out = new Map(), th = copyK(start); let done = 0;
  for (const E of [...steps].sort((a, b) => a - b)) {
    for (let i = done; i < E; i++) data.forEach((d, k) => { const g = grad(d, th[k]); for (let a = 0; a < P; a++) th[k][a] -= FT_ETA * g[a]; });
    done = E; out.set(E, copyK(th));
  }
  return out;
}

/**
 * Full-batch DP-FedAvg (McMahan et al., 2018): each round every member sends its noisy mean gradient, the server
 * takes an n-weighted step.
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {ArrayLike<number>} init start
 * @param {number} rho total zCDP budget
 * @param {number} R rounds
 * @param {number} eta step size
 * @param {number[]|null} [members=null] subset of hospitals (clusters)
 * @returns {Float64Array} global model
 * @see Section 6.3, Simulation 15; literature.fedavg
 */
export function fedavg(rng, data, n, init, rho, R, eta, members = null) {
  const idx = members || data.map((_, k) => k), nn = idx.map((k) => n[k]), sd = sdRound(nn, rho, R), th = Float64Array.from(init);
  for (let r = 0; r < R; r++) {
    const g = noisyAvg(rng, idx.map((k) => grad(data[k], th)), nn, sd);
    for (let a = 0; a < P; a++) th[a] -= eta * g[a];
  }
  return th;
}

/**
 * DP-FedAvg with secure aggregation (Bonawitz et al., 2017): Gaussian noise on the summed gradient, sensitivity 2G.
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {ArrayLike<number>} init start
 * @param {number} rho zCDP budget
 * @param {number} R rounds
 * @param {number} eta step
 * @returns {Float64Array}
 * @see Section 7.2; ghb_sa.fedavg_sa
 */
export function fedavgSA(rng, data, n, init, rho, R, eta) {
  const N = n.reduce((a, b) => a + b, 0), sd = isFinite(rho) ? 2 * G / Math.sqrt(2 * rho / R) : 0, th = Float64Array.from(init);
  for (let r = 0; r < R; r++) {
    const g = new Float64Array(P);
    data.forEach((d) => { const gk = grad(d, th); for (let a = 0; a < P; a++) g[a] += gk[a] * d.n; });
    for (let a = 0; a < P; a++) th[a] -= eta * (g[a] + sd * rng.n()) / N;
  }
  return th;
}

/**
 * Decentralized DP-SGD (Lian et al., 2017): each hospital steps on its own noisy gradient, then averages models
 * with its neighbours (Metropolis-Hastings weights), adapt-then-combine.
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {ArrayLike<number>} init start
 * @param {number} rho zCDP budget
 * @param {number} R rounds
 * @param {number} eta step
 * @param {Float64Array[]} W gossip matrix
 * @returns {Float64Array[]}
 * @see Section 2.4, Simulation 15; literature.decentralized
 */
export function decentralized(rng, data, n, init, rho, R, eta, W) {
  const K = data.length, sd = sdRound(n, rho, R); let th = repeat(init, K);
  for (let r = 0; r < R; r++) {
    const loc = th.map((t, k) => { const g = grad(data[k], t); return Float64Array.from(t, (x, a) => x - eta * (g[a] + sd[k] * rng.n())); });
    th = loc.map((_, k) => { const o = new Float64Array(P); for (let j = 0; j < K; j++) { const w = W[k][j]; if (w) for (let a = 0; a < P; a++) o[a] += w * loc[j][a]; } return o; });
  }
  return th;
}

/**
 * Private cluster assignment (Ghosh et al., 2020, made private): one ridge release per hospital with budget rho, then
 * weighted k-means++ and k-means on the releases (post-processing).
 * @param {{n:()=>number, u:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {number} rho budget of the release
 * @param {number} C clusters
 * @param {number} [iters=25] k-means iterations
 * @returns {number[]} labels
 * @see Simulation 15; literature.private_clusters
 */
export function privateClusters(rng, data, n, rho, C, iters = 25) {
  const K = data.length;
  const x = data.map((d, k) => {
    const raw = fit(d, new Float64Array(P), new Float64Array(P).fill(d.n * LAM)), s2 = isFinite(rho) ? (2 * R_X / (LAM * n[k])) ** 2 / (2 * rho) : 0;
    return Float64Array.from(raw, (v) => v + Math.sqrt(s2) * rng.n());
  });
  const w = n.map((nk) => { const s2 = isFinite(rho) ? (2 * R_X / (LAM * nk)) ** 2 / (2 * rho) : 0; return 1 / (s2 + 1 / nk); });
  const d2 = (a, b) => { let s = 0; for (let i = 0; i < P; i++) s += (a[i] - b[i]) ** 2; return s; };
  let best = 0; for (let k = 1; k < K; k++) if (w[k] > w[best]) best = k;
  const cent = [Float64Array.from(x[best])];
  for (let c = 1; c < C; c++) {
    const dd = x.map((xk, k) => Math.min(...cent.map((ce) => d2(xk, ce))) * w[k]), tot = dd.reduce((a, b) => a + b, 0);
    let pick = K - 1;
    if (tot > 0) { let u = rng.u() * tot; for (let k = 0; k < K; k++) { u -= dd[k]; if (u < 0) { pick = k; break; } } } else pick = Math.floor(rng.u() * K);
    cent.push(Float64Array.from(x[pick]));
  }
  let lab = new Array(K).fill(0);
  for (let it = 0; it < iters; it++) {
    lab = x.map((xk) => { let bc = 0, bd = Infinity; cent.forEach((ce, c) => { const v = d2(xk, ce); if (v < bd) { bd = v; bc = c; } }); return bc; });
    for (let c = 0; c < C; c++) {
      let sw = 0; const m = new Float64Array(P);
      lab.forEach((l, k) => { if (l === c) { sw += w[k]; for (let a = 0; a < P; a++) m[a] += w[k] * x[k][a]; } });
      if (sw > 0) cent[c] = m.map((v) => v / sw);
    }
  }
  return lab;
}

/**
 * Clustered DP-FedAvg (IFCA-style): DP-FedAvg within each privately assigned cluster with half the budget.
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {ArrayLike<number>} init start
 * @param {number} rhoHalf budget of the training stage (rho / 2)
 * @param {number[]} lab cluster labels
 * @param {number} C clusters
 * @param {number} R rounds
 * @param {number} eta step
 * @returns {Float64Array[]}
 * @see Simulation 15; literature.estimates (clustered branch)
 */
export function clusteredFedavg(rng, data, n, init, rhoHalf, lab, C, R, eta) {
  const th0 = zerosK(data.length);
  for (let c = 0; c < C; c++) {
    const mem = lab.map((l, k) => (l === c ? k : -1)).filter((k) => k >= 0);
    if (mem.length) { const g = fedavg(rng, data, n, init, rhoHalf, R, eta, mem); mem.forEach((k) => { th0[k] = Float64Array.from(g); }); }
  }
  return th0;
}

/* ---------- one-shot baselines on GHB-lite's releases (Simulation 8) ---------- */

/**
 * One-shot federated averaging (McMahan et al., 2017) of the inverted releases, sample-size weighted.
 * @param {Float64Array[]} rel releases on the parameter scale
 * @param {number[]} n sizes
 * @returns {Float64Array}
 * @see Simulation 8; logit_world.one_rep (fedavg)
 */
export function oneShotFedavg(rel, n) { const N = n.reduce((a, b) => a + b, 0), o = new Float64Array(P); rel.forEach((v, k) => { for (let a = 0; a < P; a++) o[a] += n[k] / N * v[a]; }); return o; }

/**
 * Random-effects (DerSimonian and Laird, 1986) pooled mean of the releases, coefficient by coefficient.
 * @param {Float64Array[]} rel releases
 * @param {Float64Array[]} r their variances
 * @returns {Float64Array}
 * @see Simulation 8; logit_world.one_rep (meta)
 */
export function metaAnalysis(rel, r) {
  const K = rel.length, o = new Float64Array(P), avg = avgOp(null, K);
  for (let d = 0; d < P; d++) {
    const x = Float64Array.from(rel, (v) => v[d]), rr = Float64Array.from(r, (v) => v[d]), t2 = dersimonianLaird(x, rr, avg)[0];
    let sw = 0, swx = 0; for (let k = 0; k < K; k++) { const w = 1 / (t2 + rr[k]); sw += w; swx += w * x[k]; }
    o[d] = swx / sw;
  }
  return o;
}

/**
 * Ditto-style one-shot personalization (after Li et al., 2021): each local fit shrunk toward the one-shot federated
 * average with a proximal weight lambda.
 * @param {Array} data hospitals
 * @param {ArrayLike<number>} centre federated average of the releases
 * @param {number} lam proximal weight
 * @returns {Float64Array[]}
 * @see Simulation 8; logit_world.one_rep (Ditto-style)
 */
export function dittoStyle(data, centre, lam) { const pr = new Float64Array(P).fill(lam); return data.map((d) => fit(d, centre, pr)); }

/* ---------- personalized federated learning (Section 7.2) ---------- */

/**
 * Ditto (Li et al., 2021): global model by DP-FedAvg from the origin; personal models v_k = argmin f_k(v) + lam/2 ||v - w||^2,
 * computed exactly at the end. (The per-round personal updates of personalized.ditto do not affect its output or its
 * noise draws, so they are not recomputed.)
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {number} rho budget
 * @param {number} R rounds
 * @param {number} eta step
 * @param {number} lam proximal weight
 * @returns {Float64Array[]}
 * @see Section 7.2; personalized.ditto
 */
export function ditto(rng, data, n, rho, R, eta, lam) {
  const sd = sdRound(n, rho, R), w = new Float64Array(P);
  for (let r = 0; r < R; r++) { const g = noisyAvg(rng, data.map((d) => grad(d, w)), n, sd); for (let a = 0; a < P; a++) w[a] -= eta * g[a]; }
  return dittoStyle(data, w, lam);
}

/**
 * pFedMe (T. Dinh et al., 2020): Moreau-envelope personalization; shared change with sensitivity S beta 2G / lam
 * (the proximal fit minimizes the sum of the losses; corrected 2026-10-14 from S beta 2G / (n lam), n times too small).
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {number} rho budget
 * @param {number} R rounds
 * @param {number} eta step
 * @param {number} lam proximal weight
 * @param {number} [S=5] local steps
 * @returns {Float64Array[]}
 * @see Section 7.2; personalized.pfedme
 */
export function pfedme(rng, data, n, rho, R, eta, lam, S = 5) {
  const beta = Math.min(eta * lam, 1), sd = sdRound(n, rho, R, S * beta / lam).map((v, i) => v * n[i]), w = new Float64Array(P), pr = new Float64Array(P).fill(lam);
  for (let r = 0; r < R; r++) {
    const deltas = data.map((d) => { const wk = Float64Array.from(w); for (let s = 0; s < S; s++) { const th = fit(d, wk, pr); for (let a = 0; a < P; a++) wk[a] -= beta * (wk[a] - th[a]); } return wk.map((v, a) => v - w[a]); });
    const m = noisyAvg(rng, deltas, n, sd); for (let a = 0; a < P; a++) w[a] += m[a];
  }
  return dittoStyle(data, w, lam);
}

/**
 * APFL (Deng et al., 2020): personalized model alpha v_k + (1 - alpha) w, w by DP-FedAvg, v_k local.
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {number} rho budget
 * @param {number} R rounds
 * @param {number} eta step
 * @param {number} alpha mixing weight
 * @param {number} [steps=5] local steps
 * @param {number} [etaP=0.5] local step size
 * @returns {Float64Array[]}
 * @see Section 7.2; personalized.apfl
 */
export function apfl(rng, data, n, rho, R, eta, alpha, steps = 5, etaP = 0.5) {
  const sd = sdRound(n, rho, R), w = new Float64Array(P), v = zerosK(data.length);
  for (let r = 0; r < R; r++) {
    const g = noisyAvg(rng, data.map((d) => grad(d, w)), n, sd);
    data.forEach((d, k) => { for (let s = 0; s < steps; s++) { const mix = Float64Array.from(v[k], (x, a) => alpha * x + (1 - alpha) * w[a]), gk = grad(d, mix); for (let a = 0; a < P; a++) v[k][a] -= etaP * alpha * gk[a]; } });
    for (let a = 0; a < P; a++) w[a] -= eta * g[a];
  }
  return v.map((vk) => Float64Array.from(vk, (x, a) => alpha * x + (1 - alpha) * w[a]));
}

/**
 * Per-FedAvg (Fallah et al., 2020, first order): gradient at the adapted point w - a grad f_k(w), sensitivity (1 + a L) 2G / n;
 * each hospital's model is one local step from the final w.
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {number} rho budget
 * @param {number} R rounds
 * @param {number} eta step
 * @param {number} a adaptation step
 * @returns {Float64Array[]}
 * @see Section 7.2; personalized.per_fedavg
 */
export function perFedavg(rng, data, n, rho, R, eta, a) {
  const sd = sdRound(n, rho, R, 1 + a * L_SMOOTH), w = new Float64Array(P);
  const adapt = (d, x) => { const g = grad(d, x); return Float64Array.from(x, (v, i) => v - a * g[i]); };
  for (let r = 0; r < R; r++) { const g = noisyAvg(rng, data.map((d) => grad(d, adapt(d, w))), n, sd); for (let i = 0; i < P; i++) w[i] -= eta * g[i]; }
  return data.map((d) => adapt(d, w));
}

/**
 * FedRep / FedPer (Collins et al., 2021): coefficients shared and trained by DP-FedAvg, the intercept local and exact;
 * fine-tuning snapshots at the end.
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {number} rho budget
 * @param {number} R rounds
 * @param {number} eta step
 * @param {number[]} [ft=[0, 5]] fine-tuning steps
 * @returns {Map<number, Float64Array[]>}
 * @see Section 7.2; personalized.fedrep
 */
export function fedrep(rng, data, n, rho, R, eta, ft = [0, 5]) {
  const sd = sdRound(n, rho, R), w = new Float64Array(P), b = new Float64Array(data.length);
  for (let r = 0; r < R; r++) {
    data.forEach((d, k) => {
      for (let it = 0; it < 10; it++) {
        let gm = 0, hm = 0;
        for (let i = 0; i < d.n; i++) { const o = i * P; let s = b[k]; for (let a = 1; a < P; a++) s += d.X[o + a] * w[a]; const p = sigmoid(s); gm += p - d.y[i]; hm += p * (1 - p); }
        b[k] -= (gm / d.n) / Math.max(hm / d.n, 1e-6);
      }
    });
    const gs = data.map((d, k) => { const th = Float64Array.from(w); th[0] = b[k]; const g = grad(d, th); g[0] = 0; return g; });
    const m = noisyAvg(rng, gs, n, sd); for (let a = 0; a < P; a++) w[a] -= eta * m[a];
  }
  const start = data.map((_, k) => { const s = Float64Array.from(w); s[0] = b[k]; return s; });
  return fineTune(data, start, ft);
}

/**
 * FedProx (Li et al., 2020): E local proximal gradient steps per round, the change shared with sensitivity eta E 2G / n;
 * fine-tuning snapshots at the end.
 * @param {{n:()=>number}} rng generator
 * @param {Array} data hospitals
 * @param {number[]} n sizes
 * @param {number} rho budget
 * @param {number} R rounds
 * @param {number} eta step
 * @param {number} mu proximal weight
 * @param {number} [E=5] local steps
 * @param {number[]} [ft=[0, 5, 30]] fine-tuning steps
 * @returns {Map<number, Float64Array[]>}
 * @see Section 7.2; personalized.fedprox
 */
export function fedprox(rng, data, n, rho, R, eta, mu, E = 5, ft = [0, 5, 30]) {
  const sd = sdRound(n, rho, R, eta * E), w = new Float64Array(P);
  for (let r = 0; r < R; r++) {
    const deltas = data.map((d) => { const th = Float64Array.from(w); for (let e = 0; e < E; e++) { const g = grad(d, th); for (let a = 0; a < P; a++) th[a] -= eta * (g[a] + mu * (th[a] - w[a])); } return th.map((v, a) => v - w[a]); });
    const m = noisyAvg(rng, deltas, n, sd); for (let a = 0; a < P; a++) w[a] += m[a];
  }
  return fineTune(data, repeat(w, data.length), ft);
}

export { repeat };
