// Prototype (development only): GHB with network-pooled evidence about tau ("v6") vs released GHB, Gaussian model.
// node v6_proto.mjs (from simulations/dev)
import { jointGHB, priorQuantiles, GRID, DGRID } from '../../app/js/math/ghb.js';
import { jointNoiseSd, zcdpRho, zcdpSigma } from '../../app/js/math/dp.js';
import { siteSizes } from '../../app/js/math/random.js';

function rng(seed) { let s = seed >>> 0; const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return { u, n: () => Math.sqrt(-2 * Math.log(u() + 1e-300)) * Math.cos(2 * Math.PI * u()) }; }
const npdf = (x, v) => -0.5 * Math.log(2 * Math.PI * v) - 0.5 * x * x / v;
const lse = (a) => { const m = Math.max(...a); return m + Math.log(a.reduce((s, x) => s + Math.exp(x - m), 0)); };

// v6: each hospital k computes, for each prior quantile tau_q^2, the clipped log-evidence of its own estimate against its
// leave-one-out network mean; the network adds them with a secure sum plus Gaussian noise (half of the privacy budget);
// each hospital combines the network evidence (its own clipped term replaced by its exact term) with the safety components.
function ghbV6(th, n, v, sigma2, eps, delta, C, rngD, o = {}) {
  const K = th.length, tq = o.grid ? Array.from(GRID, (t) => t * t) : Array.from(priorQuantiles(0.5, 12)), Q = o.grid ? 12 : tq.length,
    lp = o.grid ? Array.from(GRID, (t, i) => Math.log(DGRID[i] / (1 + (t / 0.5) ** 2))) : tq.map(() => 0), sds = [0.5, 1, 2.5], safetyW = 1 / 3;
  const rho = isFinite(eps) ? zcdpRho(eps, delta) : Infinity;
  const s = isFinite(rho) ? zcdpSigma(2 * C, rho / 2) : 0;            // the sum gets half the budget
  let N = 0, S = s * rngD.n(); for (let k = 0; k < K; k++) { N += n[k]; S += n[k] * th[k]; }
  let sn2 = 0; for (const x of n) sn2 += x * x;
  const mu = [], M = [], lc = [], le = [];
  for (let k = 0; k < K; k++) {
    const Nk = N - n[k]; mu[k] = (S - n[k] * th[k]) / Nk;
    const W2 = (sn2 - n[k] * n[k]) / (Nk * Nk), M0 = (Nk * sigma2 + s * s) / (Nk * Nk);
    M[k] = tq.map((t2) => M0 + t2 * W2);
    const r = th[k] - mu[k];
    lc[k] = tq.map((t2, q) => { const V = t2 + M[k][q] + v[k], z2 = Math.min(r * r / V, 9); return -0.5 * Math.log(2 * Math.PI * V) - 0.5 * z2; });
    le[k] = tq.map((t2, q) => npdf(r, t2 + M[k][q] + v[k]));
  }
  // per-patient sensitivity of each coordinate of the evidence sum (data clipped to [-C, C], sigma assumed known)
  let sens = 0; for (let k = 0; k < K; k++) { let a = 1 / Math.sqrt(n[k]); for (let j = 0; j < K; j++) if (j !== k) a += Math.sqrt(n[j]) / (N - n[j]); sens = Math.max(sens, 6 * C / Math.sqrt(sigma2) * a); }
  const sE = isFinite(rho) ? zcdpSigma(sens * Math.sqrt(Q), rho / 2) : 0;
  const E = tq.map((_, q) => lc.reduce((a, l) => a + l[q], 0) + sE * rngD.n());
  const est = new Float64Array(K), alpha = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    const lw = [], al = [];
    const net = tq.map((_, q) => lp[q] + E[q] - lc[k][q]), Z = lse(net);
    tq.forEach((t2, q) => { const V = t2 + M[k][q]; lw.push(net[q] - Z + le[k][q]); al.push(V / (V + v[k])); });
    // the borrowing mixture as a whole gets weight 1, the safety components 1/3 (as in released GHB)
    for (const sd of sds) { lw.push(Math.log(safetyW / sds.length) + npdf(th[k], sd * sd + v[k])); al.push(1); }
    const mx = Math.max(...lw); let sw = 0, a = 0; lw.forEach((l, i) => { const w = Math.exp(l - mx); sw += w; a += w * al[i]; });
    alpha[k] = a / sw; est[k] = alpha[k] * th[k] + (1 - alpha[k]) * mu[k];
  }
  return { est, alpha };
}

function run(sizes, tau2, { eps = 1, R = 2000, seed = 7, atyp = null, grid = false } = {}) {
  const K = sizes.length, sigma2 = 1, C = 1, delta = 1e-5, r = rng(seed), v = sizes.map((x) => sigma2 / x);
  const sJ = jointNoiseSd(C, eps, delta, 1), m = { local: Array(K).fill(0), v5: Array(K).fill(0), v6: Array(K).fill(0), known: Array(K).fill(0) };
  for (let it = 0; it < R; it++) {
    const theta = sizes.map(() => Math.sqrt(tau2) * r.n()); if (atyp != null) theta[atyp] += 1.0;
    const th = theta.map((t, k) => t + Math.sqrt(v[k]) * r.n()), xi = sJ * r.n();
    const e5 = jointGHB(th, th, sizes, v, sigma2, sJ, xi).est, ek = jointGHB(th, th, sizes, v, sigma2, sJ, xi, { knownTau2: tau2 }).est;
    const e6 = ghbV6(th, sizes, v, sigma2, eps, delta, C, r, { grid }).est;
    for (let k = 0; k < K; k++) { m.local[k] += (th[k] - theta[k]) ** 2; m.v5[k] += (e5[k] - theta[k]) ** 2; m.v6[k] += (e6[k] - theta[k]) ** 2; m.known[k] += (ek[k] - theta[k]) ** 2; }
  }
  const rel = (id) => m[id].map((x, k) => x / R / v[k]);
  return { v5: rel('v5'), v6: rel('v6'), known: rel('known') };
}

const fmt = (res, sizes, i = 0) => ['v5', 'v6', 'known'].map((id) => { const x = res[id]; const avg = x.reduce((a, b) => a + b, 0) / x.length;
  return `${id}: like ${String(Math.round(sizes[i] / x[i])).padStart(5)}  avg ${avg.toFixed(2)}  worst ${Math.max(...x).toFixed(2)}  worse ${x.filter((y) => y > 1.02).length}`; }).join(' | ');
const A = [30, ...Array(9).fill(1000)], B = siteSizes(12, 100);
for (const eps of [1, 10]) {
  console.log(`\n== eps ${eps}`);
  for (const t of [0.0001, 0.001, 0.005, 0.02, 0.2]) console.log(`30+9x1000 tau2 ${t}`.padEnd(26), fmt(run(A, t, { eps }), A));
  for (const t of [0.001, 0.005, 0.02, 0.2]) console.log(`paper sizes K12 tau2 ${t}`.padEnd(26), fmt(run(B, t, { eps }), B));
  for (const t of [0.001, 0.02]) console.log(`atypical small, tau2 ${t}`.padEnd(26), fmt(run(B, t, { eps, atyp: 0 }), B));
}
if (process.argv[2] === 'nodp') for (const t of [0.0001, 0.001, 0.02]) console.log(`noDP 30+9x1000 tau2 ${t}`.padEnd(26), fmt(run(A, t, { eps: Infinity, R: 1000 }), A));
if (process.argv[2] === 'grid') for (const eps of [1, 10, Infinity]) { for (const t of [0.0001, 0.001, 0.005, 0.02, 0.2]) console.log(`grid eps ${eps} 30+9x1000 tau2 ${t}`.padEnd(34), fmt(run(A, t, { eps, R: 1000, grid: true }), A));
  for (const t of [0.001, 0.02]) console.log(`grid eps ${eps} atypical tau2 ${t}`.padEnd(34), fmt(run(B, t, { eps, R: 1000, grid: true, atyp: 0 }), B)); }
