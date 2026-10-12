// tests/gauss_ghb.test.mjs
// GHB with a joint release in the Gaussian model (Section 5.2): the closed form of M_k (Eq. 4 with the noise
// calibrated to the network sum) against simulation, and the regimes in which GHB and GHB-lite each win, on fixed
// seeds. Run from app/: node --test tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { jointLooM, jointGHB, drawReplicate, avgOp, borrow, shrink, looMSEiv } from '../js/math/ghb.js';
import { jointNoiseSd, zcdpRho, dpVar, collusionRho, zcdpToEps } from '../js/math/dp.js';
import { makeRng, PAPER_SIZES, siteSizes } from '../js/math/random.js';

test('GHB joint release: noise sd is 2C / sqrt(2 rho), independent of hospital sizes (Proposition 5.1)', () => {
  const rho = zcdpRho(1, 1e-5);
  assert.ok(Math.abs(jointNoiseSd(1, 1, 1e-5) - 2 / Math.sqrt(2 * rho)) < 1e-12);
  assert.equal(jointNoiseSd(1, Infinity, 1e-5), 0);
  assert.ok(Math.abs(jointNoiseSd(1, 1, 1e-5, 4) - 2 / Math.sqrt(2 * rho / 4)) < 1e-12, 'refits compose in zCDP');
  // collusion: c of K colluders leave rho K / (K - c)
  assert.ok(Math.abs(collusionRho(rho, 24, 6) - rho * 24 / 18) < 1e-15);
  assert.ok(zcdpToEps(collusionRho(rho, 24, 6), 1e-5) > 1);
});

test('closed form of M_k for the leave-one-out mean of the noisy sum matches simulation (Eq. 4, joint)', () => {
  const n = PAPER_SIZES, K = n.length, tau2 = 0.02, sigma2 = 1, s = jointNoiseSd(1, 0.3, 1e-5), rng = makeRng(404), R = 40000;
  const sq = new Float64Array(K);
  for (let r = 0; r < R; r++) {
    const theta = n.map(() => tau2 ** 0.5 * rng.n()), th = theta.map((t, k) => t + Math.sqrt(sigma2 / n[k]) * rng.n()), xi = s * rng.n();
    let S = xi, N = 0; for (let k = 0; k < K; k++) { S += n[k] * th[k]; N += n[k]; }
    for (let k = 0; k < K; k++) sq[k] += ((S - n[k] * th[k]) / (N - n[k])) ** 2 / R; // mu = 0
  }
  for (const k of [0, 11, 23]) {
    const M = jointLooM(n, k, tau2, sigma2, s * s), se = M * Math.sqrt(2 / R);
    assert.ok(Math.abs(sq[k] - M) < 4 * se, `k = ${k}: simulated ${sq[k]} vs closed form ${M}`);
  }
  // explicit formula: sum_{j != k} n_j^2 (tau^2 + sigma^2 / n_j) / (N - n_k)^2 + s^2 / (N - n_k)^2
  const N = n.reduce((a, b) => a + b, 0); let a = 0; for (let j = 1; j < K; j++) a += n[j] ** 2 * (tau2 + sigma2 / n[j]);
  assert.ok(Math.abs(jointLooM(n, 0, tau2, sigma2, s * s) - (a + s * s) / (N - n[0]) ** 2) < 1e-15);
  // the privacy term is of order 1 / N^2, far below GHB-lite's for the smallest hospitals
  const totals = n.map((x) => tau2 + sigma2 / x + dpVar(x, 0.3, 1e-5, 1));
  assert.ok(jointLooM(n, 0, tau2, sigma2, s * s) < looMSEiv(totals, 0), 'joint M below per-hospital M at eps = 0.3');
});

function compare(n, tau2, eps, R, seed) {
  const K = n.length, q = { tau2, eps, delta: 1e-5, C: 1 }, rng = makeRng(seed), avg = avgOp(null, K), s = jointNoiseSd(1, eps, 1e-5);
  let lite = 0, joint = 0;
  for (let r = 0; r < R; r++) {
    const d = drawReplicate(rng, n, q), xi = s * rng.n();
    const l = shrink(d.th, d.v, borrow(d.x, d.r, avg)), j = jointGHB(d.th, d.th, n, d.v, 1, s, xi).est;
    for (let k = 0; k < K; k++) { lite += (l[k] - d.theta[k]) ** 2 * n[k] / (R * K); joint += (j[k] - d.theta[k]) ** 2 * n[k] / (R * K); }
  }
  return { lite, joint };
}

test('GHB beats GHB-lite at small epsilon with small hospitals (fixed seed)', () => {
  const r = compare(siteSizes(24, 30), 0.02, 0.1, 1500, 2);
  assert.ok(r.joint < 0.9 * r.lite, `GHB ${r.joint.toFixed(3)} vs GHB-lite ${r.lite.toFixed(3)}`);
  assert.ok(r.joint < 0.8, 'GHB still borrows');
});

test('GHB-lite beats GHB when releases are informative: many large hospitals, large epsilon (fixed seed)', () => {
  const r = compare(siteSizes(48, 400), 0.02, 10, 1500, 3);
  assert.ok(r.lite < 0.97 * r.joint, `GHB-lite ${r.lite.toFixed(3)} vs GHB ${r.joint.toFixed(3)}`);
  const r2 = compare(siteSizes(48, 100), 0.02, 3, 1500, 3);
  assert.ok(r2.lite < 0.9 * r2.joint, `48 hospitals of median 100 at eps 3: GHB-lite ${r2.lite.toFixed(3)} vs GHB ${r2.joint.toFixed(3)}`);
});

test('GHB with known tau^2 is Proposition 4.1 with the joint M; an atypical hospital borrows less (Remark 4.4)', () => {
  const n = [50, 50, 50, 50, 50, 50], v = n.map((x) => 1 / x), th = [0, 0.01, -0.01, 0.02, -0.02, 3];
  const o = jointGHB(th, th, n, v, 1, 0, 0, { knownTau2: 0.02 });
  const M = jointLooM(n, 0, 0.02, 1, 0), a = (0.02 + M) / (0.02 + M + v[0]), mu = (th.slice(1).reduce((s, x) => s + 50 * x, 0)) / 250;
  assert.ok(Math.abs(o.est[0] - (a * th[0] + (1 - a) * mu)) < 1e-12);
  const m = jointGHB(th, th, n, v, 1, 0, 0);
  assert.ok(m.alpha[5] > m.alpha[0], 'the atypical hospital keeps more of its own estimate');
  assert.ok(Math.abs(m.est[5] - th[5]) < 0.1 * Math.abs(th[5] - m.mu[5]), 'and moves less than a tenth of the way to the network');
});
