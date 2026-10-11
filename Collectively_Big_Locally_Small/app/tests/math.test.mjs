// Reference values from the paper (Section 2, Figure 9, Section 8.7) and from the Python implementation
// (scripts/blueprint.py, simulations/ghb.py). Run from app/: node --test tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { neffParts, rileyN, regime, effectiveEvents } from '../js/math/smallness.js';
import { dpVar, zcdpRho, zcdpToEps, noiseBasic, noiseZcdp } from '../js/math/dp.js';
import { topology, metropolisHastings, slem, matPow, connected } from '../js/math/graph.js';
import { riskHB, looMSE, borrowedPatients, drawReplicate, avgOp, borrow, shrink, oracleHB, robustPosteriorMean, weightedMedian, gmrfJacobi } from '../js/math/ghb.js';
import { makeRng, PAPER_SIZES } from '../js/math/random.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

test('ICU example: n_eff ~ 414 and 33 effective events (Example 2.4, Eq. 2)', () => {
  const { neff, deff } = neffParts(2000, 1.6, 0.5, 0.7, 0.1, 0.6);
  close(deff, 1.3, 1e-12, 'DEFF');
  assert.equal(Math.round(neff), 414);
  assert.equal(Math.round(effectiveEvents(neff, 0.08)), 33);
});

test('Riley n* = 2707 for p = 20, R2_CS 15% of max at prevalence 8%, S = 0.9 (Eq. 1)', () => {
  const n = rileyN(20, 0.15, 0.08, 0.9);
  assert.equal(Math.round(n), 2707);
  const sigma = n / neffParts(2000, 1.6, 0.5, 0.7, 0.1, 0.6).neff;
  close(sigma, 6.5, 0.05, 'smallness index');
  assert.equal(regime(sigma).name, 'strained');
});

test('SLEM 0.977 (ring) and 0.933 (ring + 2-hop) for K = 24 (Figure 9)', () => {
  close(slem(metropolisHastings(topology('ring', 24))), 0.977, 5e-4, 'ring');
  close(slem(metropolisHastings(topology('ring2', 24))), 0.933, 5e-4, 'ring + 2-hop');
  close(slem(metropolisHastings(topology('regional', 24))), 0.964, 5e-4, 'regional (paper seed)');
  close(slem(metropolisHastings(topology('random', 24, 0.15))), 0.937, 5e-4, 'sparse random (paper seed)');
  assert.ok(connected(topology('random', 24, 0.15)));
});

test('risk_hb and loo_mse match scripts/blueprint.py', () => {
  // python: M = loo_mse([200]*11, 0.02, eps=1.0); risk_hb([16,50,200,1000], 0.02, M)
  const M = looMSE(Array(12).fill(200), 11, { tau2: 0.02, eps: 1, delta: 1e-5, C: 1 });
  close(M, 0.0024861103457506266, 1e-15, 'M (Eq. 6)');
  const ref = [0.01653660687483957, 0.010585158378942843, 0.00409044969675511, 0.0009574216426101];
  [16, 50, 200, 1000].forEach((n, i) => close(riskHB(n, 0.02, M), ref[i], 1e-15, `risk_hb n=${n}`));
  close(borrowedPatients(0.02, M), 44.47189774593353, 1e-9, 'B (Eq. 5)');
  close(dpVar(100, 1, 1e-5, 1), (0.02 * Math.sqrt(2 * Math.log(1.25e5))) ** 2, 1e-15, 'Gaussian mechanism');
});

test('zCDP accounting round-trips and beats basic composition for many refits (Simulation 14)', () => {
  const rho = zcdpRho(1, 1e-5);
  close(zcdpToEps(rho, 1e-5), 1, 1e-12, 'eps(rho)');
  assert.ok(noiseZcdp(0.04, 1, 1e-5, 10) < noiseBasic(0.04, 1, 1e-5, 10));
});

test('plug-in GHB mean relative risk ~ 0.72 at tau2 = 0.02, eps = 1 on the paper sizes (Section 8.7)', () => {
  const n = PAPER_SIZES, K = n.length, q = { tau2: 0.02, eps: 1, delta: 1e-5, C: 1 };
  const P = matPow(metropolisHastings(topology('ring2', K)), 60), avg = avgOp(P, K), rng = makeRng(7);
  const R = 6000, acc = new Float64Array(K), accO = new Float64Array(K);
  for (let r = 0; r < R; r++) {
    const d = drawReplicate(rng, n, q);
    const g = shrink(d.th, d.v, borrow(d.x, d.r, avg)), o = oracleHB(d.th, d.x, d.v, d.s2, q.tau2);
    for (let k = 0; k < K; k++) { acc[k] += (g[k] - d.theta[k]) ** 2; accO[k] += (o[k] - d.theta[k]) ** 2; }
  }
  const rel = Array.from(acc, (s, k) => s / R * n[k]).reduce((a, b) => a + b) / K;
  const relO = Array.from(accO, (s, k) => s / R * n[k]).reduce((a, b) => a + b) / K;
  close(rel, 0.72, 0.015, 'plug-in GHB mean R_k/v_k');
  assert.ok(relO < 1, 'oracle never worse than local on average');
});

test('robust t posterior shrinks less for a far-away estimate; weighted median; GMRF fixed point', () => {
  const near = robustPosteriorMean(0.1, 0.01, 0, 0.02), far = robustPosteriorMean(2, 0.01, 0, 0.02);
  assert.ok(near < 0.1 && near > 0);
  assert.ok(far / 2 > near / 0.1, 'relative shrinkage smaller far away');
  assert.equal(weightedMedian([3, 1, 2], [1, 1, 1]), 2);
  const A = [[0, 1], [1, 0]].map((r) => Float64Array.from(r));
  const e = gmrfJacobi([1, -1], [1, -1], [1, 1], [1, 1], A, 0);
  assert.deepEqual(Array.from(e), [1, -1]);
});

test('joint release closed form matches exp5 design (16 x 20 + 8 x 200, tau2 = 0.02)', async () => {
  const { releaseUnits, jointReleaseB } = await import('../js/math/ghb.js');
  const sizes = [...Array(16).fill(20), ...Array(8).fill(200)], q = { tau2: 0.02, eps: 0.1, delta: 1e-5, C: 1 };
  const B1 = jointReleaseB(releaseUnits(sizes, 1, 20), q), B8 = jointReleaseB(releaseUnits(sizes, 8, 20), q);
  // independent re-derivation of exp5_joint_release.theory_M for g = 8
  const dp = (N) => (2 / N * Math.sqrt(2 * Math.log(1.25e5)) / 0.1) ** 2;
  let inv = 2 / (0.02 * 8 * 400 / 160 ** 2 + 1 / 160 + dp(160)); inv += 8 / (0.02 + 1 / 200 + dp(200));
  close(B8, 1 / (0.02 + 1 / inv), 1e-9, 'B, groups of 8');
  assert.ok(B8 > B1, 'joint release helps small hospitals');
});
