// tests/logit.test.mjs
// The logistic heads of the Literature arena against the paper's Python code (simulations/*.py) on one small fixed
// network: every method must reproduce the Python output to 1e-8 when it is given the same data and the same normal
// draws. fixtures/logit_reference.json was written by the script below, run from simulations/ as
//   python3 make_fixture.py ../app/tests/fixtures/logit_reference.json
// Run from app/: node --test tests/
//
// """Reference values for app/tests/logit.test.mjs, computed by the paper's Python code on a small fixed network."""
// import json, sys
// import numpy as np
// sys.path.insert(0, '/home/user/TabuLLM/Collectively_Big_Locally_Small/simulations')
// import ghb, ghb_logit as GL, ghb_v4 as V4, literature as LIT, personalized as PZ, logit_world as L
// from exp7_logistic import fit, MU, P
// from scipy.special import expit
//
// rng = np.random.default_rng(2026)
// n = np.array([12., 25., 40., 60., 90.])
// K = len(n)
// theta = MU + np.sqrt(0.05) * rng.standard_normal((K, P))
// mu0 = MU + 0.3 * rng.standard_normal(P)
// data = []
// for k in range(K):
//     X = L.features(rng, int(n[k]))
//     data.append((X, rng.binomial(1, expit(X @ theta[k])).astype(float)))
// pubX = L.features(rng, 400)
// pub = GL.Public(pubX)
// W = ghb.metropolis_hastings(np.array([[0,1,1,0,0],[1,0,1,1,0],[1,1,0,1,1],[0,1,1,0,1],[0,0,1,1,0]], float))
// avg = ghb.Averager(W, 7)
// out = dict(n=n.tolist(), mu0=mu0.tolist(), data=[dict(X=X.ravel().tolist(), y=y.tolist()) for X, y in data],
//            pubX=pubX.ravel().tolist(), W=W.tolist(), T=7)
// # 1 fit
// out['fit_local2'] = fit(data[2][0], data[2][1], mu0, np.ones(P)).tolist()
// out['fit_ridge4'] = fit(data[4][0], data[4][1], np.zeros(P), np.full(P, 3.0)).tolist()
// # 2 GHB-lite without noise
// g0, sd0 = GL.release(rng, data, n, mu0, np.inf)
// th, r, drop = GL.invert(pub, g0, sd0, n, mu0, avg, 0.3, True)
// out['lite_inf'] = dict(th=th.tolist(), r=r.tolist(), drop=drop.tolist(), est=GL.estimate(data, th, r, drop, mu0, avg, 0.3).tolist(),
//                        oracle=GL.estimate(data, th, r, drop, mu0, avg, 0.3, oracle_tau2=0.05).tolist())
// out['fedavg1'] = (th * (n / n.sum())[:, None]).sum(0).tolist()
// t2 = np.array([ghb.dersimonian_laird(th[None, :, d], r[None, :, d], ghb.Averager())[0, 0] for d in range(P)])
// w = 1.0 / (t2[None, :] + r); out['meta'] = ((th * w).sum(0) / w.sum(0)).tolist()
// # 3 GHB-lite with given noisy releases
// g1, sd1 = GL.release(rng, data, n, mu0, 1.0)
// th1, r1, drop1 = GL.invert(pub, g1, sd1, n, mu0, avg, 0.3, True)
// out['lite_eps1'] = dict(g=g1.tolist(), sd=sd1.tolist(), th=th1.tolist(), r=r1.tolist(), est=GL.estimate(data, th1, r1, drop1, mu0, avg, 0.3).tolist())
// # 4 GHB (v4): no privacy, then eps = 1 with replayed normals
// cfg = dict(f=1.0, safety=True, R1=1, pan=False, anchor=(mu0, 0.3))
// out['v4_inf'] = V4.estimate({'x': data}, np.random.default_rng(1), np.inf, 1e-5, np.sqrt(5.0), cfg=cfg)['x'].tolist()
// z = np.random.default_rng(77).standard_normal(P * P + P)
// out['v4_eps1'] = dict(z=z.tolist(), est=V4.estimate({'x': data}, np.random.default_rng(77), 1.0, 1e-5, np.sqrt(5.0), cfg=cfg)['x'].tolist())
// cfg2 = dict(cfg, anchor=None)
// out['v4_eps1_noanchor'] = V4.estimate({'x': data}, np.random.default_rng(77), 1.0, 1e-5, np.sqrt(5.0), cfg=cfg2)['x'].tolist()
// # 5 literature with replayed normals
// rho = LIT.rho_total(1.0, 1e-5)
// z = np.random.default_rng(78).standard_normal(500)
// out['lit_z'] = z.tolist()
// out['fedavg'] = LIT.fedavg(np.random.default_rng(78), data, n, mu0, rho, 3, 0.5).tolist()
// out['dec'] = LIT.decentralized(np.random.default_rng(78), data, n, mu0, rho, 3, 0.2, W).tolist()
// out['ft'] = {str(k): v.tolist() for k, v in LIT.fine_tune(data, np.repeat(mu0[None], K, 0), steps=(0, 3, 10)).items()}
// out['ditto'] = PZ.ditto(np.random.default_rng(78), data, n, rho, 3, 0.5, 1.0).tolist()
// out['pfedme'] = PZ.pfedme(np.random.default_rng(78), data, n, rho, 3, 2.0, 10.0).tolist()
// out['apfl'] = PZ.apfl(np.random.default_rng(78), data, n, rho, 3, 0.5, 0.75).tolist()
// out['perfedavg'] = PZ.per_fedavg(np.random.default_rng(78), data, n, rho, 3, 0.5, 0.1).tolist()
// out['fedrep'] = {str(k): v.tolist() for k, v in PZ.fedrep(np.random.default_rng(78), data, n, rho, 3, 0.5).items()}
// out['fedprox'] = {str(k): v.tolist() for k, v in PZ.fedprox(np.random.default_rng(78), data, n, rho, 3, 0.2, 0.1).items()}
// import ghb_sa as SA
// out['fedavg_sa'] = SA.fedavg_sa(np.random.default_rng(78), data, n, mu0, rho, 3, 0.5).tolist()
// out['rho'] = {str(e): GL.rho_total(e, 1e-5) for e in (0.3, 1.0, 3.0, 10.0)}
// # grids
// out['grid_q'] = GL.tau2_quantiles(th1, r1).tolist()
// pp = 1.0 / (1 + (GL.GRID / GL.SCALE) ** 2) * np.gradient(GL.GRID)
// import ghb_v3 as V3
// out['prior_q'] = V3.quantiles(pp / pp.sum()).tolist()
// json.dump(out, open(sys.argv[1], 'w'))
// print('ok')
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as L from '../js/math/logit.js';
import * as GL from '../js/math/ghbLogit.js';
import * as LIT from '../js/math/literature.js';
import { avgOp } from '../js/math/ghb.js';
import { matPow } from '../js/math/graph.js';

const F = JSON.parse(fs.readFileSync(fileURLToPath(new URL('./fixtures/logit_reference.json', import.meta.url))));
const data = F.data.map((d) => ({ X: Float64Array.from(d.X), y: Float64Array.from(d.y), n: d.y.length }));
const n = F.n, K = n.length, mu0 = Float64Array.from(F.mu0);
const W = F.W.map((r) => Float64Array.from(r)), avg = avgOp(matPow(W, F.T), K), pub = new L.Public(Float64Array.from(F.pubX));
const flat = (a) => (Array.isArray(a) || ArrayBuffer.isView(a) ? Array.from(a).flatMap((x) => (Array.isArray(x) || ArrayBuffer.isView(x) ? Array.from(x) : [x])) : [a]);
const maxDiff = (a, b) => { const A = flat(a), B = flat(b); assert.equal(A.length, B.length, 'shape'); return Math.max(...A.map((x, i) => Math.abs(x - B[i]))); };
const close = (a, b, tol, msg) => { const d = maxDiff(a, b); assert.ok(d <= tol, `${msg}: max difference ${d} > ${tol}`); };
const replay = (z) => { let i = 0; return { n: () => { if (i >= z.length) throw new Error('replay exhausted'); return z[i++]; }, u: () => { throw new Error('no uniforms in replay'); } }; };
const rho1 = L.rhoTotal(1, 1e-5);

test('zCDP budget rho_total matches the Python root of rho + 2 sqrt(rho ln(1/delta)) = eps (Proposition 5.1)', () => {
  for (const [e, v] of Object.entries(F.rho)) assert.ok(Math.abs(L.rhoTotal(+e, 1e-5) - v) < 1e-12, `eps ${e}`);
  assert.equal(L.rhoTotal(Infinity, 1e-5), Infinity);
});

test('logistic MAP fit: closed form for an intercept-only design, and the Python fit on fixed data', () => {
  const m = 40, X = new Float64Array(m * L.P), y = new Float64Array(m);
  for (let i = 0; i < m; i++) { X[i * L.P] = 1; y[i] = i < 10 ? 1 : 0; }
  const t = L.fit({ X, y, n: m }, new Float64Array(L.P), new Float64Array(L.P).fill(1e-9));
  assert.ok(Math.abs(t[0] - Math.log(10 / 30)) < 1e-6, `intercept ${t[0]}`);
  for (let a = 1; a < L.P; a++) assert.ok(Math.abs(t[a]) < 1e-9);
  close(L.fit(data[2], mu0, new Float64Array(L.P).fill(1)), F.fit_local2, 1e-9, 'local fit, prior N(mu0, I)');
  close(L.fit(data[4], new Float64Array(L.P), new Float64Array(L.P).fill(3)), F.fit_ridge4, 1e-8, 'ridge fit');
});

test('GHB-lite (Section 5.3) reproduces ghb_logit.invert and estimate, with and without privacy noise', () => {
  const g0 = data.map((d) => L.scoreSum(d, mu0).map((v) => v / d.n));
  let iv = GL.liteInvert(pub, g0, new Float64Array(K), n, mu0, avg, 0.3, true);
  close(iv.th, F.lite_inf.th, 1e-12, 'inverted releases'); close(iv.r, F.lite_inf.r, 1e-12, 'their variances');
  assert.deepEqual(iv.drop, F.lite_inf.drop);
  close(GL.liteEstimate(data, iv, mu0, avg, 0.3), F.lite_inf.est, 1e-8, 'estimates, no DP');
  close(GL.liteEstimate(data, iv, mu0, avg, 0.3, 0.05), F.lite_inf.oracle, 1e-8, 'estimates, known tau^2');
  close(LIT.oneShotFedavg(iv.th, n), F.fedavg1, 1e-12, 'one-shot FedAvg of the releases');
  close(LIT.metaAnalysis(iv.th, iv.r), F.meta, 1e-12, 'DerSimonian-Laird meta-analysis');
  iv = GL.liteInvert(pub, F.lite_eps1.g.map((v) => Float64Array.from(v)), Float64Array.from(F.lite_eps1.sd), n, mu0, avg, 0.3, true);
  close(iv.th, F.lite_eps1.th, 1e-12, 'noisy releases inverted');
  close(GL.liteEstimate(data, iv, mu0, avg, 0.3), F.lite_eps1.est, 1e-8, 'estimates at eps = 1');
  close(GL.posteriorQuantiles(iv.th.map((v) => v[0]), iv.r.map((v) => v[0])), F.grid_q.map((q) => q[0]), 1e-14, 'posterior quantiles of tau^2');
  close(GL.priorQuantiles(), F.prior_q, 1e-12, 'half-Cauchy prior quantiles');
});

test('GHB (Section 5.2, two joint releases) reproduces ghb_v4.estimate with the frozen configuration', () => {
  close(GL.ghbJoint(replay(new Array(40).fill(0)), data, Infinity, 1e-5, { anchor: { mu0, sd: 0.3 } }).est, F.v4_inf, 1e-8, 'no privacy, anchor');
  close(GL.ghbJoint(replay(F.v4_eps1.z), data, 1, 1e-5, { anchor: { mu0, sd: 0.3 } }).est, F.v4_eps1.est, 1e-8, 'eps = 1, anchor');
  close(GL.ghbJoint(replay(F.v4_eps1.z), data, 1, 1e-5, {}).est, F.v4_eps1_noanchor, 1e-8, 'eps = 1, no anchor');
});

test('literature methods reproduce literature.py, personalized.py and ghb_sa.py with the same noise draws', () => {
  const z = F.lit_z;
  close(LIT.fedavg(replay(z), data, n, mu0, rho1, 3, 0.5), F.fedavg, 1e-12, 'DP-FedAvg (McMahan et al., 2018)');
  close(LIT.fedavgSA(replay(z), data, n, mu0, rho1, 3, 0.5), F.fedavg_sa, 1e-12, 'DP-FedAvg with secure aggregation');
  close(LIT.decentralized(replay(z), data, n, mu0, rho1, 3, 0.2, W), F.dec, 1e-12, 'decentralized DP-SGD (Lian et al., 2017)');
  const ft = LIT.fineTune(data, LIT.repeat(mu0, K), [0, 3, 10]);
  for (const E of [0, 3, 10]) close(ft.get(E), F.ft[E], 1e-12, `fine-tuning, E = ${E}`);
  close(LIT.ditto(replay(z), data, n, rho1, 3, 0.5, 1), F.ditto, 1e-9, 'Ditto (Li et al., 2021)');
  close(LIT.pfedme(replay(z), data, n, rho1, 3, 2, 10), F.pfedme, 1e-8, 'pFedMe (T. Dinh et al., 2020)');
  close(LIT.apfl(replay(z), data, n, rho1, 3, 0.5, 0.75), F.apfl, 1e-12, 'APFL (Deng et al., 2020)');
  close(LIT.perFedavg(replay(z), data, n, rho1, 3, 0.5, 0.1), F.perfedavg, 1e-12, 'Per-FedAvg (Fallah et al., 2020)');
  const fr = LIT.fedrep(replay(z), data, n, rho1, 3, 0.5);
  for (const E of [0, 5]) close(fr.get(E), F.fedrep[E], 1e-12, `FedRep (Collins et al., 2021), E = ${E}`);
  const fp = LIT.fedprox(replay(z), data, n, rho1, 3, 0.2, 0.1);
  for (const E of [0, 5, 30]) close(fp.get(E), F.fedprox[E], 1e-12, `FedProx (Li et al., 2020), E = ${E}`);
});
