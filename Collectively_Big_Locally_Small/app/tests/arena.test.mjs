// tests/arena.test.mjs
// The Literature arena run headless on the setting of Simulation 15 at tau^2 = 0.05, epsilon = 1, K = 24 (oracle tuning
// on separate replicates, as exp15_literature.py does), checked against the paper's means (supplement/results/
// sim15_literature.json, 60 replicates, tuning on 20; sim8_baselines.json, 40 replicates) within Monte Carlo
// tolerance: |JS - Python| <= max(20% of the Python mean, 2.5 combined standard errors). The arena draws its own
// random numbers, so only agreement in distribution is expected. Runs in about a minute. GHB is the released v5 (its
// Simulation 15 rerun); GHB v4 is checked against the Simulation 15 run made with v4 before that rerun.
// Run from app/: node --test tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runArenaSync, defaultArenaSpec, ARENA_METHOD } from '../js/sim/arena.js';

// [mean, standard error] of the mean excess log-loss (nats), from the paper's runs
const REF = {
  local: [0.02709, 0.00058, 'Simulation 15'], ghb: [0.01923, 0.00063, 'Simulation 15'], ghbV4: [0.01588, 0.00044, 'Simulation 15, v4 run'], lite: [0.01382, 0.00039, 'Simulation 15'],
  liteOracle: [0.01244, 0.00032, 'Simulation 15'], dittoStyle: [0.01542, 0.00056, 'Simulation 15'], dpfedavg: [0.02742, 0.0008, 'Simulation 15'],
  dpfedavgFT: [0.01532, 0.0004, 'Simulation 15'], dec: [0.01905, 0.00068, 'Simulation 15'], clu: [0.02137, 0.00145, 'Simulation 15'],
  lgd: [0.01903, 0.00068, 'Simulation 15'], pooled: [0.01735, 0.00034, 'Simulation 15'],
  fedavg1: [0.03646, 0.00203, 'Simulation 8'], meta: [0.03028, 0.00154, 'Simulation 8']
};

test('arena reproduces the Simulation 15 (and Simulation 8) means at tau^2 = 0.05, eps = 1, K = 24', () => {
  const spec = { ...defaultArenaSpec(), K: 24, tau2: 0.05, eps: 1, tuning: 'oracle', reps: 20, tuneReps: 10, seed: 15, methods: Object.keys(REF) };
  const r = runArenaSync(spec);
  const lines = [], bad = [];
  for (const [id, [m, se, src]] of Object.entries(REF)) {
    const s = r.summary[id], tol = Math.max(0.2 * m, 2.5 * Math.hypot(se, s.se)), ok = Math.abs(s.mean - m) <= tol;
    lines.push(`${ARENA_METHOD[id].label.padEnd(34)} JS ${(1000 * s.mean).toFixed(1).padStart(5)} ± ${(2000 * s.se).toFixed(1)}   paper ${(1000 * m).toFixed(1).padStart(5)} (${src})   ${ok ? 'ok' : 'OUTSIDE'}${s.config ? '   ' + JSON.stringify(s.config) : ''}`);
    if (!ok) bad.push(id);
  }
  console.log(lines.join('\n'));
  assert.deepEqual(bad, [], 'methods outside Monte Carlo tolerance');
  // the qualitative findings of Section 6.3 at this setting
  const S = r.summary;
  assert.ok(S.lite.mean < S.local.mean && S.ghb.mean < S.local.mean, 'GHB and GHB-lite beat local estimation');
  assert.ok(S.ghb.worst < 1.25 && S.lite.worst < 1.25, 'their worst hospital stays close to its local error');
  assert.ok(S.dpfedavgFT.worst > S.lite.worst, 'fine-tuning leaves some hospital worse off than GHB-lite does');
  assert.ok(S.fedavg1.mean > S.local.mean && S.meta.mean > S.local.mean, 'one model for all hospitals is worse than local (Simulation 8)');
});

test('arena paper-default mode runs every method of the paper and keeps each method\'s numbers independent of the others', () => {
  const base = { ...defaultArenaSpec(), reps: 2, nTest: 1000, seed: 3 };
  const all = runArenaSync({ ...base, methods: Object.keys(ARENA_METHOD) });
  for (const id of Object.keys(ARENA_METHOD)) assert.ok(Number.isFinite(all.summary[id].mean) && all.summary[id].mean > 0, id);
  const two = runArenaSync({ ...base, methods: ['ghb', 'pfedme'] });
  assert.equal(two.summary.ghb.mean, all.summary.ghb.mean, 'GHB unchanged by the selection');
  assert.equal(two.summary.pfedme.mean, all.summary.pfedme.mean, 'pFedMe unchanged by the selection');
  const man = runArenaSync({ ...base, methods: ['dpfedavgFT'], tuning: 'manual', manual: { dpfedavgFT: { R: 3, eta: 2, E: 0 } } });
  assert.deepEqual(man.summary.dpfedavgFT.config, { R: 3, eta: 2, E: 0 });
});
