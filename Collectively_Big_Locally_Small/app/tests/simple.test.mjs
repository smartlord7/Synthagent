// tests/simple.test.mjs
// Simple mode (sim/simple.js) run headless: the plain choices map to the existing simulations, every hospital gets a
// result, and every sentence agrees with the numbers of the run (no hard-coded claims). Also checks the author's
// writing rule: no long dashes in any Simple mode text.
// Run from app/: node --test tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { CHOICES, HOSPITALS, STORY, STEPS, defaultChoices, simpleConfig, simpleArenaSpec, simpleSizes, runSimpleSync, runFederatedSync, describe } from '../js/sim/simple.js';
import { siteSizes } from '../js/math/random.js';

const DASHES = /[—–]|--/;
const pct = (x) => Math.round(100 * x);

/** Check every sentence of a run against its numbers. */
function checkSentences(r, d) {
  const H = r.hospitals, all = [...d.main, ...d.compare];
  assert.ok(d.main.length >= 3 && d.main.length <= 3, 'three main findings');
  for (const s of all) { assert.ok(s.text.length > 20, s.kind); assert.ok(!DASHES.test(s.text), `no long dashes: ${s.text}`); assert.ok(!/NaN|undefined|Infinity/.test(s.text), s.text); }
  const kinds = d.main.map((s) => s.kind), worse = H.filter((h) => h.ghb > 1);
  // "no hospital worse off" exactly when it is true
  assert.equal(kinds.includes('none-worse'), worse.length === 0, 'none-worse iff every hospital is at or below working alone');
  assert.equal(kinds.includes('worse'), worse.length > 0);
  if (worse.length) assert.ok(d.main.find((s) => s.kind === 'worse').text.startsWith(`${worse.length} hospital`), 'the count of worse-off hospitals is stated');
  // the smallest hospital: the "times as many patients" figure is 1 / relative error
  const small = d.main.find((s) => s.kind === 'small').text, g = 1 / H[0].ghb;
  if (g >= 1.1) {
    const m = small.match(/about ([\d.]+) times as many patients \(about (\d+) instead of (\d+)\)/);
    assert.ok(m, small);
    assert.ok(Math.abs(Number(m[1]) - g) <= 0.05 + 0.05 * g, `multiplier ${m[1]} vs ${g}`);
    assert.equal(Number(m[2]), Math.round(H[0].n * g)); assert.equal(Number(m[3]), H[0].n);
  } else if (H[0].ghb > 1) assert.match(small, /worse off/);
  // when GHB helps little on average, the page says so
  const avg = H.reduce((a, h) => a + h.ghb, 0) / H.length;
  assert.equal(kinds.includes('weak'), avg > 0.9, 'the honest "helps little" sentence appears exactly when the average cut is under 10%');
  // pooled sentence counts hospitals above working alone
  const pooled = d.compare.find((s) => s.kind === 'pooled');
  if (r.choices.pooled) {
    const pw = H.filter((h) => h.pooled > 1).length;
    if (pw) assert.ok(pooled.text.includes(`${pw} of ${H.length} hospitals worse off`), pooled.text);
    else assert.match(pooled.text, /help every hospital/);
  } else assert.equal(pooled, undefined);
  // privacy sentence quotes the two averages
  const priv = d.compare.find((s) => s.kind === 'privacy'), np = H.reduce((a, h) => a + h.noPrivacy, 0) / H.length;
  assert.ok(priv.text.includes(String(pct(np))), priv.text);
}

test('the choices map to the existing engine parameters (documented mapping)', () => {
  const c = simpleConfig({ ...defaultChoices(), K: 8, sizes: 'small', difference: 'very', privacy: 'high' });
  assert.equal(c.sizeMode, 'table'); assert.deepEqual(c.sizesTable, siteSizes(8, 30));
  assert.equal(c.tau2, 0.2); assert.equal(c.eps, 0.1); assert.equal(c.noDP, false);
  const p = simpleConfig(defaultChoices());
  assert.equal(p.tau2, 0.02); assert.equal(p.eps, 1); assert.deepEqual(p.sizesTable, siteSizes(HOSPITALS.def, 100));
  const a = simpleArenaSpec({ ...defaultChoices(), K: 8, sizes: 'large', difference: 'similar', privacy: 'low' });
  assert.equal(a.K, 8); assert.equal(a.median, 400); assert.equal(a.tau2, 0.01); assert.equal(a.eps, 10);
  assert.deepEqual(a.methods, ['local', 'ghb', 'dpfedavgFT']);
  assert.deepEqual(simpleSizes({ ...defaultChoices(), K: 99 }).length, HOSPITALS.max, 'K is clamped to the slider range');
});

test('default scenario: per-hospital results and sentences consistent with the numbers', () => {
  const r = runSimpleSync(defaultChoices());
  assert.equal(r.hospitals.length, HOSPITALS.def);
  for (const h of r.hospitals) {
    assert.ok(Number.isFinite(h.ghb) && h.ghb > 0 && Number.isFinite(h.pooled) && Number.isFinite(h.noPrivacy), `hospital ${h.k}`);
    assert.ok(h.weightOwn > 0 && h.weightOwn <= 1);
  }
  assert.ok(r.hospitals[0].ghb < 0.8, 'the smallest hospital gains clearly at the paper defaults');
  checkSentences(r, describe(r));
});

test('similar hospitals: GHB helps everyone and the page says no hospital is worse off', () => {
  const r = runSimpleSync({ ...defaultChoices(), sizes: 'small', difference: 'similar', privacy: 'medium', pooled: false });
  assert.ok(r.hospitals.every((h) => h.ghb < 1));
  const d = describe(r);
  assert.ok(d.main.some((s) => s.kind === 'none-worse'));
  checkSentences(r, d);
});

test('very different hospitals, high privacy: an honest "helps little" statement, and worse-off hospitals are named', () => {
  const r = runSimpleSync({ ...defaultChoices(), K: 10, sizes: 'large', difference: 'very', privacy: 'high' });
  const d = describe(r);
  assert.ok(d.main.some((s) => s.kind === 'weak' && /too different/.test(s.text)), JSON.stringify(d.main));
  checkSentences(r, d);
  // a fabricated run where one hospital loses clearly must be reported as worse off, never as "no hospital worse off"
  const fake = { ...r, hospitals: r.hospitals.map((h, i) => (i === 3 ? { ...h, ghb: 1.4, ghbSE: 0.01 } : { ...h, ghb: Math.min(h.ghb, 0.95) })) };
  const df = describe(fake);
  assert.ok(df.main.some((s) => s.kind === 'worse' && /Hospital 4/.test(s.text) && /40% more error/.test(s.text)), JSON.stringify(df.main));
  checkSentences(fake, df);
});

test('federated comparison (short arena run) is summarized from its own numbers', () => {
  const ch = { ...defaultChoices(), K: 6, sizes: 'small' };
  const f = runFederatedSync(ch, { reps: 2, nTest: 500 });
  assert.equal(f.K, 6); assert.equal(f.fed.ratio.length, 6);
  assert.equal(f.fed.worse, f.fed.ratio.filter((x) => x > 1).length);
  const r = runSimpleSync(ch), d = describe(r, f), s = d.compare.find((x) => x.kind === 'federated');
  assert.ok(s.text.includes(`leaves ${f.fed.worse} of 6 hospitals worse off`) && s.text.includes(`leaves ${f.ghb.worse} worse off`), s.text);
  checkSentences(r, d);
});

test('no long dashes in any Simple mode text', () => {
  const texts = [...STORY, ...STEPS, HOSPITALS.label, HOSPITALS.help, ...Object.values(CHOICES).flatMap((g) => [g.label, g.help, ...g.options.map((o) => o.label)])];
  for (const t of texts) assert.ok(!DASHES.test(t), t);
  // the page module's visible strings (template literals and quoted text), CSS variable names excluded
  const src = readFileSync(new URL('../js/ui/simple.js', import.meta.url), 'utf8').replace(/var\(--[a-z0-9-]+\)/g, '');
  assert.ok(!/[—–]/.test(src), 'no long dash characters in ui/simple.js');
  const strings = src.match(/`[^`]*`|'[^'\n]*'/g) || [];
  for (const s of strings) assert.ok(!/(^|[^-<!])--(?![a-z>])/.test(s), `no double hyphen in: ${s.slice(0, 80)}`);
});
