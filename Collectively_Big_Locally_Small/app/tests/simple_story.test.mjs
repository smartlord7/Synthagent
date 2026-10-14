// tests/simple_story.test.mjs
// The animated network of Simple mode, run headless: the data behind every step comes from the simulation
// (sim/simpleStory.js). Trust weights are in [0, 1], the movement is exactly GHB's estimate, node sizes follow
// patients, the scrambles cancel, gossip reaches the noisy total, a very different hospital trusts the network less,
// and every step has plain text without long dashes.
// Run from app/: node --test tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { STORY_STEPS, storyData, stepCaption, nodeRadius, noiseLevel, ATYPICAL_SHIFT, CONSENSUS } from '../js/sim/simpleStory.js';
import { defaultChoices, simpleSizes, runSimpleSync } from '../js/sim/simple.js';

const DASHES = /[—–]|--/;
const SETTINGS = [
  defaultChoices(),
  { ...defaultChoices(), K: 20, sizes: 'small', difference: 'similar', privacy: 'high' },
  { ...defaultChoices(), K: 6, sizes: 'large', difference: 'very', privacy: 'low' }
];

test('story data comes from the run: sizes, trust in [0, 1], GHB movement, cancelling masks, gossip to the total', () => {
  for (const ch of SETTINGS) {
    const D = storyData(ch);
    assert.deepEqual(D.n, simpleSizes(ch), 'hospitals and sizes are the Simple mode setting');
    assert.equal(D.K, D.n.length); assert.ok(D.edges.length >= D.K, 'a connected network of agreements');
    for (let k = 0; k < D.K; k++) {
      assert.ok(D.trust[k] >= 0 && D.trust[k] <= 1, `trust of hospital ${k}`);
      assert.ok(Math.abs(D.trust[k] + D.alpha[k] - 1) < 1e-12);
      const moved = D.alpha[k] * D.own[k] + (1 - D.alpha[k]) * D.network[k];
      assert.ok(Math.abs(moved - D.ghb[k]) < 1e-9, 'the animated movement ends at GHB\'s estimate');
      assert.ok(Math.abs(D.sd[k] - Math.sqrt(1 / D.n[k])) < 1e-12, 'haze is the standard error sqrt(sigma^2 / n)');
    }
    assert.ok(Math.abs(D.masks.reduce((a, b) => a + b, 0)) < 1e-6 * Math.max(...D.masks.map(Math.abs)), 'pairwise masks cancel in the total');
    const plain = D.n.reduce((a, n, k) => a + n * D.own[k], 0) + D.xi;
    assert.ok(Math.abs(D.total - plain) < 1e-6 * Math.max(1, Math.abs(plain)), 'masked total = sum of contributions + privacy noise');
    assert.ok(D.spread <= CONSENSUS && D.final.every((x) => Math.abs(D.K * x - D.total) <= 1.0001 * CONSENSUS * Math.abs(D.total)), 'after the gossip every hospital holds the total');
    assert.ok(stepCaption('combine', D)[1].includes(`After ${D.rounds} rounds`), 'the caption reports the rounds actually run');
    assert.equal(D.frames.length, 7, 'first gossip rounds kept for the animation');
    // the network value seen by hospital k is the noisy total without its own part
    const k = 0, N = D.N;
    assert.ok(Math.abs(D.network[k] - (D.n.reduce((a, n, i) => a + n * D.own[i], 0) + D.xi - D.n[k] * D.own[k]) / (N - D.n[k])) < 1e-9);
  }
});

test('node sizes follow patients (area proportional to patients)', () => {
  const D = storyData(defaultChoices()), room = 30, r = D.n.map((_, k) => nodeRadius(D.n, k, room));
  for (let k = 1; k < D.K; k++) assert.ok(r[k] >= r[k - 1], 'bigger hospital, bigger circle');
  assert.equal(r[D.K - 1], room);
  for (let k = 0; k < D.K; k++) if (r[k] > 6) assert.ok(Math.abs(r[k] ** 2 / room ** 2 - D.n[k] / D.n[D.K - 1]) < 1e-9);
});

test('the privacy setting changes the noise drawn', () => {
  const lv = ['low', 'medium', 'high'].map((privacy) => noiseLevel(storyData({ ...defaultChoices(), privacy }).noiseSd));
  assert.ok(lv[0] < lv[1] && lv[1] < lv[2], lv.join(' < '));
  assert.ok(lv.every((x) => x >= 0 && x <= 1));
});

test('making a hospital very different lowers its trust in the network (same random draws)', () => {
  for (const ch of SETTINGS) {
    const base = storyData(ch);
    for (const k of [0, Math.floor(base.K / 2), base.K - 1]) {
      const A = storyData(ch, { atypical: k });
      assert.ok(A.trust[k] < base.trust[k] || A.trust[k] < 0.02, `hospital ${k}: ${A.trust[k]} vs ${base.trust[k]}`);
      assert.ok(Math.abs(A.theta[k] - (ATYPICAL_SHIFT * Math.sqrt(require0(ch)))) < 1e-12);
      const cap = stepCaption('trust', A, null, base).join(' ');
      assert.match(cap, new RegExp(`Hospital ${k + 1} \\(${A.n[k]} patients\\) is now very different`));
    }
  }
});
function require0(ch) { return { similar: 0.005, somewhat: 0.02, very: 0.2 }[ch.difference]; }

test('every step has a title and plain captions with numbers from the run, without long dashes', () => {
  assert.equal(STORY_STEPS.length, 7);
  const ch = defaultChoices(), D = storyData(ch), res = runSimpleSync(ch);
  for (const s of STORY_STEPS) {
    const cap = stepCaption(s.id, D, res);
    assert.ok(s.title.length > 5 && cap.length >= 1, s.id);
    for (const x of [s.title, ...cap]) { assert.ok(!DASHES.test(x), x); assert.ok(!/NaN|undefined|Infinity/.test(x), x); assert.ok(!/[Ͱ-Ͽ]/.test(x), `no Greek letters: ${x}`); }
  }
  assert.ok(stepCaption('trust', D).join(' ').includes(`trusts it ${Math.round(100 * D.trust[0])}%`));
  const better = res.hospitals.filter((h) => h.ghb < 1).length;
  assert.ok(stepCaption('result', D, res)[0].includes(`${better} of ${D.K} hospitals`));
  const src = readFileSync(new URL('../js/ui/simpleNetwork.js', import.meta.url), 'utf8');
  assert.ok(!/[—–]/.test(src), 'no long dash characters in ui/simpleNetwork.js');
  const strings = src.replace(/var\(--[a-z0-9-]+\)|tok\('--[a-z0-9-]+'\)/g, '').match(/`[^`]*`|'[^'\n]*'/g) || [];
  for (const s of strings) assert.ok(!/(^|[^-<!])--(?![a-z>])/.test(s), `no double hyphen in: ${s.slice(0, 80)}`);
});
