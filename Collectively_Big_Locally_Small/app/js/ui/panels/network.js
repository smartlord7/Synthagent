/**
 * ui/panels/network.js
 * Responsibility: the Network & gossip tab, the centrepiece. One replicate of the current configuration is
 * played as Algorithm 1: local estimates, privatized release (noise animation), T gossip rounds (particles),
 * then borrowing (shrinkage toward the network estimate, scrubbable). Shows the consensus-error curve against
 * the bound of Proposition 8.4, the shrinkage strip and the clicked-hospital inspector.
 */
import * as store from '../store.js';
import { controls, applyRelevance } from '../controls.js';
import { add, chips, explain } from '../dom.js';
import { NetworkView } from '../../viz/network.js';
import { Figure } from '../../viz/chart.js';
import { drawShrinkStrip } from '../../viz/diagrams.js';
import { fmt, reducedMotion } from '../../viz/theme.js';
import { prepare, replicate } from '../../sim/engine.js';
import { createGossip } from '../../sim/gossip.js';
import { makeRng } from '../../math/random.js';
import * as mainRun from '../mainRun.js';
import { renderInspector } from '../inspector.js';
import * as EQ from '../equations.js';

const PHASES = ['Local estimates', 'Privatized release', 'Gossip', 'Borrowing'];
const NOTES = [
  'Each hospital has fitted its own estimate θ̂<sub>k</sub> on its own patients. Colour shows the value: blue below the network average, amber above. Small hospitals scatter most, because their variance v<sub>k</sub> = σ²/n<sub>k</sub> is largest.',
  'Before anything leaves a hospital it adds Gaussian noise with sd s<sub>k</sub> = 2C√(2 ln(1.25/δ))/(n<sub>k</sub>ε) (amber halo). The release x<sub>k</sub> is all the network ever sees. The hospital keeps its noise-free estimate for itself.',
  'Each round, every hospital replaces its value by a weighted average with its neighbours (Metropolis–Hastings weights). Particles carry the values along working links; failed links are dashed. Values converge to the network average at rate λ<sup>t</sup>. Gossip only moves privatized values, so it costs no extra privacy.',
  'Each hospital removes its own share to get the leave-one-out estimate μ̃<sub>k</sub>, then shrinks its own estimate toward it with weight α<sub>k</sub>. Small hospitals move most of the way; large hospitals barely move. The strip shows local (grey), network (blue), current (teal) and true value (black tick).'
];

export function initNetwork() {
  const rail = document.getElementById('rail-network'), stage = document.getElementById('stage-network'), nb = document.getElementById('nb-network');
  add(rail, '<h3 class="rd-h">Playback</h3>');
  add(rail, `<div class="rd-btns"><button class="rd-btn primary" type="button" id="net-play" aria-pressed="false">Play</button><button class="rd-btn" type="button" id="net-step">Step</button><button class="rd-btn" type="button" id="net-end">To end</button><button class="rd-btn" type="button" id="net-reset">Restart</button></div>`);
  add(rail, `<div class="rd-ctl"><label for="net-speed"><span>Speed</span><output id="net-speed-o"></output></label><div class="rd-scale"><input type="range" id="net-speed" min="1" max="30" step="1" value="${reducedMotion ? 2 : 8}"></div></div>`);
  add(rail, `<div class="rd-ctl"><label for="net-shrink"><span>Shrinkage applied</span><output id="net-shrink-o"></output></label><div class="rd-scale"><input type="range" id="net-shrink" min="0" max="1" step="0.01" value="0"></div></div>`);
  controls(rail, ['topo', 'K', 'sizeMode', 'hops', 'clusters', 'pIn', 'pER', 'graphSeed', 'pFail', 'T', 'gammaTol'], 'net', 'Graph');
  controls(rail, ['tau2', 'eps', 'noDP', 'seed'], 'net', 'Replicate');

  const phaseEl = add(stage, `<div class="rd-phase" id="net-phase">${PHASES.map((p, i) => `<span data-i="${i}">${i + 1}. ${p}</span>`).join('')}</div>`);
  const chipEl = add(stage, '<div class="rd-chips" id="net-chips"></div>');
  const wrap = add(stage, '<figure class="rd-fig"><div class="rd-canvas-wrap rd-network-wrap"><canvas class="rd-canvas" id="net-canvas" data-h="420" tabindex="0" aria-label="Hospital network. Arrow keys move between hospitals, Enter selects one."></canvas></div><figcaption><span class="rd-fignum">Fig. N1</span> <span id="net-cap"></span></figcaption></figure>');
  const grid = add(stage, '<div class="rd-grid2"></div>');
  const figCons = new Figure(add(grid, '<div></div>'), { id: 'fig-consensus', num: 'N2', caption: '', xLabel: 'gossip round t', yLabel: 'relative consensus error', logY: true, height: 240 });
  const figStrip = new Figure(add(grid, '<div></div>'), { id: 'net-strip', num: 'N3', caption: 'Shrinkage strip, one row per hospital (labels n<sub>k</sub>): local estimate (grey), network estimate μ̃<sub>k</sub> (blue), current estimate (teal dot) and true θ<sub>k</sub> (black tick). Move the “Shrinkage applied” slider to scrub Proposition 8.1.', xLabel: '', yLabel: '', height: 300, canLogX: false, canLogY: false, toggles: false });

  const noteEl = explain(nb, 'What is happening', ['']);
  const eqEl = add(nb, '<div></div>');
  const insp = add(nb, '<div class="rd-inspector" id="net-inspector"></div>');

  const view = new NetworkView(wrap.querySelector('canvas'), { onSelect: (k) => { sel = k; refreshSide(); } });
  let ctx, rr, gp, phase = 0, shrinkT = 0, sel = 0, playing = false, busy = false;
  const speed = document.getElementById('net-speed'), shrink = document.getElementById('net-shrink');
  const setOut = () => { document.getElementById('net-speed-o').innerHTML = `${speed.value}<span class="u">rounds/s</span>`; document.getElementById('net-shrink-o').innerHTML = `${(100 * shrinkT).toFixed(0)}<span class="u">%</span>`; shrink.value = shrinkT; };

  function hospital(k) {
    return { k, n: ctx.n[k], v: ctx.v[k], s: Math.sqrt(ctx.s2[k]), x: rr.d.x[k], th: rr.d.th[k], theta: rr.d.theta[k], mu: rr.b.mu[k], M: rr.b.M[k], t2: rr.b.t2[k], alpha: rr.alpha[k], est: rr.est.ghb[k], B: ctx.Bth[k], bad: !!ctx.isBad[k] };
  }
  function refreshSide() {
    if (!ctx) return;
    const h = hospital(sel), res = mainRun.latest(), rel = res && res.ctx.K === ctx.K ? res.summary.agg.ghb.y[sel] : null;
    renderInspector(insp, h, rel);
    const c = store.get();
    eqEl.innerHTML = EQ.mechanism(h.n, c.C, ctx.priv.delta, ctx.priv.eps, h.s) + EQ.gossip(ctx.lambda, c.T, c.gammaTol, ctx.rounds) + EQ.alpha(h.t2, h.M, h.v, h.alpha, sel + 1);
    drawStrip();
  }
  function setPhase(p) {
    phase = p;
    phaseEl.querySelectorAll('span').forEach((s) => s.classList.toggle('on', +s.dataset.i === Math.min(p, 3)));
    noteEl.querySelector('p').innerHTML = NOTES[Math.min(p, 3)];
    view.label = p === 2 || (p === 1 && gp.t > 0) ? `round ${gp.t} / ${store.get().T}` : PHASES[Math.min(p, 3)].toLowerCase();
    view.draw();
  }
  function drawStrip() {
    figStrip.setCustom((c, w, h) => drawShrinkStrip(c, w, h, { n: ctx.n, th: rr.d.th, mu: rr.b.mu, est: rr.est.ghb, theta: rr.d.theta, alpha: rr.alpha }, shrinkT, sel));
  }
  function drawCons() {
    const c = store.get(), tmax = Math.max(c.T, gp.t, 10);
    figCons.setData({
      xName: 'round t', yName: 'error', xLabel: 'gossip round t (rounds)', yLabel: '‖xᵗ − x̄1‖ / ‖x⁰ − x̄1‖',
      series: [{ id: 'bound', label: 'bound λᵗ (Prop. 8.4)', tok: '--rd-muted', dash: true, width: 1.2, pts: Array.from({ length: tmax + 1 }, (_, t) => [t, Math.max(ctx.lambda ** t, 1e-12)]) },
        { id: 'obs', label: 'this replicate', tok: '--rd-ghb', width: 2, pts: gp.errors }],
      hlines: [{ y: c.gammaTol, tok: '--rd-noise', dash: true, label: `tolerance γ = ${fmt(c.gammaTol, 3)}` }]
    });
    figCons.setCaption(`Consensus error of the releases against gossip rounds (log scale, dimensionless), with the bound λ<sup>t</sup>, λ = ${fmt(ctx.lambda, 3)}. ${c.pFail > 0 ? `Each link fails with probability ${c.pFail} per round; weights are recomputed on working links, so the bound is a reference only.` : ''}`);
  }
  function rebuild() {
    stop();
    const c = store.get();
    ctx = prepare(c); rr = replicate(ctx, makeRng(c.seed * 104729 + 1), 0);
    gp = createGossip(ctx, rr.d.x, c.pFail, 99 + c.seed);
    if (sel >= ctx.K) sel = 0;
    const mean = rr.d.th.reduce((a, b) => a + b, 0) / ctx.K, sd = Math.sqrt(rr.d.th.reduce((a, b) => a + (b - mean) ** 2, 0) / ctx.K) || 0.1;
    view.setGraph({ K: ctx.K, edges: ctx.edges, sizes: ctx.n, bad: ctx.bad });
    view.failed = new Set(); view.particles = []; view.select(sel);
    view.setValues(rr.d.th, mean, 2.2 * sd);
    shrinkT = 0; setOut(); setPhase(0);
    document.getElementById('net-cap').innerHTML = `${ctx.K} hospitals; disc area ∝ n<sub>k</sub>, labels are n<sub>k</sub>; colour is the value held (blue below, amber above the average ${fmt(mean, 3)}). A red outline marks a dishonest hospital.`;
    chips(chipEl, [['SLEM λ', ctx.connected ? fmt(ctx.lambda, 3) : '1', ctx.connected ? 'ref' : 'risk'], ['Spectral gap', fmt(1 - ctx.lambda, 3)], ['Rounds to γ', ctx.connected ? Math.ceil(ctx.rounds) : '∞'],
      ['λ^T at T = ' + c.T, fmt(ctx.lambda ** c.T, 3)], ['Agreements', ctx.edges.length], ['Mean noise s/√v', fmt(Math.sqrt(ctx.s2.reduce((a, s, k) => a + s / ctx.v[k], 0) / ctx.K), 2), 'noise']]);
    drawCons(); refreshSide();
  }
  const dur = () => 1000 / +speed.value;
  async function step() {
    if (busy) return; busy = true;
    const c = store.get();
    try {
      if (phase === 0) { await view.animateNoise(rr.d.th, rr.d.x, Array.from(ctx.s2, Math.sqrt), Math.max(500, 4 * dur())); setPhase(1); }
      else if ((phase === 1 || phase === 2) && gp.t < c.T) { const r = gp.step(); setPhase(2); drawCons(); await view.animateGossip(r.prev, r.next, r.failed, dur()); }
      else if (phase === 1 || phase === 2) { setPhase(3); view.failed = new Set(); await view.animateShrink(rr.d.th, rr.est.ghb, Math.max(600, 6 * dur()), (t) => { shrinkT = t; setOut(); drawStrip(); }); setPhase(4); }
      else { rebuild(); }
    } finally { busy = false; }
  }
  async function play() { if (phase >= 4) rebuild(); playing = true; setBtn(); while (playing && phase < 4) { await step(); } playing = false; setBtn(); }
  function stop() { playing = false; view.anim?.cancel(); setBtn(); }
  const setBtn = () => { const b = document.getElementById('net-play'); b.textContent = playing ? 'Pause' : 'Play'; b.setAttribute('aria-pressed', String(playing)); };
  function toEnd() {
    stop(); const c = store.get();
    while (gp.t < c.T) gp.step();
    view.particles = []; setPhase(3); shrinkT = 1; view.setValues(rr.est.ghb); setOut(); drawCons(); drawStrip(); setPhase(4);
  }
  document.getElementById('net-play').onclick = () => (playing ? stop() : play());
  document.getElementById('net-step').onclick = () => { stop(); step(); };
  document.getElementById('net-end').onclick = toEnd;
  document.getElementById('net-reset').onclick = rebuild;
  speed.addEventListener('input', setOut);
  shrink.addEventListener('input', () => { stop(); shrinkT = +shrink.value; if (phase < 3) { while (gp.t < store.get().T) gp.step(); drawCons(); } setPhase(3); view.particles = []; view.setValues(Float64Array.from(rr.d.th, (t, k) => t + shrinkT * (rr.est.ghb[k] - t))); setOut(); drawStrip(); });

  let timer;
  store.subscribe((keys) => { applyRelevance(document.getElementById('p-network'), store.get()); clearTimeout(timer); timer = setTimeout(rebuild, 200); });
  mainRun.onResult(() => { if (ctx) renderInspector(insp, hospital(sel), mainRun.latest()?.summary.agg.ghb.y[sel] ?? null); });
  applyRelevance(document.getElementById('p-network'), store.get());
  rebuild();
  return {
    action(a) {
      if (a === 'noise') { rebuild(); step(); }
      else if (a === 'gossip') { rebuild(); view.setValues(rr.d.x); setPhase(1); play(); }
      else if (a === 'shrink') { stop(); const c = store.get(); while (gp.t < c.T) gp.step(); drawCons(); phase = 2; step(); }
    }
  };
}
