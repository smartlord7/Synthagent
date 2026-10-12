/**
 * ui/panels/network.js
 * Responsibility: the Network & gossip tab. One replicate of the current configuration is played as Algorithm 1,
 * for either protocol. GHB-lite: local estimates, privatized per-hospital release (noise animation), T gossip
 * rounds (particles), then borrowing. GHB: local estimates, masked shares of the network sum (pairwise masks that
 * cancel in the total plus a share of the noise), gossip of the shares to the noisy total, then each hospital's
 * exact leave-one-out mean and borrowing with tau integrated out. Shows the consensus-error curve against the bound
 * of Proposition 4.3, the shrinkage strip and the clicked-hospital inspector.
 */
import * as store from '../store.js';
import { controls, applyRelevance, segmented } from '../controls.js';
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

const PHASES = { lite: ['Local estimates', 'Privatized release', 'Gossip', 'Borrowing'], ghb: ['Local estimates', 'Masked shares', 'Gossip to the sum', 'Borrowing'] };
const NOTES = {
  lite: [
    'Each hospital has fitted its own estimate θ̂<sub>k</sub> on its own patients. Colour shows the value: blue below the network average, amber above. Small hospitals scatter most, because their variance v<sub>k</sub> = σ²/n<sub>k</sub> is largest.',
    'GHB-lite: before anything leaves a hospital it adds Gaussian noise with sd s<sub>k</sub> = 2C√(2 ln(1.25/δ))/(n<sub>k</sub>ε) (amber halo). The release x<sub>k</sub> is all the network ever sees. The noise grows as 1/n<sub>k</sub>, so small hospitals add the most. The hospital keeps its noise-free estimate for itself.',
    'Each round, every hospital replaces its value by a weighted average with its neighbours (Metropolis-Hastings weights). Particles carry the values along working links; failed links are dashed. Values converge to the network average at rate λ<sup>t</sup>. Gossip only moves privatized values, so it costs no extra privacy (Proposition 4.3).',
    'Each hospital removes its own share to get the leave-one-out estimate μ̃<sub>k</sub>, then shrinks its own estimate toward it with weight α<sub>k</sub>. Small hospitals move most of the way; large hospitals barely move. The strip shows local (grey), network (blue), current (teal) and true value (black tick).'
  ],
  ghb: [
    'Each hospital has fitted its own estimate θ̂<sub>k</sub> on its own patients. Colour shows the value: blue below the network average, amber above. Small hospitals scatter most, because their variance v<sub>k</sub> = σ²/n<sub>k</sub> is largest.',
    'GHB: each hospital forms its contribution n<sub>k</sub>θ̂<sub>k</sub>, adds pairwise random masks agreed with its neighbours (each mask is added by one end of a link and subtracted by the other, so they cancel in the total) and a share N(0, s²/K) of the privacy noise (amber halo). The masked share is all that leaves the hospital, and on its own it looks like noise. Shown here: share × K/N.',
    'Gossip averages the masked shares. The masks cancel, so the values converge to S̃/N, the noisy network total divided by the number of patients; every hospital then knows S̃ = Σ n<sub>j</sub>θ̂<sub>j</sub> + N(0, s²). The noise s = 2C/√(2ρ) is calibrated to the sum (sensitivity 2C), so it does not grow for small hospitals (Section 5.2, Proposition 5.1).',
    'Each hospital removes its own contribution exactly, μ<sub>−k</sub> = (S̃ − n<sub>k</sub>θ̂<sub>k</sub>)/(N − n<sub>k</sub>), and, having no per-hospital release to estimate τ from, weighs the quantiles of its half-Cauchy prior on τ by its own evidence (Remark 4.4). Small hospitals move most of the way; large ones barely move.'
  ]
};

export function initNetwork() {
  const rail = document.getElementById('rail-network'), stage = document.getElementById('stage-network'), nb = document.getElementById('nb-network');
  rail.appendChild(segmented('protocol', 'net', 'Protocol played'));
  add(rail, '<p class="rd-msg">"Both" plays GHB; switch to GHB-lite to see per-hospital releases.</p>');
  add(rail, '<h3 class="rd-h">Playback</h3>');
  add(rail, `<div class="rd-btns"><button class="rd-btn primary" type="button" id="net-play" aria-pressed="false">Play</button><button class="rd-btn" type="button" id="net-step">Step</button><button class="rd-btn" type="button" id="net-end">To end</button><button class="rd-btn" type="button" id="net-reset">Restart</button></div>`);
  add(rail, `<div class="rd-ctl"><label for="net-speed"><span>Speed</span><output id="net-speed-o"></output></label><div class="rd-scale"><input type="range" id="net-speed" min="1" max="30" step="1" value="${reducedMotion ? 2 : 8}"></div></div>`);
  add(rail, `<div class="rd-ctl"><label for="net-shrink"><span>Shrinkage applied</span><output id="net-shrink-o"></output></label><div class="rd-scale"><input type="range" id="net-shrink" min="0" max="1" step="0.01" value="0"></div></div>`);
  controls(rail, ['topo', 'K', 'sizeMode', 'hops', 'clusters', 'pIn', 'pER', 'graphSeed', 'pFail', 'T', 'gammaTol'], 'net', 'Graph');
  controls(rail, ['tau2', 'eps', 'noDP', 'seed'], 'net', 'Replicate');

  const phaseEl = add(stage, '<div class="rd-phase" id="net-phase"></div>');
  const chipEl = add(stage, '<div class="rd-chips" id="net-chips"></div>');
  const wrap = add(stage, '<figure class="rd-fig"><div class="rd-canvas-wrap rd-network-wrap"><canvas class="rd-canvas" id="net-canvas" data-h="420" tabindex="0" aria-label="Hospital network. Arrow keys move between hospitals, Enter selects one."></canvas></div><figcaption><span class="rd-fignum">Fig. N1</span> <span id="net-cap"></span></figcaption></figure>');
  const grid = add(stage, '<div class="rd-grid2"></div>');
  const figCons = new Figure(add(grid, '<div></div>'), { id: 'fig-consensus', num: 'N2', caption: '', xLabel: 'gossip round t', yLabel: 'relative consensus error', logY: true, height: 240 });
  const figStrip = new Figure(add(grid, '<div></div>'), { id: 'net-strip', num: 'N3', caption: 'Shrinkage strip, one row per hospital (labels n<sub>k</sub>): local estimate (grey), network estimate (blue: μ<sub>−k</sub> for GHB, μ̃<sub>k</sub> for GHB-lite), current estimate (teal dot) and true θ<sub>k</sub> (black tick). Move the "Shrinkage applied" slider to scrub Proposition 4.1.', xLabel: '', yLabel: '', height: 300, canLogX: false, canLogY: false, toggles: false });

  const noteEl = explain(nb, 'What is happening', ['']);
  const eqEl = add(nb, '<div></div>');
  const insp = add(nb, '<div class="rd-inspector" id="net-inspector"></div>');

  const view = new NetworkView(wrap.querySelector('canvas'), { onSelect: (k) => { sel = k; refreshSide(); } });
  let ctx, rr, gp, phase = 0, shrinkT = 0, sel = 0, playing = false, busy = false, proto = 'ghb', x0 = null, sdShare = null;
  const speed = document.getElementById('net-speed'), shrink = document.getElementById('net-shrink');
  const setOut = () => { document.getElementById('net-speed-o').innerHTML = `${speed.value}<span class="u">rounds/s</span>`; document.getElementById('net-shrink-o').innerHTML = `${(100 * shrinkT).toFixed(0)}<span class="u">%</span>`; shrink.value = shrinkT; };
  const est = () => (proto === 'ghb' ? rr.est.ghb : rr.est.lite), netMu = () => (proto === 'ghb' ? rr.j.mu : rr.b.mu);

  function hospital(k) {
    return {
      k, n: ctx.n[k], v: ctx.v[k], s: Math.sqrt(ctx.s2[k]), x: rr.d.x[k], th: rr.d.th[k], theta: rr.d.theta[k], mu: rr.b.mu[k], M: rr.b.M[k], t2: rr.b.t2[k], alpha: rr.alpha[k], est: rr.est.lite[k], B: ctx.Bth[k], bad: !!ctx.isBad[k],
      muJ: rr.j.mu[k], MJ: rr.j.M[k], alphaJ: rr.j.alpha[k], wSafe: rr.j.wSafe[k], t2J: rr.j.t2[k], estJ: rr.est.ghb[k], BJ: ctx.BthJ[k], sJ: ctx.sJ
    };
  }
  function relOf(k) { const res = mainRun.latest(); return res && res.ctx.K === ctx.K ? res.summary.agg[proto === 'ghb' ? 'ghb' : 'lite'].y[k] : null; }
  function refreshSide() {
    if (!ctx) return;
    const h = hospital(sel);
    renderInspector(insp, h, relOf(sel), proto);
    const c = store.get();
    eqEl.innerHTML = (proto === 'ghb' ? EQ.jointM(ctx.K, ctx.tau2m, c.sigma2, ctx.sJ, h.n, h.MJ) + EQ.tauMixture() : EQ.mechanism(h.n, c.C, ctx.priv.delta, ctx.priv.eps, h.s))
      + EQ.gossip(ctx.lambda, c.T, c.gammaTol, ctx.rounds) + EQ.alpha(proto === 'ghb' ? h.t2J : h.t2, proto === 'ghb' ? h.MJ : h.M, h.v, proto === 'ghb' ? h.alphaJ : h.alpha, sel + 1);
    drawStrip();
  }
  function setPhase(p) {
    phase = p;
    phaseEl.querySelectorAll('span').forEach((s) => s.classList.toggle('on', +s.dataset.i === Math.min(p, 3)));
    noteEl.querySelector('p').innerHTML = NOTES[proto][Math.min(p, 3)];
    view.label = p === 2 || (p === 1 && gp.t > 0) ? `round ${gp.t} / ${store.get().T}` : PHASES[proto][Math.min(p, 3)].toLowerCase();
    view.draw();
  }
  function drawStrip() {
    figStrip.setCustom((c, w, h) => drawShrinkStrip(c, w, h, { n: ctx.n, th: rr.d.th, mu: netMu(), est: est(), theta: rr.d.theta, alpha: proto === 'ghb' ? rr.j.alpha : rr.alpha }, shrinkT, sel));
  }
  function drawCons() {
    const c = store.get(), tmax = Math.max(c.T, gp.t, 10);
    figCons.setData({
      xName: 'round t', yName: 'error', xLabel: 'gossip round t (rounds)', yLabel: '‖xᵗ − x̄1‖ / ‖x⁰ − x̄1‖',
      series: [{ id: 'bound', label: 'bound λᵗ (Prop. 4.3)', tok: '--rd-muted', dash: true, width: 1.2, pts: Array.from({ length: tmax + 1 }, (_, t) => [t, Math.max(ctx.lambda ** t, 1e-12)]) },
        { id: 'obs', label: proto === 'ghb' ? 'masked shares, this replicate' : 'releases, this replicate', tok: proto === 'ghb' ? '--rd-ghb' : '--rd-lite', width: 2, pts: gp.errors }],
      hlines: [{ y: c.gammaTol, tok: '--rd-noise', dash: true, label: `tolerance γ = ${fmt(c.gammaTol, 3)}` }]
    });
    figCons.setCaption(`Consensus error of the ${proto === 'ghb' ? 'masked shares (GHB)' : 'releases (GHB-lite)'} against gossip rounds (log scale, dimensionless), with the bound λ<sup>t</sup>, λ = ${fmt(ctx.lambda, 3)}. ${c.pFail > 0 ? `Each link fails with probability ${c.pFail} per round; weights are recomputed on working links, so the bound is a reference only.` : ''}`);
  }
  /** Masked shares of the network sum, scaled by K/N to the parameter scale; their mean is S~/N. @see Section 5.2 */
  function maskedShares(c) {
    const r = makeRng(31 + c.seed), K = ctx.K, N = ctx.n.reduce((a, b) => a + b, 0), sd = Math.sqrt(rr.d.th.reduce((a, b) => a + b * b, 0) / K) + 0.05;
    const mask = new Float64Array(K);
    for (const [i, j] of ctx.edges) { const m = 2.5 * sd * N / K * r.n(); mask[i] += m; mask[j] -= m; }
    // the noise shares sum exactly to the engine's draw xi, so the consensus is the S~ used for the estimates
    const xi = rr.j.S - ctx.n.reduce((a, nk, k) => a + nk * (ctx.isBad[k] ? rr.d.x[k] : rr.d.th[k]), 0);
    const share = Float64Array.from({ length: K }, () => ctx.sJ / Math.sqrt(K) * r.n()), mean = share.reduce((a, b) => a + b, 0) / K;
    for (let k = 0; k < K; k++) share[k] += xi / K - mean;
    sdShare = Float64Array.from({ length: K }, () => ctx.sJ / Math.sqrt(K) * K / N);
    return Float64Array.from({ length: K }, (_, k) => (ctx.n[k] * (ctx.isBad[k] ? rr.d.x[k] : rr.d.th[k]) + mask[k] + share[k]) * K / N);
  }
  function rebuild() {
    stop();
    const c = store.get();
    proto = c.protocol === 'lite' ? 'lite' : 'ghb';
    phaseEl.innerHTML = PHASES[proto].map((p, i) => `<span data-i="${i}">${i + 1}. ${p}</span>`).join('');
    ctx = prepare(c); rr = replicate(ctx, makeRng(c.seed * 104729 + 1), 0);
    x0 = proto === 'ghb' ? maskedShares(c) : rr.d.x;
    gp = createGossip(ctx, x0, c.pFail, 99 + c.seed);
    if (sel >= ctx.K) sel = 0;
    const mean = rr.d.th.reduce((a, b) => a + b, 0) / ctx.K, sd = Math.sqrt(rr.d.th.reduce((a, b) => a + (b - mean) ** 2, 0) / ctx.K) || 0.1;
    view.setGraph({ K: ctx.K, edges: ctx.edges, sizes: ctx.n, bad: ctx.bad });
    view.failed = new Set(); view.particles = []; view.select(sel);
    view.setValues(rr.d.th, mean, 2.2 * sd);
    shrinkT = 0; setOut(); setPhase(0);
    document.getElementById('net-cap').innerHTML = `${ctx.K} hospitals playing ${proto === 'ghb' ? 'GHB (joint release by secure summation over the graph)' : 'GHB-lite (per-hospital releases)'}; disc area ∝ n<sub>k</sub>, labels are n<sub>k</sub>; colour is the value held (blue below, amber above the average ${fmt(mean, 3)}). A red outline marks a dishonest hospital.`;
    const N = ctx.n.reduce((a, b) => a + b, 0);
    chips(chipEl, [['SLEM λ', ctx.connected ? fmt(ctx.lambda, 3) : '1', ctx.connected ? 'ref' : 'risk'], ['Spectral gap', fmt(1 - ctx.lambda, 3)], ['Rounds to γ', ctx.connected ? Math.ceil(ctx.rounds) : '∞'],
      ['λ^T at T = ' + c.T, fmt(ctx.lambda ** c.T, 3)], ['Agreements', ctx.edges.length],
      proto === 'ghb' ? ['Noise sd of S̃/N', fmt(ctx.sJ / N, 4), 'noise'] : ['Mean noise s/√v', fmt(Math.sqrt(ctx.s2.reduce((a, s, k) => a + s / ctx.v[k], 0) / ctx.K), 2), 'noise'],
      ['Patients N', N]]);
    drawCons(); refreshSide();
  }
  const dur = () => 1000 / +speed.value;
  async function step() {
    if (busy) return; busy = true;
    const c = store.get();
    try {
      if (phase === 0) { await view.animateNoise(rr.d.th, x0, proto === 'ghb' ? Array.from(sdShare, (v, k) => v + Math.abs(x0[k] - rr.d.th[k]) * 0.2) : Array.from(ctx.s2, Math.sqrt), Math.max(500, 4 * dur())); setPhase(1); }
      else if ((phase === 1 || phase === 2) && gp.t < c.T) { const r = gp.step(); setPhase(2); drawCons(); await view.animateGossip(r.prev, r.next, r.failed, dur()); }
      else if (phase === 1 || phase === 2) { setPhase(3); view.failed = new Set(); await view.animateShrink(rr.d.th, est(), Math.max(600, 6 * dur()), (t) => { shrinkT = t; setOut(); drawStrip(); }); setPhase(4); }
      else { rebuild(); }
    } finally { busy = false; }
  }
  async function play() { if (phase >= 4) rebuild(); playing = true; setBtn(); while (playing && phase < 4) { await step(); } playing = false; setBtn(); }
  function stop() { playing = false; view.anim?.cancel(); setBtn(); }
  const setBtn = () => { const b = document.getElementById('net-play'); b.textContent = playing ? 'Pause' : 'Play'; b.setAttribute('aria-pressed', String(playing)); };
  function toEnd() {
    stop(); const c = store.get();
    while (gp.t < c.T) gp.step();
    view.particles = []; setPhase(3); shrinkT = 1; view.setValues(est()); setOut(); drawCons(); drawStrip(); setPhase(4);
  }
  document.getElementById('net-play').onclick = () => (playing ? stop() : play());
  document.getElementById('net-step').onclick = () => { stop(); step(); };
  document.getElementById('net-end').onclick = toEnd;
  document.getElementById('net-reset').onclick = rebuild;
  speed.addEventListener('input', setOut);
  shrink.addEventListener('input', () => { stop(); shrinkT = +shrink.value; if (phase < 3) { while (gp.t < store.get().T) gp.step(); drawCons(); } setPhase(3); view.particles = []; const e = est(); view.setValues(Float64Array.from(rr.d.th, (t, k) => t + shrinkT * (e[k] - t))); setOut(); drawStrip(); });

  let timer;
  store.subscribe(() => { applyRelevance(document.getElementById('p-network'), store.get()); clearTimeout(timer); timer = setTimeout(rebuild, 200); });
  mainRun.onResult(() => { if (ctx) renderInspector(insp, hospital(sel), relOf(sel), proto); });
  applyRelevance(document.getElementById('p-network'), store.get());
  rebuild();
  return {
    action(a) {
      if (a === 'noise') { rebuild(); step(); }
      else if (a === 'gossip') { rebuild(); view.setValues(x0); setPhase(1); play(); }
      else if (a === 'shrink') { stop(); const c = store.get(); while (gp.t < c.T) gp.step(); drawCons(); phase = 2; step(); }
    }
  };
}
