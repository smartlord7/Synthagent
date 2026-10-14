/**
 * ui/simpleNetwork.js
 * Responsibility: the animated, interactive hospital network at the top of Simple mode. A guided story in seven
 * steps (Play / Pause / Step back / Step forward and a step indicator) shows one illustrative replicate of the
 * current setting: own estimates with their haze, privacy noise, scrambled summaries gossiped along the agreements,
 * the combined picture, each hospital's trust in it, and the result of the full Monte Carlo run. Hospitals can be
 * tapped (a card tells their story), dragged and focused with the keyboard; links explain themselves on hover; one
 * hospital can be made very different to show that it then trusts the network less.
 * Every number and movement comes from sim/simpleStory.js (the engine's own building blocks) and, for the result
 * step, from the Monte Carlo digest of sim/simple.js. Drawn in SVG (focusable nodes, hover titles, dragging) with
 * the theme tokens of viz/theme.js; prefers-reduced-motion turns every tween into a jump.
 */
import { tok, mix, alpha, diverging, onThemeChange, reducedMotion } from '../viz/theme.js';
import { STORY_STEPS, ROUNDS_SHOWN, storyData, stepCaption, nodeRadius, noiseLevel } from '../sim/simpleStory.js';

const NS = 'http://www.w3.org/2000/svg';
const ANIM = { meet: 1200, own: 1400, noise: 1600, travel: ROUNDS_SHOWN * 900, combine: 1600, trust: 2600, result: 1000 };
const DWELL = 5200, SEEN_KEY = 'ghb-lab-story-seen';
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
const pc = (x) => Math.round(100 * x);
const el = (name, attrs = {}) => { const e = document.createElementNS(NS, name); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** One-line legend of the drawing at each step. */
export const LEGENDS = {
  meet: 'Circle size: number of patients. Lines: agreements between two hospitals.',
  own: 'Colour: the hospital\'s own estimate (blue lower, orange higher). Grey haze: how unsure it is.',
  noise: 'Orange dots: the random privacy noise each hospital adds. More dots, spread wider, means more noise.',
  travel: 'Envelopes: scrambled summaries travelling between neighbours. Nobody can read what is inside.',
  combine: 'Blue dot: the combined picture every hospital now holds.',
  trust: 'Blue arc and number: how much each hospital trusts the network. Colour: its estimate as it moves.',
  result: 'Teal: better than working alone (darker is better). Red: worse. Number: error, with working alone = 100.'
};

/**
 * Build the animated network.
 * @param {HTMLElement} root container (the network card)
 * @param {{choices:()=>Object, result:()=>Object|null, requestRun:()=>void}} host Simple mode page
 * @returns {{update:()=>void, setResult:()=>void, onShow:()=>void}}
 */
export function initSimpleNetwork(root, host) {
  root.innerHTML = `
    <h2 id="sm-net-h"><span class="sm-num" aria-hidden="true">1</span>See how it works</h2>
    <div class="sm-player">
      <ol class="sm-dots" aria-label="Steps">${STORY_STEPS.map((s, i) => `<li><button type="button" data-step="${i}" aria-label="Step ${i + 1}: ${esc(s.title)}"><span aria-hidden="true">${i + 1}</span></button></li>`).join('')}</ol>
      <div class="sm-pbtns">
        <button class="rd-btn" type="button" id="sm-back" aria-label="Step back">&#9664; Back</button>
        <button class="rd-btn primary" type="button" id="sm-play" aria-pressed="false">Play</button>
        <button class="rd-btn" type="button" id="sm-fwd" aria-label="Step forward">Next &#9654;</button>
      </div>
    </div>
    <div class="sm-caption" aria-live="polite"><p class="sm-step-no" id="sm-step-no"></p><h3 id="sm-step-title"></h3><div id="sm-step-text"></div></div>
    <div class="sm-net-wrap"><svg class="sm-net" id="sm-net" role="group" aria-label="Hospital network: each circle is a hospital, each line an agreement">
      <g class="links"></g><g class="under" aria-hidden="true"></g><g class="packets" aria-hidden="true"></g><g class="nodes"></g><g class="over" aria-hidden="true"></g>
      <text class="sm-round" aria-hidden="true"></text>
    </svg></div>
    <p class="sm-legend-line" id="sm-legend-line"></p>
    <p class="sm-hover" id="sm-hover" aria-live="polite">Tap or click a hospital to hear its story. Drag hospitals to move them.</p>
    <div class="sm-node-card" id="sm-card" hidden></div>
    <div class="sm-atyp">
      <label for="sm-atyp-k">Make</label><select id="sm-atyp-k"></select>
      <button class="rd-btn" type="button" id="sm-atyp" aria-pressed="false">very different</button>
      <span class="sm-help">Shows that a hospital unlike the others trusts the network less.</span>
    </div>
    <figure class="sm-strip-fig"><figcaption class="sm-help" id="sm-strip-cap"></figcaption><svg class="sm-strip" id="sm-strip" aria-hidden="true"></svg></figure>`;

  const svg = root.querySelector('#sm-net'), gLinks = svg.querySelector('.links'), gUnder = svg.querySelector('.under'), gPackets = svg.querySelector('.packets');
  const gNodes = svg.querySelector('.nodes'), gOver = svg.querySelector('.over'), roundTxt = svg.querySelector('.sm-round');
  const strip = root.querySelector('#sm-strip'), stripCap = root.querySelector('#sm-strip-cap'), hoverEl = root.querySelector('#sm-hover'), card = root.querySelector('#sm-card');
  const playBtn = root.querySelector('#sm-play'), atypSel = root.querySelector('#sm-atyp-k'), atypBtn = root.querySelector('#sm-atyp');
  const legendEl = root.querySelector('#sm-legend-line'), stepNo = root.querySelector('#sm-step-no'), stepTitle = root.querySelector('#sm-step-title'), stepText = root.querySelector('#sm-step-text');

  let D = null, Dtyp = null, atyp = -1, step = 0, t = 1, playing = false, stepStart = 0, raf = 0, selected = -1, W = 600, H = 460;
  let pos = [], dragged = new Map(), nodeEls = [], linkEls = [], T = {}, shown = false, lastK = 0;

  /* ---------- data ---------- */
  function compute() {
    const ch = host.choices();
    Dtyp = storyData(ch);
    if (atyp >= Dtyp.K) setAtyp(-1, false);
    D = atyp >= 0 ? storyData(ch, { atypical: atyp }) : Dtyp;
    if (D.K !== lastK) { dragged = new Map(); lastK = D.K; selected = -1; card.hidden = true; buildGraph(); fillAtypSelect(); }
    else buildLinks();
  }
  function fillAtypSelect() {
    const keep = Number(atypSel.value) || 0;
    atypSel.innerHTML = D.n.map((n, k) => `<option value="${k}">Hospital ${k + 1}: ${n} patients</option>`).join('');
    atypSel.value = String(Math.min(keep, D.K - 1));
  }

  /* ---------- layout ---------- */
  function layout() {
    const wrap = svg.parentElement; W = Math.max(280, Math.round(wrap.clientWidth || 600));
    H = Math.round(Math.max(300, Math.min(560, W * (W < 560 ? 1.02 : 0.72))));
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.setAttribute('height', H);
    const K = D.K, margin = longLabels() ? 70 : 70, rad = Math.min(W, H) / 2 - margin, cx = W / 2, cy = H / 2 + 6;
    const room = Math.min(W < 560 ? 19 : 32, (rad * Math.PI / K) * 0.85);
    pos = D.n.map((_, k) => {
      if (dragged.has(k)) { const [fx, fy] = dragged.get(k); return { x: fx * W, y: fy * H }; }
      const a = (2 * Math.PI * k) / K - Math.PI / 2; return { x: cx + rad * Math.cos(a), y: cy + rad * Math.sin(a) };
    });
    pos.forEach((p, k) => { p.r = nodeRadius(D.n, k, room); const dx = p.x - cx, dy = p.y - cy, d = Math.hypot(dx, dy) || 1; p.ux = dx / d; p.uy = dy / d; });
    roundTxt.setAttribute('x', 12); roundTxt.setAttribute('y', 20);
  }
  const longLabels = () => W >= 560 && D.K <= 14;

  /* ---------- static parts: links and nodes ---------- */
  function buildLinks() {
    gLinks.innerHTML = ''; linkEls = [];
    for (const [i, j] of D.edges) {
      const g = el('g', { class: 'sm-link' }), hit = el('line', { class: 'hit' }), line = el('line', { class: 'line' }), title = el('title');
      title.textContent = `An agreement between Hospital ${i + 1} and Hospital ${j + 1}`;
      g.append(title, line, hit); gLinks.appendChild(g); linkEls.push({ i, j, g, line, hit });
      const say = () => { hoverEl.textContent = `An agreement between Hospital ${i + 1} (${D.n[i]} patients) and Hospital ${j + 1} (${D.n[j]} patients): they exchange scrambled summaries, never patient records.`; g.classList.add('on'); };
      g.addEventListener('pointerenter', say); g.addEventListener('click', say);
      g.addEventListener('pointerleave', () => g.classList.remove('on'));
    }
  }
  function buildGraph() {
    buildLinks();
    gNodes.innerHTML = ''; nodeEls = [];
    D.n.forEach((n, k) => {
      const g = el('g', { class: 'sm-node', tabindex: '0', role: 'button', 'aria-label': `Hospital ${k + 1}: ${n} patients` });
      const ring = el('circle', { class: 'ring' }), body = el('circle', { class: 'body' }), val = el('text', { class: 'val' }), lab = el('text', { class: 'lab' });
      g.append(ring, body, val, lab); gNodes.appendChild(g); nodeEls.push({ g, ring, body, val, lab });
      let start = null, moved = false;
      g.addEventListener('pointerdown', (e) => { start = { x: e.clientX, y: e.clientY }; moved = false; g.setPointerCapture(e.pointerId); });
      g.addEventListener('pointermove', (e) => {
        if (!start) return;
        if (!moved && Math.hypot(e.clientX - start.x, e.clientY - start.y) < 5) return;
        moved = true; const r = svg.getBoundingClientRect(), sx = W / r.width, sy = H / r.height;
        const x = Math.max(10, Math.min(W - 10, (e.clientX - r.left) * sx)), y = Math.max(10, Math.min(H - 10, (e.clientY - r.top) * sy));
        dragged.set(k, [x / W, y / H]); draw();
      });
      g.addEventListener('pointerup', () => { if (start && !moved) openCard(k); start = null; });
      g.addEventListener('pointercancel', () => { start = null; });
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openCard(k); } });
      g.addEventListener('pointerenter', () => { hoverEl.textContent = `Hospital ${k + 1}: ${n} patients. Tap or click for its story.`; });
    });
  }

  /* ---------- the hospital card ---------- */
  function openCard(k) {
    selected = k; const res = host.result(), h = res && res.hospitals.length === D.K ? res.hospitals[k] : null;
    const typNote = atyp === k ? ` It is now very different from the others; when it was typical it trusted the network ${pc(Dtyp.trust[k])}%.` : '';
    card.innerHTML = `<button class="rd-btn sm-close" type="button" aria-label="Close">Close</button>
      <h3>Hospital ${k + 1}: ${D.n[k]} patients</h3>
      <p>It looks only at its own ${D.n[k]} patients. ${D.n[k] < 60 ? 'That is few, so its own estimate is quite unsure.' : D.n[k] < 250 ? 'Its own estimate is fairly reliable.' : 'That is many, so its own estimate is already reliable.'}</p>
      <p>In this example it trusted the network <strong>${pc(D.trust[k])}%</strong> and kept ${pc(1 - D.trust[k])}% of its own estimate.${typNote}</p>
      ${h ? `<p>Over the full simulation: error alone <strong>100</strong>, with GHB <strong class="${h.ghb > 1 ? 'lv-bad' : 'lv-good'}">${pc(h.ghb)}</strong>, like having <strong>${Math.round(h.likePatients)}</strong> patients instead of ${D.n[k]}.${h.ghb > 1 ? ' It is slightly worse off than working alone here.' : ''}${atyp >= 0 ? ' (These numbers are for the scenario without the very different hospital.)' : ''}</p>` : '<p>Press <strong>Run</strong> below to see its error over the full simulation.</p>'}
      <p><button class="rd-btn" type="button" id="sm-card-atyp">${atyp === k ? 'Make it typical again' : 'Make this hospital very different'}</button></p>`;
    card.hidden = false;
    card.querySelector('.sm-close').onclick = () => { card.hidden = true; selected = -1; draw(); nodeEls[k]?.g.focus(); };
    card.querySelector('#sm-card-atyp').onclick = () => { atypSel.value = String(k); setAtyp(atyp === k ? -1 : k); };
    draw();
  }

  /* ---------- atypical hospital ---------- */
  function setAtyp(k, recompute = true) {
    atyp = k; atypBtn.setAttribute('aria-pressed', String(k >= 0));
    atypBtn.textContent = k >= 0 ? 'Make it typical again' : 'very different';
    if (!recompute) return;
    compute(); go(STORY_STEPS.findIndex((s) => s.id === 'trust'), false);
    if (selected >= 0) openCard(selected);
  }
  atypBtn.onclick = () => setAtyp(atyp >= 0 ? -1 : Number(atypSel.value));
  atypSel.onchange = () => { if (atyp >= 0) setAtyp(Number(atypSel.value)); };

  /* ---------- player ---------- */
  function go(i, keepPlaying = playing) {
    step = Math.max(0, Math.min(STORY_STEPS.length - 1, i)); t = reducedMotion ? 1 : 0; stepStart = performance.now();
    playing = keepPlaying; syncPlay();
    if (STORY_STEPS[step].id === 'result' && !host.result()) host.requestRun();
    caption(); kick();
  }
  function syncPlay() {
    playBtn.textContent = playing ? 'Pause' : step === STORY_STEPS.length - 1 ? 'Replay' : 'Play';
    playBtn.setAttribute('aria-pressed', String(playing));
    root.querySelectorAll('.sm-dots button').forEach((b, i) => { if (i === step) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); b.classList.toggle('done', i < step); });
  }
  function caption() {
    const s = STORY_STEPS[step];
    stepNo.textContent = `Step ${step + 1} of ${STORY_STEPS.length}`;
    stepTitle.textContent = s.title; legendEl.textContent = LEGENDS[s.id];
    stepText.innerHTML = stepCaption(s.id, D, host.result(), atyp >= 0 ? Dtyp : null).map((x) => `<p>${esc(x)}</p>`).join('');
    const sid = s.id;
    stripCap.textContent = sid === 'meet' ? '' : sid === 'result'
      ? 'Each row is a hospital. Grey: its own estimate, with its haze. Blue: what the others have seen. Teal: its final estimate. Black: the truth, which only the simulation knows.'
      : `Each row is a hospital. Grey: its own estimate, with its haze.${['combine', 'trust'].includes(sid) ? ' Blue: what the others have seen.' : ''}${sid === 'trust' ? ' Teal: the estimate moving toward the network, as far as the hospital trusts it.' : ''}`;
  }
  playBtn.onclick = () => {
    if (playing) { playing = false; syncPlay(); return; }
    if (step === STORY_STEPS.length - 1) go(0, true); else { playing = true; stepStart = performance.now() - (t >= 1 ? ANIM[STORY_STEPS[step].id] : 0); syncPlay(); kick(); }
  };
  root.querySelector('#sm-back').onclick = () => go(step - 1, false);
  root.querySelector('#sm-fwd').onclick = () => go(step + 1, false);
  root.querySelectorAll('.sm-dots button').forEach((b) => { b.onclick = () => go(Number(b.dataset.step), false); });

  /* ---------- animation loop ---------- */
  const continuous = () => !reducedMotion && ['noise', 'travel'].includes(STORY_STEPS[step].id);
  function kick() { if (!raf) raf = requestAnimationFrame(tick); }
  function tick(now) {
    raf = 0;
    if (document.documentElement.dataset.mode !== 'simple') return;
    const id = STORY_STEPS[step].id, el0 = now - stepStart;
    t = reducedMotion ? 1 : Math.min(1, el0 / ANIM[id]);
    draw(now);
    if (playing && el0 >= ANIM[id] + DWELL) {
      if (step < STORY_STEPS.length - 1) { go(step + 1, true); return; }
      playing = false; syncPlay();
    }
    if (t < 1 || continuous() || playing) kick();
  }

  /* ---------- drawing ---------- */
  function readTokens() {
    T = { ink: tok('--rd-ink'), muted: tok('--rd-muted'), rule: tok('--rd-rule'), node: tok('--rd-node'), ghb: tok('--rd-ghb'), noise: tok('--rd-noise'), ref: tok('--rd-ref'),
      risk: tok('--rd-risk'), surface: tok('--rd-surface'), local: tok('--rd-local'), focus: tok('--rd-focus'), paper: tok('--rd-paper') };
  }
  const val = (v) => diverging((v - D.centre) / D.extent);
  function draw(now = performance.now()) {
    if (!D) return;
    layout();
    const id = STORY_STEPS[step].id, si = step, e = ease(t), K = D.K, res = host.result(), resOk = res && res.hospitals.length === K;
    const hazeMax = Math.max(...D.sd), lvl = noiseLevel(D.noiseSd), time = reducedMotion ? 0 : now / 1000;
    // links
    for (const L of linkEls) {
      const a = pos[L.i], b = pos[L.j];
      for (const ln of [L.line, L.hit]) { ln.setAttribute('x1', a.x); ln.setAttribute('y1', a.y); ln.setAttribute('x2', b.x); ln.setAttribute('y2', b.y); }
      const len = Math.hypot(b.x - a.x, b.y - a.y), grow = si === 0 ? e : 1;
      L.line.setAttribute('stroke-dasharray', `${len * grow} ${len}`);
    }
    // haze (own estimate uncertainty) under the nodes
    let under = '';
    if (si >= 1) for (let k = 0; k < K; k++) {
      const p = pos[k], hz = p.r + 4 + 26 * (D.sd[k] / hazeMax), a = (si === 1 ? e : 1) * (si >= 5 ? 0.35 : 0.6);
      under += `<circle cx="${p.x}" cy="${p.y}" r="${hz}" fill="url(#sm-haze)" opacity="${a}"/>`;
    }
    gUnder.innerHTML = `<defs><radialGradient id="sm-haze"><stop offset="0.35" stop-color="${T.local}" stop-opacity="0.55"/><stop offset="1" stop-color="${T.local}" stop-opacity="0"/></radialGradient></defs>${under}`;
    // noise dots and travelling packets
    let pk = '';
    if (si === 2 || si === 3) {
      const shown = si === 2 ? e : 1 - e * 0.6;
      for (let k = 0; k < K; k++) {
        const p = pos[k], cnt = 4 + Math.round(10 * lvl), spread = p.r + 4 + 30 * lvl;
        for (let q = 0; q < cnt; q++) {
          const ang = (q / cnt) * 2 * Math.PI + time * (0.6 + (q % 3) * 0.3) + k, rr = spread * (0.75 + 0.25 * Math.sin(time * 3 + q * 1.7 + k));
          pk += `<circle cx="${p.x + rr * Math.cos(ang)}" cy="${p.y + rr * Math.sin(ang)}" r="${1.6 + 1.4 * lvl}" fill="${T.noise}" opacity="${0.85 * shown}"/>`;
        }
      }
    }
    let round = '';
    if (si === 3) {
      const rf = t * ROUNDS_SHOWN, r = Math.min(ROUNDS_SHOWN - 1, Math.floor(rf)), f = reducedMotion ? 0.5 : ease(rf - r);
      round = `Round ${r + 1} of ${D.rounds}`; const es = W < 560 ? 0.75 : 1;
      for (const [i, j] of D.edges) for (const [a, b] of [[i, j], [j, i]]) {
        const A = pos[a], B = pos[b], x = A.x + (B.x - A.x) * f, y = A.y + (B.y - A.y) * f;
        pk += `<g transform="translate(${x - 6 * es} ${y - 4 * es}) scale(${es})"><rect width="12" height="8" rx="1" fill="${T.surface}" stroke="${T.noise}" stroke-width="1.2"/><path d="M0 0 6 5 12 0" fill="none" stroke="${T.noise}" stroke-width="1"/></g>`;
      }
    } else if (si === 4) round = `After ${D.rounds} rounds`;
    gPackets.innerHTML = pk; roundTxt.textContent = round;
    // nodes
    let over = '';
    for (let k = 0; k < K; k++) {
      const p = pos[k], N = nodeEls[k];
      let fill = T.node, stroke = T.ink, text = '';
      if (si === 1) fill = mix(T.node, val(D.own[k]), e);
      else if (si === 2) fill = val(D.own[k]);
      else if (si === 3) { fill = mix(val(D.own[k]), T.node, Math.min(1, e * 2)); text = '?'; }
      else if (si === 4) fill = mix(T.node, val(D.network[k]), e);
      else if (si === 5) fill = val(D.own[k] + (D.ghb[k] - D.own[k]) * e);
      else if (si === 6) {
        if (resOk) { const g = res.hospitals[k].ghb; fill = g <= 1 ? mix(T.node, T.ghb, Math.min(1, (1 - g) / 0.45) * e) : mix(T.node, T.risk, Math.min(1, (g - 1) / 0.15 + 0.4) * e); text = String(pc(g)); }
        else fill = T.node;
      }
      N.body.setAttribute('cx', p.x); N.body.setAttribute('cy', p.y); N.body.setAttribute('r', p.r);
      N.body.setAttribute('fill', fill); N.body.setAttribute('stroke', k === atyp ? T.risk : stroke); N.body.setAttribute('stroke-width', k === atyp ? 2.5 : 1);
      N.body.setAttribute('stroke-dasharray', k === atyp ? '4 3' : 'none');
      N.ring.setAttribute('cx', p.x); N.ring.setAttribute('cy', p.y); N.ring.setAttribute('r', p.r + 5);
      N.ring.setAttribute('stroke', k === selected ? T.ink : 'transparent');
      N.val.setAttribute('x', p.x); N.val.setAttribute('y', p.y); N.val.textContent = p.r >= 11 ? text : '';
      N.val.setAttribute('fill', si === 6 && resOk ? T.paper : T.ink);
      // label outside the circle
      const lx = p.x + p.ux * (p.r + 7), ly = p.y + p.uy * (p.r + 9), anchor = p.ux > 0.35 ? 'start' : p.ux < -0.35 ? 'end' : 'middle';
      N.lab.setAttribute('x', lx); N.lab.setAttribute('y', ly); N.lab.setAttribute('text-anchor', anchor);
      N.lab.innerHTML = longLabels() ? `<tspan x="${lx}" dy="${p.uy < -0.35 ? '-0.75em' : p.uy > 0.35 ? '0.35em' : '-0.2em'}">Hospital ${k + 1}</tspan><tspan x="${lx}" dy="1.15em" class="n">${D.n[k]} patients</tspan>` : `H${k + 1}<tspan class="n"> ${D.n[k]}</tspan>`;
      // combined picture: blue inner mark; trust: arc showing the share given to the network
      if (si === 4) over += `<circle cx="${p.x}" cy="${p.y}" r="${Math.max(2.5, p.r * 0.35) * e}" fill="${T.ref}" stroke="${T.surface}" stroke-width="1"/>`;
      if (si === 5) {
        const share = D.trust[k] * e, rr = p.r + 3.5;
        if (share > 0.004) {
          const a1 = -Math.PI / 2, a2 = a1 + 2 * Math.PI * Math.min(share, 0.9999), big = share > 0.5 ? 1 : 0;
          over += `<path d="M${p.x + rr * Math.cos(a1)} ${p.y + rr * Math.sin(a1)} A${rr} ${rr} 0 ${big} 1 ${p.x + rr * Math.cos(a2)} ${p.y + rr * Math.sin(a2)}" fill="none" stroke="${T.ref}" stroke-width="3" stroke-linecap="round"/>`;
        }
        if (p.r >= 13 && t >= 1) N.val.textContent = pc(D.trust[k]) + '%';
      }
    }
    gOver.innerHTML = over;
    drawStrip(si, e, resOk ? res : null);
  }

  /* ---------- estimate strip: one row per hospital ---------- */
  function drawStrip(si, e, res) {
    const K = D.K;
    if (si === 0) { strip.innerHTML = ''; strip.setAttribute('height', 0); return; }
    const rowH = W < 560 ? 15 : 17, L = W < 560 ? 64 : 120, R = 14, top = 6, h = top + K * rowH + 22;
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < K; k++) for (const v of [D.own[k] - D.sd[k], D.own[k] + D.sd[k], D.network[k], D.theta[k]]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.04 || 0.1; lo -= pad; hi += pad;
    const X = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
    strip.setAttribute('viewBox', `0 0 ${W} ${h}`); strip.setAttribute('width', W); strip.setAttribute('height', h);
    let s = '';
    for (let k = 0; k < K; k++) {
      const y = top + (k + 0.5) * rowH, sel = k === selected || k === atyp;
      if (sel) s += `<rect x="0" y="${y - rowH / 2}" width="${W}" height="${rowH}" fill="${k === atyp ? alpha(T.risk, 0.12) : alpha(T.ink, 0.07)}"/>`;
      s += `<text x="${L - 8}" y="${y}" text-anchor="end" dominant-baseline="middle" class="lab">${W < 560 ? `H${k + 1} ${D.n[k]}` : `Hospital ${k + 1}: ${D.n[k]}`}</text>`;
      s += `<line x1="${X(D.own[k] - D.sd[k])}" x2="${X(D.own[k] + D.sd[k])}" y1="${y}" y2="${y}" stroke="${alpha(T.local, 0.4)}" stroke-width="${rowH * 0.5}" stroke-linecap="round"/>`;
      s += `<line x1="${X(D.own[k])}" x2="${X(D.own[k])}" y1="${y - rowH * 0.38}" y2="${y + rowH * 0.38}" stroke="${T.local}" stroke-width="2"/>`;
      if (si >= 4) s += `<line x1="${X(D.network[k])}" x2="${X(D.network[k])}" y1="${y - rowH * 0.38}" y2="${y + rowH * 0.38}" stroke="${T.ref}" stroke-width="2" opacity="${si === 4 ? e : 1}"/>`;
      if (si >= 5) { const cur = D.own[k] + (D.ghb[k] - D.own[k]) * (si === 5 ? e : 1); s += `<circle cx="${X(cur)}" cy="${y}" r="${Math.min(5, rowH * 0.32)}" fill="${T.ghb}"/>`; }
      if (si === 6) s += `<path d="M${X(D.theta[k])} ${y - rowH * 0.45}v${rowH * 0.9}" stroke="${T.ink}" stroke-width="1.5"/>`;
    }
    s += `<text x="${L}" y="${h - 6}" class="axis">lower</text><text x="${W - R}" y="${h - 6}" text-anchor="end" class="axis">higher</text>${W >= 560 ? `<text x="${(L + W - R) / 2}" y="${h - 6}" text-anchor="middle" class="axis">each hospital's estimate of the same quantity</text>` : ''}`;
    strip.innerHTML = s;
  }

  onThemeChange(() => { readTokens(); draw(); });
  new ResizeObserver(() => { if (D) draw(); }).observe(svg.parentElement);
  readTokens(); compute(); caption(); syncPlay(); draw();

  return {
    /** Settings changed: recompute the replicate and redraw at once (the noise follows the privacy setting). */
    update() { compute(); caption(); if (selected >= 0 && selected < D.K) openCard(selected); draw(); kick(); },
    /** The Monte Carlo result changed: refresh the result step and the card. */
    setResult() { caption(); if (selected >= 0) openCard(selected); draw(); },
    /** Simple mode became visible: autoplay once on the first visit. */
    onShow() {
      readTokens(); draw(); kick();
      if (shown) return; shown = true;
      let seen = null; try { seen = localStorage.getItem(SEEN_KEY); } catch (e) { seen = null; }
      if (!seen) { try { localStorage.setItem(SEEN_KEY, '1'); } catch (e) { /* storage unavailable */ } go(0, true); }
    }
  };
}
