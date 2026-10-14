/**
 * ui/simple.js
 * Responsibility: the Simple mode page. A short story, the animated network (ui/simpleNetwork.js), a guided setup of five plain controls, a Run button, results
 * per hospital in plain terms (a bar chart against "working alone" and sentences computed from the run), a
 * "What just happened?" section and a way to the expert view. The scenario mapping and the sentences live in
 * sim/simple.js (DOM-free and tested); this module only builds the page and runs the jobs.
 */
import { add } from './dom.js';
import { initSimpleNetwork } from './simpleNetwork.js';
import { runJob, cancelJob } from '../sim/scheduler.js';
import { runArena } from '../sim/arena.js';
import { CHOICES, HOSPITALS, STORY, STEPS, defaultChoices, simpleJob, simpleArenaSpec, simpleSizes, digestArena, describe } from '../sim/simple.js';

const pct = (x) => Math.round(100 * x);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/**
 * Build the Simple mode page.
 * @param {HTMLElement} root container
 * @param {{expert:()=>void}} act actions provided by main.js (switch to the expert view)
 * @returns {{stop:()=>void, onShow:()=>void}} stop cancels any running test; onShow starts the story (autoplay on a first visit)
 */
export function initSimple(root, act) {
  const ch = defaultChoices();
  const pills = (key) => {
    const g = CHOICES[key];
    return `<fieldset class="sm-field"><legend>${g.label}</legend><p class="sm-help" id="sm-help-${key}">${g.help}</p>
      <div class="sm-pills">${g.options.map((o) => `<label><input type="radio" name="${key}" value="${o.id}" ${o.id === ch[key] ? 'checked' : ''} aria-describedby="sm-help-${key}"><span>${o.label}</span></label>`).join('')}</div></fieldset>`;
  };
  const wrap = add(root, `<div class="sm-wrap">
    <section class="sm-story" aria-labelledby="sm-title">
      <h2 id="sm-title">Can small hospitals learn from each other without sharing patient records?</h2>
      <p>${STORY.join(' ')}</p>
    </section>
    <div class="sm-stage">
    <section class="sm-card sm-netcard" id="sm-netcard" aria-labelledby="sm-net-h"></section>
    <section class="sm-card sm-setup" aria-labelledby="sm-setup-h">
      <h2 id="sm-setup-h"><span class="sm-num" aria-hidden="true">2</span>Set up your scenario</h2>
      <p class="sm-help">The network follows these settings as you change them.</p>
      <form id="sm-form" novalidate>
        <div class="sm-field">
          <label class="sm-k" for="sm-K"><span>${HOSPITALS.label}</span><output id="sm-K-out" for="sm-K">${ch.K}</output></label>
          <p class="sm-help" id="sm-help-K">${HOSPITALS.help}</p>
          <div class="rd-scale"><input type="range" id="sm-K" name="K" min="${HOSPITALS.min}" max="${HOSPITALS.max}" step="1" value="${ch.K}" aria-describedby="sm-help-K"></div>
          <p class="sm-sizes" id="sm-sizes" aria-live="polite"></p>
        </div>
        ${pills('sizes')}
        ${pills('difference')}
        ${pills('privacy')}
        <fieldset class="sm-field"><legend>Also compare with</legend><p class="sm-help">Working alone is always shown. You can add two other options.</p>
          <label class="sm-check"><input type="checkbox" name="pooled" ${ch.pooled ? 'checked' : ''}><span><strong>Everyone's patients in one place.</strong> Not allowed in practice, because records would leave their hospital, but a useful yardstick.</span></label>
          <label class="sm-check"><input type="checkbox" name="federated" ${ch.federated ? 'checked' : ''}><span><strong>A standard federated method.</strong> A common approach today: hospitals train one shared model in rounds, each adding privacy noise, then each hospital adjusts it to its own patients. Runs as a second, slower test with a yes or no prediction model.</span></label>
        </fieldset>
        <div class="sm-runrow"><button class="rd-btn primary sm-run" type="submit" id="sm-run">Run</button><span class="rd-progress" id="sm-progress" aria-live="polite"></span></div>
      </form>
    </section>
    </div>
    <section class="sm-card" aria-labelledby="sm-res-h">
      <h2 id="sm-res-h"><span class="sm-num" aria-hidden="true">3</span>Results</h2>
      <div id="sm-results"><p class="sm-help">Press <strong>Run</strong> to see how each hospital does on its own and with GHB. It takes a second or two.</p></div>
    </section>
    <details class="sm-how">
      <summary>What just happened?</summary>
      <ol>${STEPS.map((s) => `<li>${s}</li>`).join('')}</ol>
      <p class="sm-help">The simulation repeats this thousands of times with made-up patients whose true answers are known, so it can measure each hospital's error exactly.</p>
    </details>
    <div class="sm-expert"><button class="rd-btn" type="button" id="sm-expert">Show me the expert view</button><span class="sm-help">Every parameter, the network animation, the comparison with every published method and the real-data results.</span></div>
  </div>`);

  const form = wrap.querySelector('#sm-form'), kIn = wrap.querySelector('#sm-K'), kOut = wrap.querySelector('#sm-K-out');
  const sizesEl = wrap.querySelector('#sm-sizes'), resEl = wrap.querySelector('#sm-results'), progEl = wrap.querySelector('#sm-progress');
  const runBtn = wrap.querySelector('#sm-run');
  let result = null, fed = null, fedState = 'off', worker = null, stale = false;

  function read() {
    const fd = new FormData(form);
    ch.K = Number(kIn.value); ch.sizes = fd.get('sizes'); ch.difference = fd.get('difference'); ch.privacy = fd.get('privacy');
    ch.pooled = form.elements.pooled.checked; ch.federated = form.elements.federated.checked;
  }
  function showSizes() {
    const n = simpleSizes(ch);
    sizesEl.textContent = `Patients per hospital: ${n.join(', ')}.`;
  }
  form.addEventListener('input', () => {
    read(); kOut.textContent = String(ch.K); showSizes();
    if (result && !stale) { stale = true; resEl.querySelector('.sm-stale')?.removeAttribute('hidden'); }
  });
  form.addEventListener('input', () => net.update());
  form.addEventListener('submit', (e) => { e.preventDefault(); run(true); });
  wrap.querySelector('#sm-expert').onclick = () => act.expert();
  showSizes();

  function stopFed() { if (worker) { worker.terminate(); worker = null; } cancelJob('simple-fed'); }
  function run(scroll = false) {
    read(); stopFed(); stale = false; fed = null; fedState = ch.federated ? 'waiting' : 'off';
    const snap = { ...ch };
    runBtn.disabled = true; runBtn.textContent = 'Running...'; resEl.setAttribute('aria-busy', 'true');
    progEl.innerHTML = bar(0, 'Simulating the hospitals');
    runJob('simple', withProgress(snap), {
      onDone: (r) => {
        result = r; runBtn.disabled = false; runBtn.textContent = 'Run again'; resEl.removeAttribute('aria-busy');
        progEl.textContent = 'Done.';
        render(); net.setResult();
        if (snap.federated) startFed(snap);
        if (scroll) resEl.closest('section').scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
      },
      onError: (e) => { runBtn.disabled = false; runBtn.textContent = 'Run'; resEl.removeAttribute('aria-busy'); progEl.innerHTML = `<span class="lv-bad">The run failed: ${esc(e.message)}</span>`; }
    });
  }
  // the main job yields its fraction done; each yield updates the progress bar
  function* withProgress(snap) {
    const g = simpleJob(snap); let r = g.next();
    while (!r.done) { progEl.innerHTML = bar(r.value, 'Simulating the hospitals'); yield; r = g.next(); }
    return r.value;
  }
  function bar(f, label) { return `<span class="bar"><i style="width:${Math.round(100 * f)}%"></i></span>${label}`; }

  function startFed(snap) {
    const spec = simpleArenaSpec(snap);
    fedState = 'running'; renderFed();
    const onDone = (res) => { worker?.terminate(); worker = null; fed = digestArena(res); fedState = 'done'; render(); };
    const onFail = (msg) => { worker?.terminate(); worker = null; fedState = 'failed'; renderFed(msg); };
    const onProgress = (p) => { const el = resEl.querySelector('#sm-fed-prog'); if (el) el.innerHTML = bar(p.done / p.total, `Second test: ${p.done} of ${p.total} repetitions`); };
    const fallback = () => {
      const gen = runArena(spec);
      runJob('simple-fed', (function* () { let r = gen.next(); while (!r.done) { onProgress(r.value); yield; r = gen.next(); } return r.value; })(), { onDone, onError: (e) => onFail(e.message) });
    };
    let got = false;
    try {
      worker = new Worker(new URL('../sim/arena.worker.js', import.meta.url), { type: 'module' });
      worker.onmessage = (e) => { got = true; const d = e.data; if (d.type === 'progress') onProgress(d); else if (d.type === 'done') onDone(d.result); else onFail(d.message); };
      worker.onerror = (e) => { e.preventDefault?.(); if (!got) { worker?.terminate(); worker = null; fallback(); } else onFail(e.message || 'worker error'); };
      worker.postMessage({ spec });
    } catch (e) { worker = null; fallback(); }
  }

  function render() {
    const r = result, H = r.hospitals, K = H.length, c = r.choices, d = describe(r, fed);
    const showPool = c.pooled;
    // the chart shows patients gained: each hospital is as accurate as if it had "like" patients (its patients divided by
    // its error compared with working alone), so it gains like - has. Diverging bars around 0, scaled to GHB's gains;
    // everyone pooled is a diamond (a triangle when it falls off the scale).
    const like = (h, rel) => h.n / rel, gainG = H.map((h) => like(h, h.ghb) - h.n), gainP = H.map((h) => like(h, h.pooled) - h.n);
    const span = Math.max(10, ...gainG.map(Math.abs)) * 1.15, lo = gainG.some((g) => g < 0) || showPool ? -span : 0, hi = span;
    const x = (g) => (100 * (Math.min(hi, Math.max(lo, g)) - lo)) / (hi - lo);
    const step = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000].find((t) => span / t <= 3) || 10000;
    const ticks = []; for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
    const better = H.filter((h) => h.ghb <= 1).length, s0 = H[0], total = gainG.reduce((q, g) => q + g, 0);
    const sg = (g) => { const r = Math.round(g); return r > 0 ? `+${r}` : `${r}`; };
    const rows = H.map((h, i) => {
      const g = gainG[i], gp = gainP[i], worse = g < 0, off = gp > hi || gp < lo;
      return `<tr>
        <th scope="row">Hospital ${h.k + 1}</th>
        <td class="num has">${h.n}</td>
        <td class="sm-track-cell" aria-hidden="true"><div class="sm-track">
          ${ticks.map((t) => `<span class="sm-grid${t === 0 ? ' one' : ''}" style="left:${x(t)}%"></span>`).join('')}
          <span class="sm-bar${worse ? ' worse' : ''}" style="left:${Math.min(x(0), x(g))}%;width:${Math.max(Math.abs(x(g) - x(0)), 0.6)}%"></span>
          ${showPool ? `<span class="sm-pool${off ? ' off' : ''}${gp < lo ? ' left' : ''}" style="left:${x(gp)}%" title="Everyone pooled: ${sg(gp)} patients"></span>` : ''}
        </div></td>
        <td class="num${worse ? ' worse' : ''}">${sg(g)}<small>like ${Math.round(h.n + g)}</small></td>
        ${showPool ? `<td class="num pool">${sg(gp)}<small>like ${Math.round(h.n + gp)}</small></td>` : ''}
      </tr>`;
    }).join('');
    resEl.innerHTML = `
      <p class="sm-stale rd-callout warn" ${stale ? '' : 'hidden'}>You changed the settings. Press <strong>Run again</strong> to update these results.</p>
      <div class="rd-chips sm-chips">
        <div class="rd-chip ghb"><div class="k">Hospitals better off with GHB</div><div class="v">${better} of ${K}</div></div>
        <div class="rd-chip ghb"><div class="k">Smallest hospital</div><div class="v">${s0.n} &rarr; ${Math.round(s0.likePatients)} <small>patients (${sg(s0.likePatients - s0.n)})</small></div></div>
        <div class="rd-chip ghb"><div class="k">Patients gained, all hospitals</div><div class="v">${sg(total)} <small>patients' worth</small></div></div>
      </div>
      <h3 class="sm-h3">What we found</h3>
      <ul class="sm-findings">${d.main.map((s) => `<li class="${s.kind}">${esc(s.text)}</li>`).join('')}</ul>
      <h3 class="sm-h3">How many patients each hospital gains</h3>
      <p class="sm-help">GHB never moves patient records, but it makes each hospital as accurate as if it had more patients. <strong>Has</strong> is the hospital's own patients; the bar and the number are the <strong>patients gained</strong>. For example, Hospital ${s0.k + 1} has ${s0.n} patients and gains ${sg(s0.likePatients - s0.n)}: it predicts as well as a hospital with ${Math.round(s0.likePatients)} patients working alone. A red bar to the left means it ends up slightly worse than working alone.</p>
      <div class="sm-legend" aria-hidden="true">
        <span><i class="sw alone"></i>Working alone (no gain)</span><span><i class="sw ghb"></i>Patients gained with GHB</span><span><i class="sw worse"></i>Patients lost with GHB</span>${showPool ? '<span><i class="sw pool"></i>Everyone pooled (not allowed)</span>' : ''}
      </div>
      <div class="sm-chart-wrap"><table class="sm-chart">
        <caption class="sm-vh">Patients each hospital gains with GHB, compared with working alone</caption>
        <thead><tr><th scope="col">Hospital</th><th scope="col" class="num">Has</th><th scope="col" class="sm-axis-h"><span class="sm-vh">Chart</span><div class="sm-axis" aria-hidden="true">${ticks.map((t) => `<span class="${t === 0 ? 'one' : ''}" style="left:${x(t)}%">${t > 0 ? '+' : ''}${t}</span>`).join('')}</div></th><th scope="col" class="num">Gained with GHB</th>${showPool ? '<th scope="col" class="num">Pooled</th>' : ''}</tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      ${showPool && gainP.some((g) => g > hi || g < lo) ? `<p class="sm-help">Where pooling everyone gains or loses more than the scale shows, it is marked with a triangle; the numbers on the right are exact. Pooling helps the small hospitals a lot but costs the large ones many patients' worth, because one shared model ignores how the hospitals differ.</p>` : ''}
      ${d.compare.length ? `<h3 class="sm-h3">Compared with other options</h3><ul class="sm-findings">${d.compare.map((s) => `<li class="${s.kind}">${esc(s.text)}</li>`).join('')}</ul>` : ''}
      <div id="sm-fed"></div>`;
    renderFed();
  }

  function renderFed(msg) {
    const el = resEl.querySelector('#sm-fed');
    if (!el) return;
    if (fedState === 'off') { el.innerHTML = ''; return; }
    if (fedState === 'running' || fedState === 'waiting') { el.innerHTML = `<p class="rd-progress" id="sm-fed-prog">${bar(0, 'Starting the second test with a standard federated method...')}</p>`; return; }
    if (fedState === 'failed') { el.innerHTML = `<p class="lv-bad">The second test failed: ${esc(msg || '')}</p>`; return; }
    const f = fed;
    el.innerHTML = `<table class="rd-table static sm-fedtable"><caption class="sm-help">Second test: a yes or no prediction model, ${f.K} hospitals of the same sizes. Average error compared with working alone (alone = 100).</caption>
      <thead><tr><th scope="col" class="lab">Approach</th><th scope="col">Average error</th><th scope="col">Hospitals worse off than alone</th></tr></thead>
      <tbody>
        <tr><td class="lab">Each hospital alone</td><td>100</td><td>-</td></tr>
        <tr class="ghbrow"><td class="lab">GHB</td><td>${pct(f.ghb.mean)}</td><td>${f.ghb.worse} of ${f.K}</td></tr>
        <tr><td class="lab">Standard federated method</td><td>${pct(f.fed.mean)}</td><td>${f.fed.worse} of ${f.K}</td></tr>
      </tbody></table>`;
  }

  const net = initSimpleNetwork(wrap.querySelector('#sm-netcard'), { choices: () => ({ ...ch }), result: () => (stale ? null : result), requestRun: () => { if (!runBtn.disabled) run(false); } });
  return { stop: () => { cancelJob('simple'); stopFed(); }, onShow: () => net.onShow() };
}
