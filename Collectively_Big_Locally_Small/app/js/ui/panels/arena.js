/**
 * ui/panels/arena.js
 * Responsibility: the Literature arena tab (Section 6.3, Simulations 8, 10 and 15). The logistic world of the paper's
 * Python simulations, where GHB, GHB-lite and every literature method of the paper run on the same hospitals at the
 * same (epsilon, delta) with one click. Presets, world and privacy controls, method selection by family, three tuning
 * modes (paper defaults, oracle, manual) with an editor for every hyperparameter, a module-worker run with progress
 * and cancel (main-thread fallback), a sortable results table with the paper's values for the presets, a dot chart,
 * a per-hospital view, a caption written from the result, and CSV/JSON export.
 */
import * as store from '../store.js';
import { controls, applyRelevance, progress, segmented } from '../controls.js';
import { add, explain } from '../dom.js';
import { Figure } from '../../viz/chart.js';
import { fmt } from '../../viz/theme.js';
import { resolveArena } from '../../config.js';
import { ARENA_METHODS, ARENA_METHOD, ARENA_FAMILIES, runArena } from '../../sim/arena.js';
import { runJob, cancelJob } from '../../sim/scheduler.js';
import { ARENA_PRESETS, arenaPresetConfig } from '../../scenarios.js';

/** Full references of every method in the arena. @see the paper's bibliography */
export const ARENA_REFERENCES = [
  ['Bonawitz et al., 2017', 'K. Bonawitz et al. Practical secure aggregation for privacy-preserving machine learning. ACM CCS 2017.'],
  ['Collins et al., 2021', 'L. Collins, H. Hassani, A. Mokhtari, S. Shakkottai. Exploiting shared representations for personalized federated learning. ICML 2021.'],
  ['Deng et al., 2020', 'Y. Deng, M. M. Kamani, M. Mahdavi. Adaptive personalized federated learning. arXiv:2003.13461, 2020.'],
  ['DerSimonian & Laird, 1986', 'R. DerSimonian, N. Laird. Meta-analysis in clinical trials. Controlled Clinical Trials 7(3):177-188, 1986.'],
  ['Fallah et al., 2020', 'A. Fallah, A. Mokhtari, A. Ozdaglar. Personalized federated learning with theoretical guarantees: a model-agnostic meta-learning approach. NeurIPS 2020.'],
  ['Ghosh et al., 2020', 'A. Ghosh, J. Chung, D. Yin, K. Ramchandran. An efficient framework for clustered federated learning. NeurIPS 2020.'],
  ['Li et al., 2020', 'T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith. Federated optimization in heterogeneous networks (FedProx). MLSys 2020.'],
  ['Li et al., 2021', 'T. Li, S. Hu, A. Beirami, V. Smith. Ditto: fair and robust federated learning through personalization. ICML 2021.'],
  ['Lian et al., 2017', 'X. Lian, C. Zhang, H. Zhang, C.-J. Hsieh, W. Zhang, J. Liu. Can decentralized algorithms outperform centralized algorithms? NeurIPS 2017.'],
  ['McMahan et al., 2017', 'B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. Communication-efficient learning of deep networks from decentralized data. AISTATS 2017.'],
  ['McMahan et al., 2018', 'H. B. McMahan, D. Ramage, K. Talwar, L. Zhang. Learning differentially private recurrent language models. ICLR 2018.'],
  ['T. Dinh et al., 2020', 'C. T. Dinh, N. H. Tran, T. D. Nguyen. Personalized federated learning with Moreau envelopes (pFedMe). NeurIPS 2020.'],
  ['Bun & Steinke, 2016', 'M. Bun, T. Steinke. Concentrated differential privacy: simplifications, extensions, and lower bounds. TCC 2016 (zCDP accounting).']
];

const labelOf = (m) => `${m.label}${m.cite ? ` <span class="rd-cite">(${m.cite})</span>` : ''}`;
const plainLabel = (m) => `${m.label}${m.cite ? ` (${m.cite})` : ''}`;
const isCandidate = (id) => !['reference'].includes(ARENA_METHOD[id].family) || id === 'local';
const cfgText = (c) => (c ? Object.entries(c).map(([k, v]) => `${ARENA_METHODS.flatMap((m) => m.hyper || []).find((h) => h.k === k)?.label.split(' ').pop() || k}=${v}`).join(', ') : '');

export function initArena() {
  const rail = document.getElementById('rail-arena'), stage = document.getElementById('stage-arena'), nb = document.getElementById('nb-arena');
  let selected = new Set(ARENA_METHODS.map((m) => m.id)), manual = {}, worker = null, result = null, presetId = 'all', sortKey = null, sortDir = 1, t0 = 0;

  /* ---------- rail: presets, world, privacy, anchor, tuning, Monte Carlo, methods ---------- */
  add(rail, `<div class="rd-ctl"><label for="arena-preset">Preset</label><select id="arena-preset">${ARENA_PRESETS.map((p) => `<option value="${p.id}">${p.title}</option>`).join('')}</select></div>`);
  const presetNote = add(rail, '<p class="rd-msg" id="arena-preset-note"></p>');
  rail.appendChild(segmented('aTuning', 'arena', 'Tuning of the literature methods'));
  controls(rail, ['aK', 'aMedian', 'aTau2', 'aShift', 'aRegions', 'aTauR2'], 'arena', 'World (Section 6.3)');
  controls(rail, ['aEps', 'aNoDP', 'aDelta'], 'arena', 'Privacy (same budget for every method)');
  controls(rail, ['aAnchorErr', 'aAnchorSd', 'aCheck'], 'arena', 'Published anchor μ₀');
  controls(rail, ['aReps', 'aTune', 'aSeed', 'aNTest'], 'arena', 'Monte Carlo');
  add(rail, '<h3 class="rd-h">Methods</h3>');
  const famBtns = add(rail, `<div class="rd-btns"><button class="rd-btn" type="button" id="arena-all">All</button><button class="rd-btn" type="button" id="arena-none">None</button></div>`);
  const methBox = add(rail, '<div class="rd-methods" id="arena-methods"></div>');
  for (const [fam, title] of ARENA_FAMILIES) {
    add(methBox, `<div class="rd-check"><input type="checkbox" id="arena-fam-${fam}" data-fam="${fam}"><label for="arena-fam-${fam}"><strong>${title}</strong></label></div>`);
    for (const m of ARENA_METHODS.filter((x) => x.family === fam)) add(methBox, `<div class="rd-check" style="padding-left:16px"><input type="checkbox" id="arena-m-${m.id}" data-m="${m.id}" ${m.id === 'local' ? 'disabled' : ''}><label for="arena-m-${m.id}"><span class="rd-dot" style="--c:var(${m.tok})"></span>${labelOf(m)}</label></div>`);
  }

  /* ---------- stage ---------- */
  add(stage, '<div><h2 class="rd-h2">Literature arena: GHB against every method in the paper</h2><p class="rd-prose">The logistic-regression world of the paper\'s simulations (Section 6.3): every hospital fits a five-coefficient logistic head, coefficients differ between hospitals by τ², and every private method spends the same (ε, δ) per hospital under zero-concentrated accounting. Error is the excess log-loss over the true model on fresh patients of each hospital.</p></div>');
  const runRow = add(stage, '<div class="rd-btns"><button class="rd-btn primary" type="button" id="arena-run">Run all methods</button><button class="rd-btn" type="button" id="arena-cancel" disabled>Cancel</button><span class="rd-progress" id="arena-progress" aria-live="polite"></span></div>');
  const estEl = add(stage, '<p class="rd-msg" id="arena-estimate"></p>');
  const capEl = add(stage, '<div class="rd-callout" id="arena-caption">Press <strong>Run all methods</strong>. A run of every method with the paper defaults takes about a second per replicate; oracle tuning evaluates every grid configuration (about 800 candidates) on the tuning replicates first, a few seconds each.</div>');
  add(stage, '<h3 class="rd-h">Results (click a header to sort; bold: best among private methods that use other hospitals)</h3>');
  const tableWrap = add(stage, '<div class="rd-table-wrap" style="max-height:none"><table class="rd-table static" id="arena-table"><thead></thead><tbody></tbody></table></div>');
  const expRow = add(stage, '<div class="rd-btns"><button class="rd-btn" type="button" id="arena-csv" disabled>Export CSV</button><button class="rd-btn" type="button" id="arena-json" disabled>Export JSON</button><span class="rd-msg">Monte Carlo error: values within about ±2 SE of each other are not distinguishable; use more replicates to separate them.</span></div>');
  const g = add(stage, '<div class="rd-grid2"></div>');
  const fDots = new Figure(add(g, '<div></div>'), { id: 'fig-arena-dots', num: 'L1', caption: 'Mean excess log-loss (10<sup>−3</sup> nats) of every method, ±2 Monte Carlo SE; dashed: local estimation. Hollow grey rings: the paper\'s value for the preset, where it reports one.', xLabel: 'mean excess log-loss (10⁻³ nats)', yLabel: '', height: 360, canLogY: false });
  const fHosp = new Figure(add(g, '<div></div>'), { id: 'fig-arena-hosp', num: 'L2', caption: 'Each hospital\'s mean excess log-loss relative to its own local estimate (hospitals sorted by size; labels n<sub>k</sub>). Above 1, that hospital is worse off than alone. Five methods are shown at first; toggle the others with the keys.', xLabel: 'hospital size n_k (sorted)', yLabel: 'excess log-loss / local', height: 360, canLogX: false, logY: true });
  const hyperBox = add(stage, '<details class="rd-group" id="arena-hyper"><summary>Hyperparameters of every method (editable in manual mode; GHB and GHB-lite have none to tune)</summary><div class="rd-hyper" id="arena-hyper-body"></div></details>');

  /* ---------- notebook ---------- */
  explain(nb, 'The world', [
    'Hospital k has coefficients θ<sub>k</sub> = μ + √τ² z<sub>k</sub> on an intercept and four bounded features (‖x‖ ≤ √5), optionally shifted case mix and regional effects; sizes are the paper\'s log-normal profile (median 100, 24 hospitals). A published model μ₀ = μ + error serves as anchor. Gossip runs on the ring with two-hop links (T = 60).',
    'Every private method spends the same (ε, δ = 10<sup>−5</sup>) per hospital; iterative methods compose their rounds under zero-concentrated DP (Bun &amp; Steinke, 2016), the tightest standard accounting, which favours them. GHB spends ρ on two joint releases (Proposition 5.1); GHB-lite on one release per hospital.'
  ]);
  explain(nb, 'Tuning', [
    '<strong>Paper defaults</strong>: each literature method runs one fixed configuration, the one Simulation 15 chose at τ² = 0.05, ε = 1 for the simulation-world baselines and the one frozen for the realistic arm of Section 7 for the personalized methods.',
    '<strong>Oracle</strong>: every configuration of each method\'s grid is evaluated on separate tuning replicates against the true loss; the best is then reported on fresh replicates, as Simulation 15 does. No hospital can tune this way.',
    '<strong>Manual</strong>: set every hyperparameter yourself below the figures. GHB and GHB-lite have nothing to tune; their settings are fixed in advance (Section 5).'
  ]);
  add(nb, `<div><h3 class="rd-h">References</h3><ul class="rd-refs">${ARENA_REFERENCES.map(([k, v]) => `<li><strong>${k}.</strong> ${v.replace(/^[^.]*\. /, '')}</li>`).join('')}</ul></div>`);
  add(nb, '<div><h3 class="rd-h">Faithful to the paper\'s code</h3><p class="rd-prose">Every method is a line-by-line port of simulations/logit_world.py, ghb_logit.py, ghb_v4.py, literature.py, personalized.py and ghb_sa.py; tests/logit.test.mjs checks each against the Python output on fixed data to 10<sup>−8</sup>, and tests/arena.test.mjs checks the Simulation 15 means. Random draws come from this page\'s generator, so individual numbers differ from numpy\'s within Monte Carlo error.</p></div>');

  /* ---------- method selection ---------- */
  function syncChecks() {
    methBox.querySelectorAll('[data-m]').forEach((el) => { el.checked = selected.has(el.dataset.m) || el.dataset.m === 'local'; });
    methBox.querySelectorAll('[data-fam]').forEach((el) => { const ms = ARENA_METHODS.filter((m) => m.family === el.dataset.fam && m.id !== 'local'); const k = ms.filter((m) => selected.has(m.id)).length; el.checked = k === ms.length; el.indeterminate = k > 0 && k < ms.length; });
    estimate();
  }
  methBox.addEventListener('change', (e) => {
    const t = e.target;
    if (t.dataset.m) { if (t.checked) selected.add(t.dataset.m); else selected.delete(t.dataset.m); }
    if (t.dataset.fam) ARENA_METHODS.filter((m) => m.family === t.dataset.fam).forEach((m) => { if (t.checked) selected.add(m.id); else if (m.id !== 'local') selected.delete(m.id); });
    syncChecks();
  });
  famBtns.querySelector('#arena-all').onclick = () => { selected = new Set(ARENA_METHODS.map((m) => m.id)); syncChecks(); };
  famBtns.querySelector('#arena-none').onclick = () => { selected = new Set(['local']); syncChecks(); };

  /* ---------- hyperparameter editor ---------- */
  function renderHyper() {
    const c = store.get(), on = c.aTuning === 'manual', body = hyperBox.querySelector('#arena-hyper-body');
    body.innerHTML = ARENA_METHODS.filter((m) => m.hyper || m.settings).map((m) => {
      if (m.settings) return `<fieldset><legend>${m.label} (fixed, Section ${m.id === 'ghb' ? '5.2' : '5.3'})</legend><dl>${Object.entries(m.settings).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl></fieldset>`;
      const cur = { ...m.def, ...(manual[m.id] || {}) };
      return `<fieldset><legend>${plainLabel(m)}</legend>${m.hyper.map((h) => `<label for="arena-h-${m.id}-${h.k}"><span>${h.label} <span class="rd-cite">grid ${h.grid.join(', ')}</span></span><input type="number" step="any" min="0" id="arena-h-${m.id}-${h.k}" data-m="${m.id}" data-k="${h.k}" value="${cur[h.k]}" ${on ? '' : 'disabled'}></label>`).join('')}</fieldset>`;
    }).join('');
  }
  hyperBox.addEventListener('change', (e) => {
    const t = e.target; if (!t.dataset.k) return;
    const m = ARENA_METHOD[t.dataset.m], h = m.hyper.find((x) => x.k === t.dataset.k);
    let v = Math.max(0, +t.value || 0); if (h.int) v = Math.max(h.k === 'C' ? 1 : 0, Math.round(v)); if (h.k === 'R' || h.k === 'C') v = Math.max(1, v);
    if (h.k === 'C') v = Math.min(v, store.get().aK);
    t.value = v; manual[m.id] = { ...(manual[m.id] || {}), [h.k]: v };
  });

  /* ---------- running ---------- */
  function estimate() {
    const c = store.get(), n = selected.size, perRep = 0.06 * n * (c.aK / 24) * (c.aNTest / 4000 * 0.3 + 0.7);
    const tune = c.aTuning === 'oracle' ? c.aTune * 4.5 * (c.aK / 24) * ([...selected].filter((id) => ARENA_METHOD[id].hyper).length / 12) : 0;
    estEl.textContent = `${n} methods, ${c.aReps} replicates${c.aTuning === 'oracle' ? ` after ${c.aTune} tuning replicates` : ''}: roughly ${Math.max(1, Math.round(perRep * c.aReps + tune))} s on a laptop. With ${c.aReps} replicates the Monte Carlo SE of a mean is about ${fmt(2.5 / Math.sqrt(c.aReps) * 0.6, 1)} × 10⁻³ nats.`;
  }
  const btnRun = runRow.querySelector('#arena-run'), btnCancel = runRow.querySelector('#arena-cancel'), progEl = runRow.querySelector('#arena-progress');
  function setRunning(on) { btnRun.disabled = on; btnCancel.disabled = !on; }
  function onProgress(p) {
    const el = (Date.now() - t0) / 1000, eta = p.done ? el / p.done * (p.total - p.done) : 0;
    progEl.innerHTML = progress(p.done, p.total, `${p.phase === 'tuning' ? 'tuning replicates first, then' : ''} replicates`) + ` ${p.phase}, ${el.toFixed(0)} s elapsed${p.done < p.total ? `, about ${eta.toFixed(0)} s left` : ''}`;
  }
  function onDone(r) { worker?.terminate(); worker = null; setRunning(false); result = { ...r, presetId }; progEl.textContent = `Done: ${r.spec.reps} replicates${r.spec.tuning === 'oracle' ? ` after ${r.spec.tuneReps} tuning replicates` : ''} in ${r.seconds.toFixed(1)} s.`; render(); }
  function onFail(msg) { worker?.terminate(); worker = null; setRunning(false); progEl.innerHTML = `<span class="lv-bad">The run failed: ${msg}</span>`; }
  function fallback(spec) {
    progEl.textContent = 'Running on the main thread (this browser has no module workers); the page may pause between replicates.';
    const gen = runArena(spec);
    runJob('arena', (function* () { let r = gen.next(); while (!r.done) { onProgress(r.value); yield; r = gen.next(); } return r.value; })(), { onDone, onError: (e) => onFail(e.message) });
  }
  function start() {
    cancel(true);
    const spec = resolveArena(store.get(), [...selected], manual);
    t0 = Date.now(); setRunning(true); progEl.textContent = 'Starting...';
    let got = false;
    try {
      worker = new Worker(new URL('../../sim/arena.worker.js', import.meta.url), { type: 'module' });
      worker.onmessage = (e) => { got = true; const d = e.data; if (d.type === 'progress') onProgress(d); else if (d.type === 'done') onDone(d.result); else onFail(d.message); };
      worker.onerror = (e) => { e.preventDefault?.(); if (!got) { worker?.terminate(); worker = null; fallback(spec); } else onFail(e.message || 'worker error'); };
      worker.postMessage({ spec });
    } catch (e) { worker = null; fallback(spec); }
  }
  function cancel(silent) {
    if (worker) { worker.terminate(); worker = null; }
    cancelJob('arena'); setRunning(false);
    if (!silent) progEl.textContent = 'Cancelled.';
  }
  btnRun.onclick = start;
  btnCancel.onclick = () => cancel(false);

  /* ---------- results ---------- */
  const COLS = [
    { k: 'label', lab: 'Method', cls: 'lab' }, { k: 'mean', lab: 'Excess log-loss ×10⁻³', best: 'min' }, { k: 'rel', lab: 'vs local', best: 'min' }, { k: 'auc', lab: 'AUC', best: 'max' },
    { k: 'worst', lab: 'Worst hospital vs local', best: 'min' }, { k: 'share', lab: 'Hospitals that gain', best: 'max' }, { k: 'large', lab: 'Largest third ×10⁻³', best: 'min' },
    { k: 'small', lab: 'Smallest third ×10⁻³', best: 'min' }, { k: 'paper', lab: 'Paper ×10⁻³' }, { k: 'config', lab: 'Configuration', cls: 'lab' }
  ];
  function rows() {
    const S = result.summary, loc = S.local.mean, ref = ARENA_PRESETS.find((p) => p.id === result.presetId)?.ref || {};
    return Object.keys(S).map((id) => ({ id, m: ARENA_METHOD[id], s: S[id], mean: S[id].mean, rel: S[id].mean / loc, auc: S[id].auc, worst: S[id].worst, share: S[id].share, large: S[id].large, small: S[id].small, paper: ref[id] ?? null, config: S[id].config }));
  }
  function render() {
    if (!result) return;
    const R = rows(), third = Math.floor(result.K / 3);
    if (sortKey) R.sort((a, b) => sortDir * ((sortKey === 'label' ? a.m.label.localeCompare(b.m.label) : (a[sortKey] ?? Infinity) - (b[sortKey] ?? Infinity))));
    const cand = R.filter((r) => isCandidate(r.id) && r.id !== 'local'), best = {};
    for (const c of COLS.filter((x) => x.best)) { const v = cand.map((r) => r[c.k]).filter((x) => isFinite(x)); best[c.k] = c.best === 'min' ? Math.min(...v) : Math.max(...v); }
    const hasPaper = R.some((r) => r.paper != null);
    const cols = COLS.filter((c) => c.k !== 'paper' || hasPaper).map((c) => (c.k === 'large' ? { ...c, lab: `${third} largest ×10⁻³` } : c.k === 'small' ? { ...c, lab: `${third} smallest ×10⁻³` } : c));
    const T = tableWrap.querySelector('table');
    T.querySelector('thead').innerHTML = `<tr>${cols.map((c) => `<th scope="col" class="${c.cls || ''}" aria-sort="${c.k === sortKey ? (sortDir > 0 ? 'ascending' : 'descending') : 'none'}"><button type="button" data-k="${c.k}">${c.lab}${c.k === sortKey ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`).join('')}</tr>`;
    const cell = (r, c) => {
      const isBest = c.best && isCandidate(r.id) && r.id !== 'local' && r[c.k] === best[c.k];
      const b = isBest ? ' best' : '';
      switch (c.k) {
        case 'label': return `<td class="lab"><span class="rd-dot" style="--c:var(${r.m.tok})"></span> ${labelOf(r.m)}${r.m.private === false ? ' <span class="rd-tag">no DP</span>' : ''}</td>`;
        case 'mean': return `<td class="num${b}">${(1000 * r.mean).toFixed(1)}<small> ±${(2000 * r.s.se).toFixed(1)}</small></td>`;
        case 'rel': return `<td class="num${b} ${r.rel < 1 ? 'gain' : r.rel > 1.02 ? 'loss' : ''}">${r.rel.toFixed(2)}</td>`;
        case 'auc': return `<td class="num${b}">${r.auc.toFixed(3)}</td>`;
        case 'worst': return `<td class="num${b} ${r.worst > 1.02 ? 'loss' : ''}">${r.worst.toFixed(2)}</td>`;
        case 'share': return `<td class="num${b}">${Math.round(100 * r.share)}%</td>`;
        case 'large': case 'small': return `<td class="num${b}">${(1000 * r[c.k]).toFixed(1)}</td>`;
        case 'paper': return `<td class="num">${r.paper != null ? r.paper.toFixed(1) : ''}</td>`;
        default: return `<td class="lab mono">${r.config ? cfgText(r.config) : r.m.hyper ? '' : 'none to tune'}</td>`;
      }
    };
    T.querySelector('tbody').innerHTML = R.map((r) => `<tr class="${r.m.family === 'ghb' ? 'ghbrow' : ''}">${cols.map((c) => cell(r, c)).join('')}</tr>`).join('');
    // dot chart, best at the top
    const D = rows().sort((a, b) => b.mean - a.mean);
    fDots.setData({ yCats: D.map((r) => r.m.short || r.m.label), xName: 'excess ×10⁻³', yName: 'method', yZero: false,
      series: [...D.map((r, i) => ({ id: r.id, label: r.m.short || r.m.label, tok: r.m.tok, noLine: true, marker: 4.5, xerr: true, toggle: false, pts: [[1000 * r.mean, i, 1000 * r.s.se]] })),
        ...(D.some((r) => r.paper != null) ? [{ id: 'paper', label: 'paper value', tok: '--rd-muted', noLine: true, marker: 2, pts: D.map((r, i) => (r.paper != null ? [r.paper, i] : null)).filter(Boolean) }] : [])],
      vlines: [{ x: 1000 * result.summary.local.mean, tok: '--rd-local' }] });
    if (!fHosp.touched) { fHosp.touched = true; fHosp.hidden = new Set(Object.keys(result.summary).filter((id) => !['ghb', 'lite', 'dittoStyle', 'dpfedavgFT', 'pfedme'].includes(id))); fHosp.sig = null; }
    const order = result.n.map((_, k) => k).sort((a, b) => result.n[a] - result.n[b]), ids = Object.keys(result.summary).filter((id) => id !== 'local');
    fHosp.setData({ xCats: order.map((k) => String(result.n[k])), xName: 'n_k', yName: 'ratio to local',
      series: ids.map((id) => ({ id, label: ARENA_METHOD[id].short || ARENA_METHOD[id].label, tok: ARENA_METHOD[id].tok, width: ['ghb', 'lite'].includes(id) ? 2.2 : 1.2, marker: 2, pts: order.map((k, i) => [i, result.summary[id].ratio[k]]) })),
      hlines: [{ y: 1, tok: '--rd-local', dash: true, label: 'local = 1' }] });
    capEl.innerHTML = caption();
    expRow.querySelectorAll('button').forEach((b) => { b.disabled = false; });
  }
  function caption() {
    const R = rows(), sp = result.spec, S = result.summary, priv = R.filter((r) => isCandidate(r.id) && r.id !== 'local').sort((a, b) => a.mean - b.mean), loc = S.local.mean;
    if (!priv.length) return `Only local estimation was run (${(1000 * loc).toFixed(1)} × 10⁻³ nats). Select methods in the rail.`;
    const top = priv[0], gain = priv.filter((r) => r.mean < loc).length, fair = priv.filter((r) => r.worst <= 1.1).map((r) => r.m.short || r.m.label);
    const g = S.ghb, l = S.lite, within = (a, b) => Math.abs(a.mean - b.mean) <= 2 * Math.hypot(a.se, b.se);
    let t = `<strong>What this shows.</strong> With ${result.K} hospitals, τ² = ${fmt(sp.tau2, 3)}${sp.regions ? ` (plus ${sp.regions} regions, τ_r² = ${fmt(sp.tauR2, 3)})` : ''}, ${isFinite(sp.eps) ? `ε = ${fmt(sp.eps, 2)}` : 'no privacy'}${sp.shift ? `, case-mix shift ${fmt(sp.shift, 2)}` : ''} and ${sp.tuning === 'oracle' ? 'baselines tuned with oracle access to the true loss' : sp.tuning === 'manual' ? 'manually set baselines' : 'the paper-default configurations'}, `
      + `the most accurate private method that uses other hospitals is <strong>${plainLabel(top.m)}</strong> at ${(1000 * top.mean).toFixed(1)} × 10⁻³ nats against ${(1000 * loc).toFixed(1)} for local estimation; ${gain} of ${priv.length} such methods beat local on average.`;
    if (g && l) t += ` GHB reaches ${(1000 * g.mean).toFixed(1)} and GHB-lite ${(1000 * l.mean).toFixed(1)}${within(g, l) ? ', within Monte Carlo error of each other' : g.mean < l.mean ? ': the joint release wins here, as it does when per-hospital releases are noisy' : ': GHB-lite wins here, as in the paper\'s simulated worlds where its releases are informative'}.`;
    t += ` ${fair.length ? `Worst hospital within 10% of its own local error: ${fair.join(', ')}.` : 'Every method leaves some hospital more than 10% worse off than alone.'}`;
    const worstLit = priv.filter((r) => !['ghb', 'lite'].includes(r.id)).sort((a, b) => b.worst - a.worst)[0];
    if (worstLit && worstLit.worst > 1.2) t += ` The hardest hit hospital under ${worstLit.m.short || worstLit.m.label} is ${worstLit.worst.toFixed(1)} times worse off than alone.`;
    t += ` Differences smaller than about ±${(2000 * top.s.se).toFixed(1)} × 10⁻³ are within Monte Carlo error with ${sp.reps} replicates.`;
    return t;
  }
  tableWrap.addEventListener('click', (e) => { const b = e.target.closest('button[data-k]'); if (!b) return; const k = b.dataset.k; if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = k === 'auc' || k === 'share' ? -1 : 1; } render(); });

  /* ---------- export ---------- */
  function download(name, text, type) {
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name;
    document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }
  expRow.querySelector('#arena-csv').onclick = () => {
    if (!result) return;
    const head = ['method', 'citation', 'mean_excess_logloss', 'se', 'relative_to_local', 'auc', 'worst_ratio', 'share_gain', 'largest_third', 'smallest_third', 'paper_value_1e-3', 'configuration'];
    const q = (s) => `"${String(s ?? '').replace(/"/g, '""')}"`;
    const lines = rows().map((r) => [q(r.m.label), q(r.m.cite || ''), r.mean, r.s.se, r.rel, r.auc, r.worst, r.share, r.large, r.small, r.paper ?? '', q(r.config ? cfgText(r.config) : '')].join(','));
    download(`ghb-arena-${result.presetId}.csv`, [head.join(','), ...lines].join('\n') + '\n', 'text/csv');
  };
  expRow.querySelector('#arena-json').onclick = () => { if (result) download(`ghb-arena-${result.presetId}.json`, JSON.stringify({ ...result, spec: { ...result.spec, eps: isFinite(result.spec.eps) ? result.spec.eps : 'Infinity' } }, null, 1), 'application/json'); };

  /* ---------- presets and wiring ---------- */
  const presetSel = rail.querySelector('#arena-preset');
  function applyPreset(id) {
    const p = ARENA_PRESETS.find((x) => x.id === id) || ARENA_PRESETS[0];
    presetId = p.id; presetSel.value = p.id;
    store.replace(arenaPresetConfig(store.get(), p.id));
    selected = new Set(p.methods || ARENA_METHODS.map((m) => m.id)); selected.add('local'); manual = {};
    presetNote.textContent = p.note + (p.ref ? ' The table shows the paper\'s values next to yours.' : '');
    syncChecks(); renderHyper();
  }
  presetSel.addEventListener('change', () => applyPreset(presetSel.value));
  store.subscribe((keys) => {
    if (keys.some((k) => k.startsWith('a'))) { applyRelevance(document.getElementById('p-arena'), store.get()); estimate(); }
    if (keys.includes('aTuning')) renderHyper();
  });
  applyPreset('all');
  applyRelevance(document.getElementById('p-arena'), store.get());
  return {
    /** Load a preset and optionally run it at once (Start here demos). */
    runPreset(id, run = true) { applyPreset(id); if (run) start(); },
    onShow() { estimate(); }
  };
}
