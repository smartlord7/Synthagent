/**
 * ui/panels/setup.js
 * Responsibility: the Setup tab, where every parameter of the simulation lives: network (with an editable
 * table of hospital sizes and a graph editor: click two hospitals to add or remove their agreement), model,
 * privacy, estimation, safeguards, adversaries and Monte Carlo settings, plus JSON export and import.
 */
import * as store from '../store.js';
import { PARAMS, GROUPS, resolveSizes, resolveGraph } from '../../config.js';
import { control, applyRelevance } from '../controls.js';
import { add, explain } from '../dom.js';
import { NetworkView } from '../../viz/network.js';
import { edgeList, connected, metropolisHastings, slem } from '../../math/graph.js';
import { fmt } from '../../viz/theme.js';

export function initSetup() {
  const rail = document.getElementById('rail-setup'), stage = document.getElementById('stage-setup'), nb = document.getElementById('nb-setup');
  const groupBox = (parent, g, label, open) => {
    const d = add(parent, `<details class="rd-group" ${open ? 'open' : ''} id="setup-group-${g}"><summary>${label}</summary><div class="rd-group-body"></div></details>`);
    const body = d.querySelector('.rd-group-body');
    PARAMS.filter((p) => p.group === g).forEach((p) => body.appendChild(control(p.key, 'setup')));
    return d;
  };
  // rail: network parameters
  add(rail, '<h3 class="rd-h">Network</h3>');
  PARAMS.filter((p) => p.group === 'network').forEach((p) => rail.appendChild(control(p.key, 'setup')));
  // stage: graph editor and sizes
  add(stage, '<h2 class="rd-h2">Hospitals and agreements</h2>');
  const info = add(stage, '<p class="rd-msg" id="setup-graph-info"></p>');
  const wrap = add(stage, '<div class="rd-canvas-wrap"><canvas class="rd-canvas" id="setup-canvas" data-h="340" tabindex="0" aria-label="Graph editor: select two hospitals to add or remove an agreement"></canvas></div>');
  add(stage, '<p class="rd-prose">Click one hospital, then another, to add or remove the agreement between them (keyboard: arrows to move, Enter to pick). Editing switches the topology to Custom. Node area is proportional to hospital size.</p>');
  const view = new NetworkView(wrap.querySelector('canvas'), {
    onEdgeToggle: (i, j) => {
      const c = store.get(), K = resolveSizes(c).length;
      let E = c.topo === 'custom' && c.customEdges ? c.customEdges.slice() : edgeList(resolveGraph(c, K));
      const idx = E.findIndex(([a, b]) => a === i && b === j);
      if (idx >= 0) E.splice(idx, 1); else E.push([i, j]);
      store.replace({ ...c, topo: 'custom', customEdges: E });
    }
  });
  view.edit = true;
  add(stage, '<h3 class="rd-h">Hospital sizes n<sub>k</sub> (editable table; editing switches sizes to the table)</h3>');
  const sizesBox = add(stage, '<div class="rd-sizes" id="setup-sizes"></div>');
  const sizeBtns = add(stage, '<div class="rd-btns"><button class="rd-btn" type="button" id="setup-add-h">Add hospital</button><button class="rd-btn" type="button" id="setup-del-h">Remove last</button><button class="rd-btn" type="button" id="setup-copy-sizes">Copy current sizes into table</button></div>');
  for (const [g, label] of GROUPS) if (g !== 'network' && g !== 'calc' && g !== 'arena') groupBox(stage, g, label, ['model', 'privacy', 'estimation'].includes(g));
  // notebook: explanation and JSON
  explain(nb, 'Nothing is hard-coded', ['Every quantity the simulations use is set here and shared by all tabs. The defaults are the paper\'s: 24 hospitals with sizes from 16 to 479 patients, τ² = 0.02, σ² = 1, ε = 1, δ = 10⁻⁵, C = 1, 60 gossip rounds on the ring with two-hop links.', 'Dimmed controls have no effect under the current choices (for example ν when site effects are Gaussian).']);
  add(nb, '<h3 class="rd-h">Configuration as JSON</h3>');
  const ta = add(nb, '<textarea class="rd-input" id="setup-json" rows="10" spellcheck="false" aria-label="Configuration JSON"></textarea>');
  add(nb, '<div class="rd-btns"><button class="rd-btn" type="button" id="setup-export">Copy JSON</button><button class="rd-btn primary" type="button" id="setup-import">Apply JSON</button></div>');
  const msg = add(nb, '<p class="rd-msg" id="setup-json-msg" aria-live="polite"></p>');

  document.getElementById('setup-export').onclick = () => {
    ta.value = JSON.stringify(store.get(), null, 1);
    navigator.clipboard?.writeText(ta.value).then(() => { msg.textContent = 'Copied to the clipboard.'; msg.className = 'rd-msg'; }, () => { ta.select(); msg.textContent = 'Selected: press Ctrl+C (Cmd+C) to copy.'; });
  };
  document.getElementById('setup-import').onclick = () => {
    try {
      const w = store.replace(JSON.parse(ta.value));
      msg.className = 'rd-msg'; msg.textContent = w.length ? 'Applied with notes: ' + w.join(' ') : 'Applied.';
    } catch (e) { msg.className = 'rd-msg err'; msg.textContent = 'This is not valid JSON: ' + e.message + '. Fix the text and apply again.'; }
  };
  const setTable = (arr) => store.replace({ ...store.get(), sizeMode: 'table', sizesTable: arr, K: arr.length });
  document.getElementById('setup-add-h').onclick = () => { const t = store.get().sizeMode === 'table' ? store.get().sizesTable.slice() : resolveSizes(store.get()); if (t.length < 48) { t.push(100); setTable(t); } };
  document.getElementById('setup-del-h').onclick = () => { const t = store.get().sizeMode === 'table' ? store.get().sizesTable.slice() : resolveSizes(store.get()); if (t.length > 3) { t.pop(); setTable(t); } };
  document.getElementById('setup-copy-sizes').onclick = () => setTable(resolveSizes(store.get()));
  sizesBox.addEventListener('change', (e) => {
    const i = +e.target.dataset.i; if (!(i >= 0)) return;
    const t = resolveSizes(store.get()); t[i] = Math.max(2, Math.round(+e.target.value || 2)); setTable(t);
  });

  function update() {
    const c = store.get(), n = resolveSizes(c), K = n.length, A = resolveGraph(c, K), E = edgeList(A);
    view.setGraph({ K, edges: E, sizes: n });
    const conn = connected(A);
    info.innerHTML = `${K} hospitals, ${E.length} agreements, mean degree ${fmt(2 * E.length / K, 2)}; ${conn ? `λ = ${fmt(slem(metropolisHastings(A)), 3)}` : '<span class="lv-bad">the graph is disconnected, so gossip cannot reach consensus (λ = 1)</span>'}.`;
    sizesBox.innerHTML = n.map((x, i) => `<label for="setup-n-${i}">n<sub>${i + 1}</sub><input type="number" min="2" max="100000" id="setup-n-${i}" data-i="${i}" value="${x}"></label>`).join('');
    applyRelevance(document.getElementById('p-setup'), c);
    if (document.activeElement !== ta) ta.value = JSON.stringify(c, null, 1);
  }
  store.subscribe(update);
  update();
  return { redraw: () => view.draw() };
}
