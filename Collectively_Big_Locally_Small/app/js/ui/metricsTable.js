/**
 * ui/metricsTable.js
 * Responsibility: the sortable per-hospital metrics table (every metric of the engine summary, with Monte
 * Carlo standard errors). Clicking a header sorts; clicking a row selects that hospital.
 */
import { fmt } from '../viz/theme.js';
import { METHOD } from '../sim/engine.js';

/**
 * Create a metrics table.
 * @param {HTMLElement} host container
 * @param {string} id stable id prefix
 * @param {(k:number)=>void} onSelect row selection callback
 * @returns {{update:(summary:Object, methods:string[])=>void, select:(k:number)=>void}}
 */
export function createMetricsTable(host, id, onSelect) {
  host.innerHTML = `<div class="rd-table-wrap"><table class="rd-table" id="${id}"><thead></thead><tbody></tbody></table></div>`;
  const table = host.querySelector('table');
  let sortKey = 'n', dir = 1, data = null, methods = [], sel = -1;
  const cols = () => [
    ['k', 'k', (h) => h.k + 1, 0], ['n', 'n_k', (h) => h.n, 0], ['v', 'v_k', (h) => h.v, 5], ['s', 's_k', (h) => h.s, 4], ['nsr', 's²/v', (h) => h.nsr, 3],
    ['x', 'release', (h) => h.x, 4], ['mu', 'μ̃_k', (h) => h.mu, 4], ['M', 'M_k', (h) => h.M, 5], ['t2', 'τ̂²', (h) => h.t2, 4], ['alpha', 'α_k', (h) => h.alpha, 3],
    ['B', 'B_k', (h) => h.B, 1], ['neffTheory', 'n_k+B', (h) => h.neffTheory, 0], ['neff', 'n_eff (MC)', (h) => h.neff, 0], ['gain', 'n_eff/n_k', (h) => h.gain, 2],
    ...methods.map((m) => ['rel:' + m, 'R/v ' + METHOD[m].label, (h) => h.rel[m].y, 3, (h) => h.rel[m].se])
  ];
  function render() {
    if (!data) return;
    const C = cols();
    table.querySelector('thead').innerHTML = `<tr>${C.map(([k, lab]) => `<th scope="col" aria-sort="${k === sortKey ? (dir > 0 ? 'ascending' : 'descending') : 'none'}"><button type="button" data-k="${k}">${lab}${k === sortKey ? (dir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`).join('')}</tr>`;
    const col = C.find((c) => c[0] === sortKey) || C[1];
    const rows = data.perHospital.slice().sort((a, b) => dir * (col[2](a) - col[2](b)));
    table.querySelector('tbody').innerHTML = rows.map((h) => `<tr data-k="${h.k}" class="${h.k === sel ? 'sel' : ''}">${C.map(([k, , f, d, se]) => {
      const v = f(h), cls = k.startsWith('rel:') ? (v < 1 ? 'gain' : v > 1.02 ? 'loss' : '') : '';
      return `<td class="num ${cls}">${k === 'k' ? v + (h.bad ? '*' : '') : fmt(v, d)}${se ? `<small> ±${fmt(2 * se(h), 2)}</small>` : ''}</td>`;
    }).join('')}</tr>`).join('');
  }
  table.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-k]');
    if (b) { const k = b.dataset.k; if (k === sortKey) dir = -dir; else { sortKey = k; dir = 1; } render(); return; }
    const tr = e.target.closest('tr[data-k]');
    if (tr) { sel = +tr.dataset.k; render(); onSelect(sel); }
  });
  return { update(s, m) { data = s; methods = m; render(); }, select(k) { sel = k; render(); } };
}
