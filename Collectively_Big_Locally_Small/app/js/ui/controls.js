/**
 * ui/controls.js
 * Responsibility: render parameters from the config schema as instrument controls (scale sliders with tick
 * marks and a unit readout, selects, switches) bound two-way to the store. The same parameter may appear in
 * several tabs; each instance has a stable id `<prefix>-<key>` and all instances stay in sync.
 */
import { PARAM } from '../config.js';
import * as store from './store.js';
import { fmt } from '../viz/theme.js';

const bound = new Map(); // key -> Set of update functions

/**
 * Readout text for a parameter value.
 * @param {Object} p schema entry
 * @param {*} v value
 * @returns {string} HTML
 */
export function readout(p, v) {
  if (p.unit === '%') return `${(100 * v).toFixed(0)}<span class="u">%</span>`;
  const s = p.int ? String(Math.round(v)) : fmt(v, p.step < 0.01 ? 4 : 3);
  return s + (p.unit ? `<span class="u">${p.unit}</span>` : '');
}

/**
 * Create a control element for one parameter.
 * @param {string} key configuration key
 * @param {string} prefix id prefix of this instance (e.g. the tab id)
 * @returns {HTMLElement}
 */
export function control(key, prefix) {
  const p = PARAM[key], id = `${prefix}-${key}`, d = document.createElement('div');
  if (!p) throw new Error('unknown parameter ' + key);
  d.dataset.key = key;
  if (p.type === 'check') {
    d.className = 'rd-check';
    d.innerHTML = `<input type="checkbox" id="${id}"><label for="${id}">${p.label}</label>`;
    const el = d.querySelector('input');
    el.addEventListener('change', () => store.set(key, el.checked));
    register(key, (v) => { el.checked = !!v; });
  } else if (p.type === 'select') {
    d.className = 'rd-ctl';
    d.innerHTML = `<label for="${id}">${p.label}</label><select id="${id}">${p.options.map((o) => `<option value="${o[0]}">${o[1]}</option>`).join('')}</select>`;
    const el = d.querySelector('select');
    el.addEventListener('change', () => store.set(key, el.value));
    register(key, (v) => { el.value = v; });
  } else {
    d.className = 'rd-ctl';
    d.innerHTML = `<label for="${id}"><span>${p.label}</span><output id="${id}-o" for="${id}"></output></label><div class="rd-scale"><input type="range" id="${id}" min="${p.min}" max="${p.max}" step="${p.step}"></div>`;
    const el = d.querySelector('input'), out = d.querySelector('output');
    const toVal = () => { const x = parseFloat(el.value); return p.log ? +(10 ** x).toPrecision(6) : p.int ? Math.round(x) : x; };
    el.addEventListener('input', () => { out.innerHTML = readout(p, toVal()); store.set(key, toVal()); });
    register(key, (v) => { el.value = p.log ? Math.log10(v) : v; out.innerHTML = readout(p, v); });
  }
  return d;
}
function register(key, fn) {
  if (!bound.has(key)) bound.set(key, new Set());
  bound.get(key).add(fn);
  fn(store.get()[key]);
}

store.subscribe((keys, c) => { for (const k of keys) bound.get(k)?.forEach((f) => f(c[k])); });

/**
 * Append controls for several keys to a container, optionally under a section heading.
 * @param {HTMLElement} box container
 * @param {string[]} keys parameter keys
 * @param {string} prefix id prefix
 * @param {string} [heading] section heading
 */
export function controls(box, keys, prefix, heading) {
  if (heading) { const h = document.createElement('h3'); h.className = 'rd-h'; h.textContent = heading; box.appendChild(h); }
  for (const k of keys) box.appendChild(control(k, prefix));
}

/**
 * Dim controls that have no effect under the current configuration (e.g. ν when effects are Gaussian).
 * @param {HTMLElement} root element containing controls
 * @param {Object} c configuration
 */
export function applyRelevance(root, c) {
  const off = {
    median: c.sizeMode !== 'lognormal', spread: c.sizeMode !== 'lognormal', nMin: c.sizeMode !== 'lognormal', nMax: c.sizeMode !== 'lognormal', sizeSeed: c.sizeMode !== 'lognormal',
    K: c.sizeMode === 'paper', hops: c.topo !== 'ring2', clusters: c.topo !== 'regional', pIn: c.topo !== 'regional', pER: c.topo !== 'random',
    graphSeed: !(c.topo === 'regional' || c.topo === 'random'), nu: c.effects !== 't', outK: c.effects !== 'outlier', outAt: c.effects !== 'outlier',
    regions: !c.regional, tauR2: !c.regional, tauW2: !c.regional, tau2: c.regional, eps: c.noDP, delta: c.noDP, C: c.noDP, cauchy: c.tauEst !== 'bayes',
    T: c.exact, optZ: !c.optOut, robustNu: !c.robustT, screenZ: !c.screen, anchorErr: !c.anchor, anchorSd: !c.anchor, anchorCheck: !c.anchor,
    badWhich: c.nBad === 0, lie: c.nBad === 0, beta: !c.gmrf, gamma: !c.gmrf
  };
  root.querySelectorAll('[data-key]').forEach((el) => el.classList.toggle('is-off', !!off[el.dataset.key]));
}

/**
 * Progress readout HTML.
 * @param {number} done completed replicates
 * @param {number} total total replicates
 * @param {string} [label] trailing text
 * @returns {string}
 */
export function progress(done, total, label = 'replicates') {
  const f = total ? Math.min(1, done / total) : 1;
  return `<span class="bar"><i style="width:${(100 * f).toFixed(0)}%"></i></span>${done.toLocaleString('en-US')} / ${total.toLocaleString('en-US')} ${label}`;
}
