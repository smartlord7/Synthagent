/**
 * ui/dom.js
 * Responsibility: small DOM helpers shared by the panels: element creation, summary chips and
 * notebook explanation blocks.
 */

/**
 * Create an element from an HTML string and append it.
 * @param {HTMLElement} parent container
 * @param {string} html markup with a single root element
 * @returns {HTMLElement} the new element
 */
export function add(parent, html) {
  const t = document.createElement('template'); t.innerHTML = html.trim();
  const el = t.content.firstElementChild; parent.appendChild(el); return el;
}

/**
 * Render summary chips.
 * @param {HTMLElement} el container (class rd-chips)
 * @param {Array<[string, string, string?]>} items [label, value HTML, optional class: ghb|risk|noise|ref|robust]
 */
export function chips(el, items) {
  el.innerHTML = items.map(([k, v, cls]) => `<div class="rd-chip ${cls || ''}"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
}

/**
 * Plain-language explanation block for the lab notebook.
 * @param {HTMLElement} parent notebook
 * @param {string} title heading
 * @param {string[]} paras paragraphs (HTML allowed)
 * @returns {HTMLElement}
 */
export function explain(parent, title, paras) {
  return add(parent, `<div><h3 class="rd-h">${title}</h3>${paras.map((p) => `<p class="rd-prose">${p}</p>`).join('')}</div>`);
}

/**
 * Value with Monte Carlo error.
 * @param {string} v formatted value
 * @param {number} se standard error
 * @param {(x:number,d:number)=>string} fmt formatter
 * @returns {string}
 */
export function pm(v, se, fmt) { return se > 0 ? `${v}<small> ±${fmt(2 * se, 2)}</small>` : v; }
