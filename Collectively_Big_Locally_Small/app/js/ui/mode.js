/**
 * ui/mode.js
 * Responsibility: the Simple mode / Expert mode switch in the header. The mode is the `data-mode` attribute of the
 * root element (CSS hides `.expert-only` or `.simple-only` content accordingly), remembered per viewer in
 * localStorage (guarded: the page works without storage). First-time visitors get Simple mode; a deep link to an
 * expert tab (#arena, #setup, ...) opens Expert mode, and #simple opens Simple mode. index.html applies the same
 * rule in a tiny inline script before first paint, so the page does not flash the wrong mode.
 */

export const MODE_KEY = 'ghb-lab-mode';

/**
 * Initial mode from the address and the viewer's stored choice.
 * @param {string[]} expertTabs names of the expert tabs (deep-link targets)
 * @returns {'simple'|'expert'}
 */
export function initialMode(expertTabs) {
  const h = location.hash.replace('#', '');
  if (h === 'simple') return 'simple';
  if (expertTabs.includes(h) || h === 'expert') return 'expert';
  let m = null;
  try { m = localStorage.getItem(MODE_KEY); } catch (e) { m = null; }
  return m === 'expert' ? 'expert' : 'simple';
}

/**
 * Wire the header switch.
 * @param {HTMLElement} group element holding buttons with data-mode="simple" and data-mode="expert"
 * @param {'simple'|'expert'} start initial mode
 * @param {(mode:string)=>void} onChange called after every switch, and once at start
 * @returns {{set:(mode:string, focus?:boolean)=>void, get:()=>string}}
 */
export function initMode(group, start, onChange) {
  const btns = [...group.querySelectorAll('button[data-mode]')];
  let cur = null;
  function set(mode, persist = true) {
    cur = mode === 'expert' ? 'expert' : 'simple';
    document.documentElement.setAttribute('data-mode', cur);
    for (const b of btns) b.setAttribute('aria-pressed', String(b.dataset.mode === cur));
    if (persist) { try { localStorage.setItem(MODE_KEY, cur); } catch (e) { /* storage unavailable */ } }
    onChange(cur);
  }
  btns.forEach((b) => b.addEventListener('click', () => { if (b.dataset.mode !== cur) { set(b.dataset.mode); window.scrollTo(0, 0); } }));
  set(start, false);
  return { set, get: () => cur };
}
