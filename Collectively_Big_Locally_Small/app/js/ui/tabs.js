/**
 * ui/tabs.js
 * Responsibility: accessible tabs (roving tabindex, arrow keys), deep links through a bare #tab token,
 * and a per-viewer memory of the last tab (localStorage, guarded).
 */

/**
 * Initialize tabs.
 * @param {HTMLElement} list element with role="tablist"
 * @param {(name:string)=>void} onShow called after a tab becomes visible
 * @returns {{show:(name:string, focus?:boolean)=>void, current:()=>string}}
 */
export function initTabs(list, onShow) {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  let cur = tabs[0].dataset.tab;
  function show(name, focus) {
    const t = tabs.find((x) => x.dataset.tab === name) || tabs[0];
    for (const b of tabs) {
      const on = b === t;
      b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1;
      document.getElementById(b.getAttribute('aria-controls')).hidden = !on;
    }
    cur = t.dataset.tab;
    if (focus) t.focus();
    try { localStorage.setItem('ghb-lab-tab', cur); } catch (e) { /* storage unavailable */ }
    onShow(cur);
  }
  tabs.forEach((b, i) => {
    b.addEventListener('click', () => show(b.dataset.tab));
    b.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      show(tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length].dataset.tab, true);
    });
  });
  let start = location.hash.replace('#', '');
  if (!tabs.some((t) => t.dataset.tab === start)) { try { start = localStorage.getItem('ghb-lab-tab') || ''; } catch (e) { start = ''; } }
  return { show, current: () => cur, start: start || tabs[0].dataset.tab };
}
