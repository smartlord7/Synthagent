/**
 * main.js
 * Responsibility: wiring only. Builds the panels, connects tabs, presets, header buttons, the Start here demos,
 * the guided tour and the Simple mode / Expert mode switch, and starts the shared Monte Carlo run the first time the
 * expert view is shown. All logic lives in math/, sim/, viz/ and ui/.
 */
import * as store from './ui/store.js';
import * as mainRun from './ui/mainRun.js';
import { initTabs } from './ui/tabs.js';
import { createTour } from './ui/tour.js';
import { SCENARIOS, scenarioConfig } from './scenarios.js';
import { randomConfig } from './config.js';
import { initCalc } from './ui/panels/calc.js';
import { initSetup } from './ui/panels/setup.js';
import { initNetwork } from './ui/panels/network.js';
import { initBorrow } from './ui/panels/borrow.js';
import { initStress } from './ui/panels/stress.js';
import { initPrivacy } from './ui/panels/privacy.js';
import { initRegions } from './ui/panels/regions.js';
import { initSweeps } from './ui/panels/sweeps.js';
import { initAbout } from './ui/panels/about.js';
import { initStart } from './ui/panels/start.js';
import { initArena } from './ui/panels/arena.js';
import { initReal } from './ui/panels/real.js';
import { initSimple } from './ui/simple.js';
import { initMode, initialMode } from './ui/mode.js';

const panels = {
  calc: initCalc(), setup: initSetup(), network: initNetwork(), borrow: initBorrow(), arena: initArena(), real: initReal(), stress: initStress(),
  privacy: initPrivacy(), regions: initRegions(), sweeps: initSweeps(), about: initAbout()
};
const tabs = initTabs(document.querySelector('[role="tablist"]'), (name) => {
  panels[name]?.onShow?.();
  requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
});

// scenarios and header actions
const sel = document.getElementById('scenario'), note = document.getElementById('scenario-note');
sel.innerHTML = SCENARIOS.map((s) => `<option value="${s.id}">${s.title}</option>`).join('');
function applyScenario(id, open = true) {
  const s = SCENARIOS.find((x) => x.id === id);
  store.replace(scenarioConfig(id));
  note.textContent = `${s.title}. ${s.question}${s.note ? ' ' + s.note : ''}`;
  if (open) { tabs.show(s.tab); if (s.figure && s.tab === 'sweeps') panels.sweeps.load(s.figure); }
}
sel.addEventListener('change', () => applyScenario(sel.value));
document.getElementById('btn-defaults').onclick = () => { sel.value = 'paper'; applyScenario('paper', false); };
let rs = 1;
document.getElementById('btn-random').onclick = () => { store.replace(randomConfig(store.get(), 1000 + rs++)); note.textContent = 'Random configuration: every tab now uses it. Paper defaults restores the paper\'s design.'; };
const tour = createTour(tabs, (a) => panels.network.action?.(a));
document.getElementById('btn-tour').onclick = () => tour.start();

// Start here: one-click demos
panels.start = initStart({
  demo(id) {
    if (id === 'ghb-vs-lite' || id === 'lite-wins') { sel.value = id === 'ghb-vs-lite' ? 'ghbVsLite' : 'liteWins'; applyScenario(sel.value); }
    else if (id === 'arena-all') { tabs.show('arena'); panels.arena.runPreset('all', true); }
    else if (id === 'real') tabs.show('real');
  },
  tour: () => tour.start(),
  open: (tab) => tabs.show(tab),
  simple: () => mode.set('simple')
});

tabs.show(tabs.start);

// Simple mode / Expert mode. The expert Monte Carlo starts the first time the expert view is shown.
let expertStarted = false;
initSimple(document.getElementById('simple-view'), { expert: () => { mode.set('expert'); window.scrollTo(0, 0); document.querySelector('#mode-switch [data-mode="expert"]').focus(); } });
const expertTabs = [...document.querySelectorAll('[role="tab"]')].map((b) => b.dataset.tab);
const mode = initMode(document.getElementById('mode-switch'), initialMode(expertTabs), (m) => {
  if (m === 'expert') {
    if (!expertStarted) { expertStarted = true; mainRun.restart(); }
    tabs.show(tabs.current());
  } else tour.stop();
});
