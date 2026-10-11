/**
 * main.js
 * Responsibility: wiring only. Builds the panels, connects tabs, presets, header buttons and the guided
 * tour, and starts the shared Monte Carlo run. All logic lives in math/, sim/, viz/ and ui/.
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

const panels = {
  calc: initCalc(), setup: initSetup(), network: initNetwork(), borrow: initBorrow(), stress: initStress(),
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

tabs.show(tabs.start);
mainRun.restart();
