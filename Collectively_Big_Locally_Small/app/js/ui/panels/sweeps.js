/**
 * ui/panels/sweeps.js
 * Responsibility: the Sweeps & figures tab. A sweep builder (any sweepable parameter, linear or log grid,
 * any metrics and methods, replicates) and a gallery that reproduces each simulation figure of the paper on
 * the current configuration.
 */
import * as store from '../store.js';
import { PARAMS, PARAM } from '../../config.js';
import { add, explain } from '../dom.js';
import { Figure } from '../../viz/chart.js';
import { METRICS, PAPER_FIGURES } from '../../sim/sweep.js';
import { METHODS } from '../../sim/engine.js';
import { startSweep } from '../sweepFigure.js';

export function initSweeps() {
  const rail = document.getElementById('rail-sweeps'), stage = document.getElementById('stage-sweeps'), nb = document.getElementById('nb-sweeps');
  const sweepable = PARAMS.filter((p) => p.sweep);
  add(rail, '<h3 class="rd-h">Sweep builder</h3>');
  add(rail, `<div class="rd-ctl"><label for="sw-param">Parameter</label><select id="sw-param">${sweepable.map((p) => `<option value="${p.key}">${p.label}</option>`).join('')}</select></div>`);
  add(rail, '<div class="rd-btns"><div class="rd-ctl" style="flex:1"><label for="sw-start">Start</label><input class="rd-input mono" id="sw-start" type="number" step="any"></div><div class="rd-ctl" style="flex:1"><label for="sw-stop">Stop</label><input class="rd-input mono" id="sw-stop" type="number" step="any"></div><div class="rd-ctl" style="flex:0 0 70px"><label for="sw-steps">Steps</label><input class="rd-input mono" id="sw-steps" type="number" min="2" max="30" value="8"></div></div>');
  add(rail, '<div class="rd-check"><input type="checkbox" id="sw-log"><label for="sw-log">Log spacing</label></div>');
  add(rail, '<h3 class="rd-h">Metrics</h3>');
  const mBox = add(rail, '<div></div>');
  METRICS.forEach((m) => add(mBox, `<div class="rd-check"><input type="checkbox" id="sw-m-${m.key}" ${m.key === 'mean' ? 'checked' : ''}><label for="sw-m-${m.key}">${m.label}${m.kind === 'closed' ? ' (closed form)' : ''}</label></div>`));
  add(rail, '<h3 class="rd-h">Methods (simulated metrics)</h3>');
  const meBox = add(rail, '<div></div>');
  METHODS.forEach((m) => add(meBox, `<div class="rd-check"><input type="checkbox" id="sw-meth-${m.id}" ${['ghb', 'lite', 'oracle'].includes(m.id) ? 'checked' : ''}><label for="sw-meth-${m.id}" style="color:var(${m.tok})">${m.label}</label></div>`));
  add(rail, '<div class="rd-ctl"><label for="sw-R">Replicates per value</label><input class="rd-input mono" id="sw-R" type="number" min="50" max="20000" step="50" value="500"></div>');
  add(rail, '<button class="rd-btn primary" type="button" id="sw-run">Run sweep</button>');

  const prog = add(stage, '<p class="rd-progress" id="sweep-progress" aria-live="polite"></p>');
  const fig = new Figure(add(stage, '<div></div>'), { id: 'fig-sweep', num: 'W1', caption: 'Sweep of the selected parameter on the current configuration; Monte Carlo metrics carry ±2 SE bars.', xLabel: '', yLabel: '', height: 320 });
  add(stage, '<h2 class="rd-h2">Figures of the paper</h2>');
  const gal = add(stage, '<div class="rd-btns" id="sweep-gallery"></div>');
  PAPER_FIGURES.forEach((f) => add(gal, `<button class="rd-btn" type="button" id="sw-fig-${f.id}" title="${f.sim}">${f.sim}: ${f.title}</button>`));
  add(stage, '<p class="rd-prose">Other paper figures live where they belong: risk against n<sub>k</sub>, n<sub>eff</sub> gain and the forest plot on Borrowing &amp; metrics; consensus error against rounds on Network &amp; gossip; relative risk against |u<sub>k</sub>|/τ on Stress tests; B against refits, B against ε and the ESS heatmap on Privacy; GMRF against τ<sub>r</sub>² on Regions.</p>');
  explain(nb, 'How sweeps work', [
    'A sweep reruns the Monte Carlo of the current configuration for each value of one parameter, with everything else held fixed. Simulated metrics carry ±2 standard-error bars from the replicates; closed-form metrics need no simulation.',
    'Some parameters switch on what they need: sweeping the anchor error turns the anchor on, sweeping τ<sub>r</sub>² turns regional effects on, and choosing the GMRF or screened method enables it for the sweep. Sweeping K with the paper size profile draws log-normal sizes for each K.'
  ]);

  const $ = (id) => document.getElementById(id);
  function fillRange() {
    const p = PARAM[$('sw-param').value], lo = p.log ? 10 ** p.min : p.min, hi = p.log ? 10 ** p.max : p.max;
    $('sw-start').value = +lo.toPrecision(3); $('sw-stop').value = +hi.toPrecision(3); $('sw-log').checked = !!p.log;
  }
  $('sw-param').addEventListener('change', fillRange);
  function run(spec) {
    startSweep('sweep', fig, spec, store.get(), prog);
    fig.setCaption(`${spec.title ? `${spec.sim}: ${spec.title}. ` : ''}Sweep of ${PARAM[spec.param].label} from ${spec.start} to ${spec.stop} (${spec.steps} values, ${spec.log ? 'log' : 'linear'} spacing) on the current configuration; Monte Carlo metrics carry ±2 SE bars.`);
  }
  $('sw-run').onclick = () => {
    const metrics = METRICS.filter((m) => $('sw-m-' + m.key).checked).map((m) => m.key), methods = METHODS.filter((m) => $('sw-meth-' + m.id).checked).map((m) => m.id);
    if (!metrics.length) { prog.textContent = 'Choose at least one metric.'; return; }
    const start = +$('sw-start').value, stop = +$('sw-stop').value, log = $('sw-log').checked;
    if (log && (start <= 0 || stop <= 0)) { prog.textContent = 'Log spacing needs positive start and stop values.'; return; }
    run({ param: $('sw-param').value, start, stop, steps: Math.max(2, Math.min(30, +$('sw-steps').value || 8)), log, metrics, methods: methods.length ? methods : ['ghb', 'lite'], R: Math.max(50, +$('sw-R').value || 500) });
  };
  function load(id) {
    const f = PAPER_FIGURES.find((x) => x.id === id); if (!f) return;
    $('sw-param').value = f.param; $('sw-start').value = f.start; $('sw-stop').value = f.stop; $('sw-steps').value = f.steps; $('sw-log').checked = f.log;
    METRICS.forEach((m) => { $('sw-m-' + m.key).checked = f.metrics.includes(m.key); });
    METHODS.forEach((m) => { $('sw-meth-' + m.id).checked = f.methods.includes(m.id); });
    if (f.R) $('sw-R').value = f.R;
    run(f);
  }
  PAPER_FIGURES.forEach((f) => { $('sw-fig-' + f.id).onclick = () => load(f.id); });
  fillRange();
  let started = false;
  return { load, onShow() { if (!started) { started = true; load('tau2'); } } };
}
