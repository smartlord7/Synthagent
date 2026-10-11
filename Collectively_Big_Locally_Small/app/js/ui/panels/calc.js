/**
 * ui/panels/calc.js
 * Responsibility: the Smallness tab (Section 2): effective sample size, effective events, Riley's n*,
 * the smallness index and its regime, with the deflation figure and live equations.
 */
import * as store from '../store.js';
import { controls } from '../controls.js';
import { add, chips, explain } from '../dom.js';
import { Figure } from '../../viz/chart.js';
import { drawDeflation } from '../../viz/diagrams.js';
import { neffParts, rileyN, regime, effectiveEvents, r2csMax } from '../../math/smallness.js';
import * as EQ from '../equations.js';
import { PARAM } from '../../config.js';

export function initCalc() {
  const rail = document.getElementById('rail-calc'), stage = document.getElementById('stage-calc'), nb = document.getElementById('nb-calc');
  controls(rail, ['cN', 'cM', 'cICC', 'cRhoL', 'cEta', 'cRhoS', 'cPrev'], 'calc', 'Dataset');
  controls(rail, ['cP', 'cR2', 'cS'], 'calc', 'Riley criterion');
  const chipEl = add(stage, '<div class="rd-chips" id="calc-chips"></div>');
  const host = add(stage, '<div></div>');
  const fig = new Figure(host, { id: 'fig-deflate', num: 'A1', caption: '', xLabel: '', yLabel: '', height: 250, canLogX: false, canLogY: false, toggles: false });
  explain(nb, 'What this measures', [
    'A dataset is <em>small for a task</em> when the information it holds falls short of what the task needs. The effective sample size n<sub>eff</sub> counts how many clean, independent, correctly labeled records the data are worth; Riley\'s first criterion gives the size n* a clinical prediction model needs.',
    'The defaults reproduce the ICU example of Section 2: 2,000 stays become about 414 effective records, against 2,707 needed, so σ ≈ 6.5 and the data are strained.'
  ]);
  const eqEl = add(nb, '<div></div>');
  function update() {
    const c = store.get(), P = neffParts(c.cN, c.cM, c.cICC, c.cRhoL, c.cEta, c.cRhoS), nstar = rileyN(c.cP, c.cR2, c.cPrev, c.cS), s = nstar / P.neff, rg = regime(s);
    chips(chipEl, [['n_eff', P.neff.toFixed(0), 'ghb'], ['Effective events', effectiveEvents(P.neff, c.cPrev).toFixed(1)], ['Riley n*', Math.ceil(nstar).toLocaleString('en-US'), 'ref'],
      ['Smallness σ', s.toFixed(2), rg.level === 'good' ? 'ghb' : 'risk'], ['Regime', `<span class="rd-tag lv-${rg.level}">${rg.name}</span>`], ['DEFF', P.deff.toFixed(2)], ['max R²_CS', r2csMax(c.cPrev).toFixed(3)]]);
    fig.setCaption(`Deflation of a nominal sample of ${c.cN.toLocaleString('en-US')} records into an effective sample size (Eq. 2). Dashed outlines show the information lost to each mechanism; the dashed line is Riley's n* for a ${c.cP}-parameter model at ${(100 * c.cPrev).toFixed(1)}% prevalence (Eq. 1). Units: records.`);
    fig.setCustom((ctx, w, h) => drawDeflation(ctx, w, h, P.steps, nstar));
    eqEl.innerHTML = EQ.neff(c.cN, P.deff, c.cRhoL, c.cEta, c.cRhoS, P.neff) + EQ.riley(c.cP, c.cR2 * r2csMax(c.cPrev), c.cS, nstar) + EQ.smallness(nstar, P.neff, s);
  }
  store.subscribe((keys) => { if (keys.some((k) => PARAM[k]?.group === 'calc')) update(); });
  update();
}
