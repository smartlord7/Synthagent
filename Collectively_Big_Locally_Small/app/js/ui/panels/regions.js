/**
 * ui/panels/regions.js
 * Responsibility: the Regions tab (Eq. 5, Simulation 6). Runs its own Monte Carlo with regional site effects
 * and the GMRF prior switched on, and compares exchangeable GHB with the GMRF prior per hospital, over the
 * smoothing strength beta and over the between-region heterogeneity tau_r^2.
 */
import * as store from '../store.js';
import { controls, applyRelevance, progress } from '../controls.js';
import { add, chips, explain, pm } from '../dom.js';
import { Figure, methodSeries } from '../../viz/chart.js';
import { fmt } from '../../viz/theme.js';
import { METHOD, runMonteCarlo } from '../../sim/engine.js';
import { runJob } from '../../sim/scheduler.js';
import { PAPER_FIGURES } from '../../sim/sweep.js';
import { startSweep } from '../sweepFigure.js';
import * as EQ from '../equations.js';

export function initRegions() {
  const rail = document.getElementById('rail-regions'), stage = document.getElementById('stage-regions'), nb = document.getElementById('nb-regions');
  controls(rail, ['regions', 'tauR2', 'tauW2'], 'regions', 'Regional structure');
  controls(rail, ['beta', 'gamma', 'eps'], 'regions', 'GMRF prior');
  add(rail, '<p class="rd-prose">This tab always simulates regional effects with the GMRF prior on. Switch them on in Setup to use them in the other tabs too.</p>');
  const prog = add(stage, '<p class="rd-progress" id="regions-progress"></p>');
  const chipEl = add(stage, '<div class="rd-chips" id="regions-chips"></div>');
  const fPer = new Figure(add(stage, '<div></div>'), { id: 'fig-gmrf-per', num: 'G1', caption: 'Relative risk per hospital (sorted by size, labels n<sub>k</sub>) for exchangeable GHB-lite, GHB and the GMRF prior over the regional similarity graph. ±2 SE.', xLabel: 'hospital size n_k (patients, sorted)', yLabel: 'relative risk R_k / v_k', height: 250, canLogX: false });
  const g = add(stage, '<div class="rd-grid2"></div>');
  const bBeta = add(g, '<div></div>');
  const fBeta = new Figure(bBeta, { id: 'fig-gmrf-beta', num: 'G2', caption: 'Mean relative risk against the smoothing strength β (log scale) at the current τ<sub>r</sub>², with exchangeable GHB-lite for reference. ±2 SE.', xLabel: 'β', yLabel: 'mean R / v', height: 230 });
  const pBeta = add(bBeta, '<p class="rd-progress"></p>');
  const bTr = add(g, '<div></div>');
  const fTr = new Figure(bTr, { id: 'fig-gmrf-taur', num: 'G3', caption: 'Mean relative risk against the between-region heterogeneity τ<sub>r</sub>² at the current β (Simulation 6). The GMRF prior pays off when regions differ. ±2 SE.', xLabel: 'τ_r²', yLabel: 'mean R / v', height: 230 });
  const pTr = add(bTr, '<p class="rd-progress"></p>');
  add(stage, '<div class="rd-btns"><button class="rd-btn" type="button" id="regions-run">Run β and τ_r² sweeps</button></div>');
  explain(nb, 'Structured heterogeneity', [
    'The exchangeable model treats all other hospitals as equally relevant. When hospitals in the same region resemble each other, a Gaussian Markov random field prior over a similarity graph lets each hospital borrow mostly from similar ones.',
    'Neighbours run Jacobi sweeps on privatized releases; each hospital then applies the last update with its own noise-free estimate, so no extra privacy is spent. β sets how strongly neighbours are pulled together; with γ > 0 an anchor μ₀ is added (the anchor of the Stress tab, or 0).'
  ]);
  const eqEl = add(nb, '<div></div>');
  const cfg = () => ({ ...store.get(), regional: true, gmrf: true, R: Math.min(store.get().R, 800) });
  function run() {
    const c = cfg(), total = c.R;
    runJob('regions-mc', runMonteCarlo(c, { onProgress: (s, done, ctx) => {
      prog.innerHTML = progress(s.R, total);
      const A = s.agg, order = ctx.n.map((_, k) => k).sort((a, b) => ctx.n[a] - ctx.n[b]);
      chips(chipEl, [['Exchangeable GHB-lite', pm(fmt(A.lite.mean, 3), A.lite.meanSE, fmt), 'ghb'], ['GMRF', pm(fmt(A.gmrf.mean, 3), A.gmrf.meanSE, fmt), A.gmrf.mean < A.lite.mean ? 'ghb' : 'risk'],
        ['GHB (joint release)', pm(fmt(A.ghb.mean, 3), A.ghb.meanSE, fmt), 'ghb'], ['GHB-lite worst', fmt(A.lite.worst, 3)], ['GMRF worst', fmt(A.gmrf.worst, 3)], ['Regions', c.regions], ['τ_r² / τ_w²', `${fmt(c.tauR2, 4)} / ${fmt(c.tauW2, 4)}`]]);
      fPer.setData({ xCats: order.map((k) => String(ctx.n[k])), xName: 'n_k', yName: 'R/v', series: methodSeries(['local', 'lite', 'ghb', 'gmrf'], METHOD, (id) => order.map((k, i) => [i, A[id].y[k], A[id].se[k]])), hlines: [{ y: 1, tok: '--rd-local', dash: true }] });
    } }));
    eqEl.innerHTML = EQ.jacobi(c.beta, c.gamma);
  }
  function sweeps() {
    const c = cfg();
    startSweep('regions-beta', fBeta, { param: 'beta', start: 1, stop: 1e4, steps: 9, log: true, metrics: ['mean'], methods: ['lite', 'gmrf'], R: 250 }, c, pBeta, { vlines: [{ x: c.beta, tok: '--rd-ink' }] });
    startSweep('regions-taur', fTr, { ...PAPER_FIGURES.find((f) => f.id === 'gmrf'), R: 250 }, c, pTr);
  }
  document.getElementById('regions-run').onclick = sweeps;
  let started = false, t;
  store.subscribe(() => { applyRelevance(document.getElementById('p-regions'), { ...store.get(), regional: true, gmrf: true }); if (started) { clearTimeout(t); t = setTimeout(run, 300); } });
  applyRelevance(document.getElementById('p-regions'), { ...store.get(), regional: true, gmrf: true });
  return { onShow() { if (!started) { started = true; run(); sweeps(); } } };
}
