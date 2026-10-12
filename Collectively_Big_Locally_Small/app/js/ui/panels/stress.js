/**
 * ui/panels/stress.js
 * Responsibility: the Stress tests tab (Simulations 9, 11, 12; Section 6.4): atypical hospitals (relative risk against
 * |u_k|/tau for GHB, GHB-lite, the robust Student-t prior and opt-out), dishonest hospitals (honest risk against
 * the lie b: GHB, plain and screened GHB-lite) and a wrong knowledge anchor.
 */
import * as store from '../store.js';
import { controls, applyRelevance, progress, segmented } from '../controls.js';
import { add, chips, explain, pm } from '../dom.js';
import { Figure, methodSeries } from '../../viz/chart.js';
import { fmt } from '../../viz/theme.js';
import { METHOD, U_BINS, visibleMethods } from '../../sim/engine.js';
import { PAPER_FIGURES } from '../../sim/sweep.js';
import * as mainRun from '../mainRun.js';
import { startSweep } from '../sweepFigure.js';

export function initStress() {
  const rail = document.getElementById('rail-stress'), stage = document.getElementById('stage-stress'), nb = document.getElementById('nb-stress');
  rail.appendChild(segmented('protocol', 'stress', 'Protocol shown'));
  controls(rail, ['effects', 'nu', 'outK', 'outAt'], 'stress', 'Atypical hospitals');
  controls(rail, ['robustT', 'robustNu', 'optOut', 'optZ'], 'stress', 'Local safeguards');
  controls(rail, ['nBad', 'badWhich', 'lie', 'screen', 'screenZ'], 'stress', 'Dishonest hospitals');
  controls(rail, ['anchor', 'anchorErr', 'anchorSd', 'anchorCheck'], 'stress', 'Knowledge anchor');
  const prog = add(stage, '<p class="rd-progress" id="stress-progress"></p>');
  const chipEl = add(stage, '<div class="rd-chips" id="stress-chips"></div>');
  const fAtyp = new Figure(add(stage, '<div></div>'), { id: 'fig-atyp', num: 'S1', caption: 'Relative risk R/v (dimensionless) binned by the standardized site effect |u<sub>k</sub>|/τ (Simulation 9). Borrowing loses for hospitals far from the others; GHB\'s evidence weights and no-borrowing components, and GHB-lite\'s Student-t prior and opt-out rule, cap the loss at little cost elsewhere. Bins with fewer than 30 hospital-replicates are omitted. Error bars are not shown for binned means.', xLabel: '|u_k| / τ (standard deviations)', yLabel: 'relative risk R / v', height: 260 });
  const g = add(stage, '<div class="rd-grid2"></div>');
  const boxLie = add(g, '<div></div>');
  const fLie = new Figure(boxLie, { id: 'fig-lie', num: 'S2', caption: 'Mean relative risk of the honest hospitals against the size b of the lie (in units of τ) told by the dishonest hospitals (Simulation 12), for GHB (a lie enters its sum unseen), plain GHB-lite and screened GHB-lite. ±2 SE.', xLabel: 'lie b (τ)', yLabel: 'honest R / v', height: 240 });
  const lieProg = add(boxLie, '<p class="rd-progress" id="stress-lie-progress"></p>');
  add(boxLie, '<button class="rd-btn" type="button" id="stress-run-lie">Run lie sweep</button>');
  const boxAnc = add(g, '<div></div>');
  const fAnc = new Figure(boxAnc, { id: 'fig-anchor', num: 'S3', caption: 'Mean relative risk against the true error of a knowledge anchor μ₀ (Gaussian analogue of Simulation 11), with the stated uncertainty and conflict check set on the left. ±2 SE.', xLabel: 'anchor error (sd, parameter units)', yLabel: 'mean R / v', height: 240 });
  const ancProg = add(boxAnc, '<p class="rd-progress" id="stress-anchor-progress"></p>');
  add(boxAnc, '<button class="rd-btn" type="button" id="stress-run-anchor">Run anchor sweep</button>');

  explain(nb, 'Why stress-test', [
    'Propositions 4.1 and 4.2 bound the risk <em>on average over hospitals</em>. For a given site effect u<sub>k</sub>, the risk of borrowing is α²v<sub>k</sub> + (1 − α)²(M + u<sub>k</sub>²), which exceeds v<sub>k</sub> when |u<sub>k</sub>| is large compared with τ: a genuinely atypical hospital can lose.',
    'The robust prior replaces N(μ̃<sub>k</sub>, τ² + M) by a Student-t with the same variance, so a far-away local estimate is shrunk less. Opt-out keeps the local estimate when |θ̂<sub>k</sub> − μ̃<sub>k</sub>|/√(v<sub>k</sub> + τ̂² + M<sub>k</sub>) exceeds the threshold. Both are local computations and cost no privacy.',
    'A dishonest hospital releases (or, under GHB, contributes to the sum) μ + bτ instead of its estimate. GHB-lite\'s screening drops releases far from the inverse-variance weighted median (MAD spread) before DerSimonian-Laird and pooling. GHB has no per-hospital release in which to see a lie: its defence is each hospital\'s own evidence, which moves weight to the no-borrowing components when the network conflicts with its data (Section 8.2).'
  ]);
  const lieSpec = () => ({ ...PAPER_FIGURES.find((f) => f.id === 'lie') });
  const ancSpec = () => ({ ...PAPER_FIGURES.find((f) => f.id === 'anchor'), methods: ['ghb', 'lite', 'local'] });
  const runLie = () => { const c = store.get(); startSweep('stress-lie', fLie, lieSpec(), { ...c, nBad: Math.max(1, c.nBad || 3), screen: true }, lieProg); };
  const runAnc = () => { const c = store.get(); startSweep('stress-anchor', fAnc, ancSpec(), { ...c, anchor: true }, ancProg); };
  document.getElementById('stress-run-lie').onclick = runLie;
  document.getElementById('stress-run-anchor').onclick = runAnc;
  let started = false;

  function render(r) {
    const { ctx, summary: s } = r, A = s.agg, d = s.diagnostics, c = store.get(), ids = visibleMethods(ctx.methods.filter((m) => m !== 'local'), c.protocol);
    prog.innerHTML = progress(s.R, r.total);
    const items = [['GHB mean', pm(fmt(A.ghb.mean, 3), A.ghb.meanSE, fmt), 'ghb'], ['GHB-lite mean', pm(fmt(A.lite.mean, 3), A.lite.meanSE, fmt), 'ghb'], ['GHB: weight on not borrowing', `${(100 * d.wSafe).toFixed(0)}%`]];
    if (A.robust) items.push(['Robust t mean', pm(fmt(A.robust.mean, 3), A.robust.meanSE, fmt), 'robust']);
    if (A.optout) items.push(['Opt-out mean', pm(fmt(A.optout.mean, 3), A.optout.meanSE, fmt), 'noise']);
    if (store.get().effects === 'outlier') { const j = ctx.outIdx; ids.forEach((m) => items.push([`Outlier: ${METHOD[m].label}`, fmt(A[m].y[j], 2), A[m].y[j] > 1 ? 'risk' : ''])); }
    if (ctx.bad.length) { items.push(['Honest mean, GHB', pm(fmt(A.ghb.honest, 3), A.ghb.honestSE, fmt), A.ghb.honest > 1 ? 'risk' : 'ghb'], ['Honest mean, GHB-lite', pm(fmt(A.lite.honest, 3), A.lite.honestSE, fmt), A.lite.honest > 1 ? 'risk' : 'ghb']); if (A.screen) items.push(['Honest mean, screened', pm(fmt(A.screen.honest, 3), A.screen.honestSE, fmt), 'robust']); }
    if (d.keptBad != null) items.push(['Lies kept by screening', `${(100 * d.keptBad).toFixed(0)}%`, d.keptBad > 0.5 ? 'risk' : '']);
    if (d.droppedHonest != null) items.push(['Honest releases screened out', `${(100 * d.droppedHonest).toFixed(1)}%`]);
    if (d.anchorsDropped != null) items.push(['Anchors dropped by check', `${(100 * d.anchorsDropped).toFixed(0)}%`, 'noise']);
    chips(chipEl, items);
    const mid = U_BINS.slice(0, -1).map((b, i) => (b + U_BINS[i + 1]) / 2);
    fAtyp.setData({ xName: '|u|/τ', yName: 'R/v', series: methodSeries(ids, METHOD, (id) => mid.map((x, i) => [x, A[id].bins[i]])), hlines: [{ y: 1, tok: '--rd-local', dash: true, label: 'local = 1' }] });
  }
  mainRun.onResult(render);
  store.subscribe((keys) => { applyRelevance(document.getElementById('p-stress'), store.get()); if (keys.includes('protocol') && mainRun.latest()) render(mainRun.latest()); });
  applyRelevance(document.getElementById('p-stress'), store.get());
  return { onShow() { if (!started) { started = true; runLie(); runAnc(); } } };
}
