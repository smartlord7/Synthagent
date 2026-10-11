/**
 * ui/panels/borrow.js
 * Responsibility: the Borrowing & metrics tab (Propositions 8.1 and 8.2, Simulations 1 to 3). Shows the
 * progressive Monte Carlo of the current configuration: summary chips, relative risk per hospital, the
 * effective-sample-size gain against 1 + B/n_k, absolute risk against n_k with closed forms, a forest plot
 * of all methods, and the sortable metrics table.
 */
import * as store from '../store.js';
import { controls, applyRelevance, progress } from '../controls.js';
import { add, chips, explain, pm } from '../dom.js';
import { Figure, methodSeries } from '../../viz/chart.js';
import { fmt } from '../../viz/theme.js';
import { METHOD } from '../../sim/engine.js';
import { riskHB, riskFullPooling, looMSE } from '../../math/ghb.js';
import * as mainRun from '../mainRun.js';
import { createMetricsTable } from '../metricsTable.js';
import { renderInspector } from '../inspector.js';
import * as EQ from '../equations.js';

export function initBorrow() {
  const rail = document.getElementById('rail-borrow'), stage = document.getElementById('stage-borrow'), nb = document.getElementById('nb-borrow');
  controls(rail, ['tau2', 'sigma2', 'mu'], 'borrow', 'Model');
  controls(rail, ['eps', 'noDP', 'delta', 'C'], 'borrow', 'Privacy');
  controls(rail, ['tauEst', 'cauchy', 'weights', 'exact', 'T'], 'borrow', 'Estimation');
  controls(rail, ['R', 'chunk', 'seed'], 'borrow', 'Monte Carlo');
  const prog = add(stage, '<p class="rd-progress" id="borrow-progress" aria-live="polite"></p>');
  const chipEl = add(stage, '<div class="rd-chips" id="borrow-chips"></div>');
  const g1 = add(stage, '<div class="rd-grid2"></div>');
  const fRel = new Figure(add(g1, '<div></div>'), { id: 'fig-rel', num: 'B1', caption: 'Monte Carlo risk of each hospital relative to local estimation, R<sub>k</sub>/v<sub>k</sub> (dimensionless), hospitals sorted by size (x labels are n<sub>k</sub>, patients). Below the dashed line the hospital gains. Error bars: ±2 standard errors.', xLabel: 'hospital size n_k (patients, sorted)', yLabel: 'relative risk R_k / v_k', height: 270, canLogX: false });
  const fGain = new Figure(add(g1, '<div></div>'), { id: 'fig-gain', num: 'B2', caption: 'Effective-sample-size gain n<sub>eff,k</sub>/n<sub>k</sub> = v<sub>k</sub>/R<sub>k</sub> (dimensionless). Dashed: Eq. (5), 1 + B<sub>k</sub>/n<sub>k</sub>. Every hospital receives about the same B, which is a large share for small hospitals.', xLabel: 'hospital size n_k (patients, sorted)', yLabel: 'n_eff,k / n_k', logY: true, height: 270, canLogX: false });
  const g2 = add(stage, '<div class="rd-grid2"></div>');
  const fRisk = new Figure(add(g2, '<div></div>'), { id: 'fig-risk-n', num: 'B3', caption: 'Absolute risk R<sub>k</sub> (squared error, parameter units²) against hospital size. Points: Monte Carlo; lines: closed forms of local estimation σ²/n, Proposition 8.1 with M of Eq. (6), and the single global model.', xLabel: 'hospital size n_k (patients)', yLabel: 'risk R_k', logX: true, logY: true, height: 270 });
  const fForest = new Figure(add(g2, '<div></div>'), { id: 'fig-forest', num: 'B4', caption: 'Forest plot of every method in the current scenario: mean relative risk over hospitals (dot, ±2 SE) and the worst hospital (ring). Left of 1, hospitals gain on average.', xLabel: 'relative risk R_k / v_k', yLabel: '', height: 250, canLogY: false });
  add(stage, '<h3 class="rd-h">Metrics per hospital (click a header to sort, a row to inspect; * marks a dishonest hospital)</h3>');
  const table = createMetricsTable(add(stage, '<div></div>'), 'borrow-table', (k) => { sel = k; side(); });

  explain(nb, 'What is happening', [
    'Each hospital estimates its parameter alone (variance v<sub>k</sub> = σ²/n<sub>k</sub>) or borrows from the others. Borrowing shrinks the local estimate toward a leave-one-out network estimate μ̃<sub>k</sub> with error M; the best weight is α<sub>k</sub> = (τ² + M)/(τ² + M + v<sub>k</sub>).',
    'Plug-in GHB estimates τ² and M from privatized releases only, through gossip. The oracle knows τ²; the global pooled model ignores heterogeneity and hurts large hospitals when τ² is not tiny.'
  ]);
  const eqEl = add(nb, '<div></div>');
  const insp = add(nb, '<div class="rd-inspector" id="borrow-inspector"></div>');
  let sel = 0, last = null;

  function side() {
    const c = store.get();
    if (!last) return;
    const h = last.summary.perHospital[sel] || last.summary.perHospital[0];
    renderInspector(insp, h, h.rel.ghb.y); table.select(sel);
    eqEl.innerHTML = EQ.model(c.mu, last.ctx.tau2m, c.sigma2, h.n) + EQ.looM(last.ctx.K, last.ctx.tau2m, c.sigma2, c.C, last.ctx.priv.delta, last.ctx.priv.eps, looMSE(last.ctx.n, h.k, last.ctx.q)) + EQ.borrowed(c.sigma2, last.ctx.tau2m, last.ctx.Mth[h.k], h.B, h.n);
  }
  function render(r) {
    last = r; const { ctx, summary: s } = r, c = store.get(), ids = ctx.methods, order = ctx.n.map((n, k) => k).sort((a, b) => ctx.n[a] - ctx.n[b]);
    prog.innerHTML = progress(s.R, r.total) + (r.done ? '' : ' (running)');
    const A = s.agg;
    chips(chipEl, [
      ['GHB mean R/v', pm(fmt(A.ghb.mean, 3), A.ghb.meanSE, fmt), 'ghb'], ['GHB worst', pm(fmt(A.ghb.worst, 3), A.ghb.worstSE, fmt), A.ghb.worst > 1.02 ? 'risk' : 'ghb'],
      ['Share that gain', `${(100 * A.ghb.share).toFixed(0)}%`, 'ghb'], ['Oracle mean', fmt(A.oracle.mean, 3)], ['Pooled mean / worst', `${fmt(A.pooled.mean, 2)} / ${fmt(A.pooled.worst, 2)}`, 'ref'],
      ['B (virtual patients)', fmt(s.diagnostics.Bmean, 1), 'ghb'], ['Mean τ̂²', fmt(s.diagnostics.t2hat, 4)], ['True τ²', fmt(ctx.tau2m, 4)],
      ['Noise-to-signal s²/v', fmt(s.diagnostics.nsrMean, 3), 'noise'], ['Privacy per release', isFinite(ctx.priv.perEps) ? `ε ${fmt(ctx.priv.perEps, 3)}` : 'none', 'noise'],
      ['SLEM λ / rounds', `${fmt(ctx.lambda, 3)} / ${isFinite(ctx.rounds) ? Math.ceil(ctx.rounds) : '∞'}`, 'ref'], ['Consensus λ^T', c.exact ? 'exact' : fmt(ctx.lambda ** c.T, 3)]
    ]);
    const cats = order.map((k) => String(ctx.n[k]));
    fRel.setData({ xCats: cats, xName: 'n_k', yName: 'R/v', series: methodSeries(ids, METHOD, (id) => order.map((k, i) => [i, A[id].y[k], A[id].se[k]])), hlines: [{ y: 1, tok: '--rd-local', dash: true, label: 'local = 1' }] });
    fGain.setData({ xCats: cats, xName: 'n_k', yName: 'n_eff/n', series: [
      { id: 'theory', label: '1 + B/n_k (Eq. 5)', tok: '--rd-muted', dash: true, width: 1.3, pts: order.map((k, i) => [i, 1 + ctx.Bth[k] / ctx.n[k]]) },
      ...methodSeries(ids.filter((m) => m !== 'local'), METHOD, (id) => order.map((k, i) => [i, 1 / A[id].y[k], A[id].se[k] / A[id].y[k] ** 2]))], hlines: [{ y: 1, tok: '--rd-local' }] });
    const nGrid = []; const lo = Math.max(2, Math.min(...ctx.n) / 2), hi = Math.max(...ctx.n) * 2;
    for (let i = 0; i <= 40; i++) nGrid.push(lo * (hi / lo) ** (i / 40));
    // M for a hospital outside the network that borrows from all K others (Eq. 6 with K others)
    const Meq = looMSE(ctx.n, -1, ctx.q) * (ctx.K - 1) ** 2 / ctx.K ** 2;
    fRisk.setData({ xName: 'n_k', yName: 'R_k', series: [
      { id: 'cf-local', label: 'σ²/n (local, closed form)', tok: '--rd-local', dash: true, width: 1.2, pts: nGrid.map((n) => [n, c.sigma2 / n]) },
      { id: 'cf-hb', label: 'Prop. 8.1 (closed form)', tok: '--rd-oracle', dash: true, width: 1.2, pts: nGrid.map((n) => [n, riskHB(n, ctx.tau2m, Meq, c.sigma2)]) },
      { id: 'cf-pool', label: 'global model (closed form)', tok: '--rd-ref', dash: true, width: 1.2, pts: order.map((k) => [ctx.n[k], riskFullPooling(ctx.n, k, ctx.tau2m, c.sigma2)]) },
      ...methodSeries(ids, METHOD, (id) => order.map((k) => [ctx.n[k], A[id].y[k] * ctx.v[k], A[id].se[k] * ctx.v[k]]), { noLine: true, marker: 3 })] });
    fForest.setData({ yCats: ids.map((id) => METHOD[id].label), xName: 'R/v', yName: 'method', series: [
      ...ids.map((id, i) => ({ id, label: METHOD[id].label, tok: METHOD[id].tok, noLine: true, marker: 4, xerr: true, pts: [[A[id].mean, i, A[id].meanSE]] })),
      { id: 'worst', label: 'worst hospital', tok: '--rd-risk', noLine: true, marker: 2.5, toggle: true, pts: ids.map((id, i) => [A[id].worst, i, 0, `hospital ${A[id].worstK + 1} (n = ${ctx.n[A[id].worstK]})`]) }],
      vlines: [{ x: 1, tok: '--rd-local' }] });
    table.update(s, ids);
    side();
  }
  mainRun.onResult(render);
  store.subscribe(() => applyRelevance(document.getElementById('p-borrow'), store.get()));
  applyRelevance(document.getElementById('p-borrow'), store.get());
}
