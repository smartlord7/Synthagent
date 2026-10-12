/**
 * ui/panels/borrow.js
 * Responsibility: the Borrowing & metrics tab (Propositions 4.1 and 4.2, Section 5.2, Simulations 1 to 3). Shows the
 * progressive Monte Carlo of the current configuration for both protocols side by side, GHB (one joint release of
 * the network sum) and GHB-lite (per-hospital releases through gossip), with a protocol switch: summary chips,
 * relative risk per hospital, the effective-sample-size gain against 1 + B/n_k, absolute risk against n_k with
 * closed forms, a forest plot of all methods, and the sortable metrics table.
 */
import * as store from '../store.js';
import { controls, applyRelevance, progress, segmented } from '../controls.js';
import { add, chips, explain, pm } from '../dom.js';
import { Figure, methodSeries } from '../../viz/chart.js';
import { fmt } from '../../viz/theme.js';
import { METHOD, visibleMethods } from '../../sim/engine.js';
import { riskHB, riskFullPooling, looMSE, jointLooM } from '../../math/ghb.js';
import * as mainRun from '../mainRun.js';
import { createMetricsTable } from '../metricsTable.js';
import { renderInspector } from '../inspector.js';
import * as EQ from '../equations.js';

export function initBorrow() {
  const rail = document.getElementById('rail-borrow'), stage = document.getElementById('stage-borrow'), nb = document.getElementById('nb-borrow');
  rail.appendChild(segmented('protocol', 'borrow', 'Protocol shown'));
  controls(rail, ['tau2', 'sigma2', 'mu'], 'borrow', 'Model');
  controls(rail, ['eps', 'noDP', 'delta', 'C'], 'borrow', 'Privacy');
  controls(rail, ['cauchy', 'tauEst', 'weights', 'exact', 'T'], 'borrow', 'Estimation');
  controls(rail, ['R', 'chunk', 'seed'], 'borrow', 'Monte Carlo');
  const prog = add(stage, '<p class="rd-progress" id="borrow-progress" aria-live="polite"></p>');
  const chipEl = add(stage, '<div class="rd-chips" id="borrow-chips"></div>');
  const g1 = add(stage, '<div class="rd-grid2"></div>');
  const fRel = new Figure(add(g1, '<div></div>'), { id: 'fig-rel', num: 'B1', caption: 'Monte Carlo risk of each hospital relative to local estimation, R<sub>k</sub>/v<sub>k</sub> (dimensionless), hospitals sorted by size (x labels are n<sub>k</sub>, patients). Dark teal: GHB (joint release); light teal: GHB-lite (per-hospital releases). Below the dashed line the hospital gains. Error bars: ±2 standard errors.', xLabel: 'hospital size n_k (patients, sorted)', yLabel: 'relative risk R_k / v_k', height: 270, canLogX: false });
  const fGain = new Figure(add(g1, '<div></div>'), { id: 'fig-gain', num: 'B2', caption: 'Effective-sample-size gain n<sub>eff,k</sub>/n<sub>k</sub> = v<sub>k</sub>/R<sub>k</sub> (dimensionless). Dashed: Eq. (3), 1 + B<sub>k</sub>/n<sub>k</sub>, with M of GHB-lite\'s inverse-variance mean (grey) and of GHB\'s noisy sum (teal, known τ²). Every hospital receives about the same B, which is a large share for small hospitals.', xLabel: 'hospital size n_k (patients, sorted)', yLabel: 'n_eff,k / n_k', logY: true, height: 270, canLogX: false });
  const g2 = add(stage, '<div class="rd-grid2"></div>');
  const fRisk = new Figure(add(g2, '<div></div>'), { id: 'fig-risk-n', num: 'B3', caption: 'Absolute risk R<sub>k</sub> (squared error, parameter units²) against hospital size. Points: Monte Carlo; lines: closed forms of local estimation σ²/n, Proposition 4.1 with M of Eq. (4) (per-hospital releases) and with M of the joint release, and the single global model.', xLabel: 'hospital size n_k (patients)', yLabel: 'risk R_k', logX: true, logY: true, height: 270 });
  const fForest = new Figure(add(g2, '<div></div>'), { id: 'fig-forest', num: 'B4', caption: 'Forest plot of every method in the current scenario: mean relative risk over hospitals (dot, ±2 SE) and the worst hospital (ring). Left of 1, hospitals gain on average.', xLabel: 'relative risk R_k / v_k', yLabel: '', height: 270, canLogY: false });
  add(stage, '<h3 class="rd-h">Metrics per hospital (click a header to sort, a row to inspect; * marks a dishonest hospital)</h3>');
  const table = createMetricsTable(add(stage, '<div></div>'), 'borrow-table', (k) => { sel = k; side(); });

  explain(nb, 'What is happening', [
    'Each hospital estimates its parameter alone (variance v<sub>k</sub> = σ²/n<sub>k</sub>) or borrows from the others. Borrowing shrinks the local estimate toward a leave-one-out network estimate μ̃<sub>k</sub> with error M; the best weight is α<sub>k</sub> = (τ² + M)/(τ² + M + v<sub>k</sub>) (Proposition 4.1).',
    '<strong>GHB</strong> (Section 5.2): the network releases one noisy total S̃ = Σ n<sub>j</sub>θ̂<sub>j</sub> + noise, by secure summation; the noise is calibrated to the sum (sensitivity 2C), so it does not grow for small hospitals. Each hospital removes its own contribution exactly and, since no per-hospital value is released, integrates τ out over its half-Cauchy prior against its own evidence (Remark 4.4), with a quarter of the weight on not borrowing.',
    '<strong>GHB-lite</strong> (Section 5.3): every hospital releases its own privatized estimate (noise ∝ 1/n<sub>k</sub>); τ² and M are estimated from the releases through gossip. It learns τ from the spread of the releases, which GHB cannot, but its releases are noisy when hospitals are small or ε is small.'
  ]);
  const eqEl = add(nb, '<div></div>');
  const insp = add(nb, '<div class="rd-inspector" id="borrow-inspector"></div>');
  let sel = 0, last = null;

  function side() {
    const c = store.get();
    if (!last) return;
    const h = last.summary.perHospital[sel] || last.summary.perHospital[0], joint = c.protocol === 'ghb';
    renderInspector(insp, h, joint ? h.rel.ghb.y : h.rel.lite.y, joint ? 'ghb' : 'lite'); table.select(sel);
    const ctx = last.ctx;
    eqEl.innerHTML = EQ.model(c.mu, ctx.tau2m, c.sigma2, h.n)
      + (c.protocol !== 'lite' ? EQ.jointM(ctx.K, ctx.tau2m, c.sigma2, ctx.sJ, h.n, jointLooM(ctx.n, h.k, ctx.tau2m, c.sigma2, ctx.sJ ** 2)) : '')
      + (c.protocol !== 'ghb' ? EQ.looM(ctx.K, ctx.tau2m, c.sigma2, c.C, ctx.priv.delta, ctx.priv.eps, looMSE(ctx.n, h.k, ctx.q)) : '')
      + (joint ? EQ.borrowed(c.sigma2, ctx.tau2m, jointLooM(ctx.n, h.k, ctx.tau2m, c.sigma2, ctx.sJ ** 2), h.BJ, h.n) : EQ.borrowed(c.sigma2, ctx.tau2m, ctx.Mth[h.k], h.B, h.n));
  }
  function render(r) {
    last = r; const { ctx, summary: s } = r, c = store.get(), ids = visibleMethods(ctx.methods, c.protocol), order = ctx.n.map((n, k) => k).sort((a, b) => ctx.n[a] - ctx.n[b]);
    prog.innerHTML = progress(s.R, r.total) + (r.done ? '' : ' (running)');
    const A = s.agg, d = s.diagnostics, both = c.protocol === 'both';
    const items = [];
    if (c.protocol !== 'lite') items.push(['GHB mean R/v', pm(fmt(A.ghb.mean, 3), A.ghb.meanSE, fmt), 'ghb'], ['GHB worst', pm(fmt(A.ghb.worst, 3), A.ghb.worstSE, fmt), A.ghb.worst > 1.02 ? 'risk' : 'ghb'], ['GHB: share that gain', `${(100 * A.ghb.share).toFixed(0)}%`, 'ghb'], ['GHB: B (closed form)', fmt(d.BmeanJ, 1), 'ghb'], ['GHB: weight on not borrowing', `${(100 * d.wSafe).toFixed(0)}%`]);
    if (c.protocol !== 'ghb') items.push(['GHB-lite mean R/v', pm(fmt(A.lite.mean, 3), A.lite.meanSE, fmt), 'ghb'], ['GHB-lite worst', pm(fmt(A.lite.worst, 3), A.lite.worstSE, fmt), A.lite.worst > 1.02 ? 'risk' : 'ghb'], ['GHB-lite: share that gain', `${(100 * A.lite.share).toFixed(0)}%`, 'ghb'], ['GHB-lite: B (closed form)', fmt(d.Bmean, 1), 'ghb'], ['GHB-lite: mean τ̂²', fmt(d.t2hat, 4)]);
    if (both) items.push(['Better here', A.ghb.mean < A.lite.mean ? 'GHB' : 'GHB-lite', 'ghb']);
    items.push(['Oracle mean', fmt(A.oracle.mean, 3)], ['Pooled mean / worst', `${fmt(A.pooled.mean, 2)} / ${fmt(A.pooled.worst, 2)}`, 'ref'], ['True τ²', fmt(ctx.tau2m, 4)],
      ['Noise sd of the sum s', fmt(ctx.sJ, 3), 'noise'], ['Noise-to-signal s²/v (lite)', fmt(d.nsrMean, 3), 'noise'], ['Privacy per release', isFinite(ctx.priv.perEps) ? `ε ${fmt(ctx.priv.perEps, 3)}` : 'none', 'noise'],
      ['SLEM λ / rounds', `${fmt(ctx.lambda, 3)} / ${isFinite(ctx.rounds) ? Math.ceil(ctx.rounds) : '∞'}`, 'ref'], ['Consensus λ^T', c.exact ? 'exact' : fmt(ctx.lambda ** c.T, 3)]);
    chips(chipEl, items);
    const cats = order.map((k) => String(ctx.n[k]));
    fRel.setData({ xCats: cats, xName: 'n_k', yName: 'R/v', series: methodSeries(ids, METHOD, (id) => order.map((k, i) => [i, A[id].y[k], A[id].se[k]])), hlines: [{ y: 1, tok: '--rd-local', dash: true, label: 'local = 1' }] });
    const theory = [];
    if (c.protocol !== 'ghb') theory.push({ id: 'theory', label: '1 + B/n_k, GHB-lite (Eq. 3)', tok: '--rd-muted', dash: true, width: 1.3, pts: order.map((k, i) => [i, 1 + ctx.Bth[k] / ctx.n[k]]) });
    if (c.protocol !== 'lite') theory.push({ id: 'theoryJ', label: '1 + B/n_k, GHB (Eq. 3)', tok: '--rd-ghb', dash: true, width: 1.3, pts: order.map((k, i) => [i, 1 + ctx.BthJ[k] / ctx.n[k]]) });
    fGain.setData({ xCats: cats, xName: 'n_k', yName: 'n_eff/n', series: [...theory,
      ...methodSeries(ids.filter((m) => m !== 'local'), METHOD, (id) => order.map((k, i) => [i, 1 / A[id].y[k], A[id].se[k] / A[id].y[k] ** 2]))], hlines: [{ y: 1, tok: '--rd-local' }] });
    const nGrid = []; const lo = Math.max(2, Math.min(...ctx.n) / 2), hi = Math.max(...ctx.n) * 2;
    for (let i = 0; i <= 40; i++) nGrid.push(lo * (hi / lo) ** (i / 40));
    // M for a hospital outside the network that borrows from all K others (Eq. 4 with K others; the joint sum over all K)
    const Meq = looMSE(ctx.n, -1, ctx.q) * (ctx.K - 1) ** 2 / ctx.K ** 2, MeqJ = jointLooM(ctx.n, -1, ctx.tau2m, c.sigma2, ctx.sJ ** 2);
    const cf = [{ id: 'cf-local', label: 'σ²/n (local, closed form)', tok: '--rd-local', dash: true, width: 1.2, pts: nGrid.map((n) => [n, c.sigma2 / n]) }];
    if (c.protocol !== 'ghb') cf.push({ id: 'cf-hb', label: 'Prop. 4.1, per-hospital releases', tok: '--rd-lite', dash: true, width: 1.2, pts: nGrid.map((n) => [n, riskHB(n, ctx.tau2m, Meq, c.sigma2)]) });
    if (c.protocol !== 'lite') cf.push({ id: 'cf-hbJ', label: 'Prop. 4.1, joint release', tok: '--rd-ghb', dash: true, width: 1.2, pts: nGrid.map((n) => [n, riskHB(n, ctx.tau2m, MeqJ, c.sigma2)]) });
    cf.push({ id: 'cf-pool', label: 'global model (closed form)', tok: '--rd-ref', dash: true, width: 1.2, pts: order.map((k) => [ctx.n[k], riskFullPooling(ctx.n, k, ctx.tau2m, c.sigma2)]) });
    fRisk.setData({ xName: 'n_k', yName: 'R_k', series: [...cf, ...methodSeries(ids, METHOD, (id) => order.map((k) => [ctx.n[k], A[id].y[k] * ctx.v[k], A[id].se[k] * ctx.v[k]]), { noLine: true, marker: 3 })] });
    fForest.setData({ yCats: ids.map((id) => METHOD[id].label), xName: 'R/v', yName: 'method', series: [
      ...ids.map((id, i) => ({ id, label: METHOD[id].label, tok: METHOD[id].tok, noLine: true, marker: 4, xerr: true, pts: [[A[id].mean, i, A[id].meanSE]] })),
      { id: 'worst', label: 'worst hospital', tok: '--rd-risk', noLine: true, marker: 2.5, toggle: true, pts: ids.map((id, i) => [A[id].worst, i, 0, `hospital ${A[id].worstK + 1} (n = ${ctx.n[A[id].worstK]})`]) }],
      vlines: [{ x: 1, tok: '--rd-local' }] });
    table.update(s, ids, c.protocol);
    side();
  }
  mainRun.onResult(render);
  store.subscribe((keys) => { applyRelevance(document.getElementById('p-borrow'), store.get()); if (keys.includes('protocol') && last) render(last); });
  applyRelevance(document.getElementById('p-borrow'), store.get());
}
