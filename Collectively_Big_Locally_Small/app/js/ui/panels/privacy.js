/**
 * ui/panels/privacy.js
 * Responsibility: the Privacy tab (Propositions 4.2 and 5.1, Section 5.2, Simulations 5 and 14), all in closed form:
 * noise per release and borrowed patients after r refits under basic composition and zCDP, borrowed patients against
 * epsilon for GHB-lite's individual releases, joint release in groups, and GHB's single release of the network sum,
 * the cost of collusion, and a heatmap of the effective-sample-size gain over (tau^2, eps).
 */
import * as store from '../store.js';
import { controls, applyRelevance } from '../controls.js';
import { add, chips, explain } from '../dom.js';
import { Figure } from '../../viz/chart.js';
import { fmt } from '../../viz/theme.js';
import { resolveSizes, resolvePrivacy } from '../../config.js';
import { sensitivity, noiseBasic, noiseZcdp, zcdpRho, dpVar, jointNoiseSd, collusionRho, zcdpToEps } from '../../math/dp.js';
import { meanBorrowed, releaseUnits, jointReleaseB, looMSE, borrowedPatients, median, jointBorrowed } from '../../math/ghb.js';
import * as EQ from '../equations.js';

export function initPrivacy() {
  const rail = document.getElementById('rail-privacy'), stage = document.getElementById('stage-privacy'), nb = document.getElementById('nb-privacy');
  controls(rail, ['eps', 'noDP', 'delta', 'C', 'accounting', 'refits', 'group', 'collude'], 'privacy', 'Privacy');
  controls(rail, ['tau2', 'sigma2', 'refN'], 'privacy', 'Model and reference hospital');
  const chipEl = add(stage, '<div class="rd-chips" id="privacy-chips"></div>');
  const g1 = add(stage, '<div class="rd-grid2"></div>');
  const fNoise = new Figure(add(g1, '<div></div>'), { id: 'fig-noise', num: 'P1', caption: '', xLabel: 'refits r on the same patients', yLabel: 'noise sd per release (parameter units)', logX: true, logY: true, height: 240 });
  const fBr = new Figure(add(g1, '<div></div>'), { id: 'fig-b-refits', num: 'P2', caption: 'Borrowed patients B (Eq. 3, inverse-variance pooling, mean over hospitals of the current size profile) after r refits on the same patients, for a fixed total budget (Simulation 14). GHB-lite under basic composition and zCDP; GHB (joint release, always zCDP). Dashed: the no-privacy limit σ²/(τ² + M).', xLabel: 'refits r', yLabel: 'B (patients)', logX: true, logY: true, height: 240 });
  const g2 = add(stage, '<div class="rd-grid2"></div>');
  const fJoint = new Figure(add(g2, '<div></div>'), { id: 'fig-joint', num: 'P3', caption: 'Borrowed patients B against the privacy budget ε when the hospitals at or below the median size release jointly in groups of g through secure aggregation (Simulation 5), and for GHB, where the whole network releases one noisy sum (Section 5.2). Dashed: the limit σ²/τ².', xLabel: 'privacy budget ε', yLabel: 'B (patients)', logX: true, logY: true, height: 240 });
  const fHeat = new Figure(add(g2, '<div></div>'), { id: 'fig-ess-heat', num: 'P4', caption: '', xLabel: 'heterogeneity τ²', yLabel: 'privacy budget ε', height: 240, canLogX: false, canLogY: false, toggles: false });

  explain(nb, 'Where privacy costs', [
    'A release is the hospital\'s estimate plus Gaussian noise with sd s<sub>k</sub> = Δ<sub>k</sub>√(2 ln(1.25/δ))/ε, where Δ<sub>k</sub> = 2C/n<sub>k</sub>. The noise enters only the network estimate, through M, and it is divided by (K − 1). It scales as 1/n<sub>k</sub>², so it is expensive only for tiny hospitals, which can release jointly.',
    'GHB goes further than groups: the whole network releases one sum, by secure summation over the agreement graph, so the noise is calibrated to one patient (sensitivity 2C) and divided by the total size N. The price is trust: if c hospitals collude and remove their noise shares, the others\' patients are protected at ρK/(K − c) instead of ρ (Proposition 5.1).',
    'Refitting on the same patients composes. Basic composition splits (ε, δ) into r equal parts; zero-concentrated DP adds ρ = Δ²/(2s²) over releases and converts once at the end, which needs much less noise for many refits.'
  ]);
  const eqEl = add(nb, '<div></div>');

  function update() {
    const c = store.get(), n = resolveSizes(c), K = n.length, priv = resolvePrivacy(c), D = sensitivity(c.refN, c.C), rs = [];
    for (let r = 1; r <= 50; r++) rs.push(r);
    const s2fn = (kind, r) => (nn) => (c.noDP ? 0 : (kind === 'zcdp' ? noiseZcdp(sensitivity(nn, c.C), c.eps, c.delta, r) : noiseBasic(sensitivity(nn, c.C), c.eps, c.delta, r)) ** 2);
    const Bof = (kind, r) => meanBorrowed(n, c.tau2, s2fn(kind, r), c.sigma2), Bfree = meanBorrowed(n, c.tau2, () => 0, c.sigma2);
    const BJ = (e, r = 1) => { const sj = c.noDP ? 0 : jointNoiseSd(c.C, e, c.delta, r); let b = 0; for (let k = 0; k < K; k++) b += jointBorrowed(n, k, c.tau2, c.sigma2, sj * sj); return b / K; };
    const sB = (r) => noiseBasic(D, c.eps, c.delta, r), sZ = (r) => noiseZcdp(D, c.eps, c.delta, r), r0 = c.refits;
    fNoise.setCaption(`Noise sd of one release for the reference hospital (n = ${c.refN}, Δ = 2C/n = ${fmt(D, 4)}) when a total budget ε = ${fmt(c.eps, 3)}, δ = ${fmt(c.delta)} is spread over r refits. Markers: current r.`);
    fNoise.setData({ xName: 'r', yName: 's', series: [
      { id: 'basic', label: 'basic composition', tok: '--rd-noise', pts: rs.map((r) => [r, sB(r)]) }, { id: 'zcdp', label: 'zCDP', tok: '--rd-ghb', pts: rs.map((r) => [r, sZ(r)]) }],
    marks: [{ x: r0, y: sB(r0), tok: '--rd-noise', label: 'basic, current r' }, { x: r0, y: sZ(r0), tok: '--rd-ghb', label: 'zCDP, current r' }] });
    fBr.setData({ xName: 'r', yName: 'B', series: [
      { id: 'basic', label: 'GHB-lite, basic composition', tok: '--rd-noise', pts: rs.map((r) => [r, Bof('basic', r)]) }, { id: 'zcdp', label: 'GHB-lite, zCDP', tok: '--rd-lite', pts: rs.map((r) => [r, Bof('zcdp', r)]) },
      { id: 'joint', label: 'GHB (joint release)', tok: '--rd-ghb', width: 2.2, pts: rs.map((r) => [r, BJ(c.eps, r)]) }],
    hlines: [{ y: Bfree, tok: '--rd-muted', dash: true, label: 'no privacy' }] });
    const E = []; for (let i = 0; i <= 40; i++) E.push(10 ** (-1.5 + 2.5 * i / 40));
    const cut = median(n), toks = { 1: '--rd-risk', 2: '--rd-noise', 4: '--rd-ref', 8: '--rd-lite' };
    const gs = [...new Set([1, 2, 4, 8, c.group])].sort((a, b) => a - b);
    fJoint.setData({ xName: 'ε', yName: 'B', series: [...gs.map((g) => ({ id: 'g' + g, label: g === 1 ? 'GHB-lite, individual release' : `GHB-lite, groups of ${g}`, tok: toks[g] || '--rd-robust', width: g === c.group ? 2 : 1.3,
      pts: E.map((e) => [e, jointReleaseB(releaseUnits(n, g, cut), { tau2: c.tau2, sigma2: c.sigma2, eps: e, delta: c.delta, C: c.C })]) })),
      { id: 'ghbsum', label: 'GHB, one release of the network sum', tok: '--rd-ghb', width: 2.4, pts: E.map((e) => [e, BJ(e)]) }],
    hlines: [{ y: c.sigma2 / c.tau2, tok: '--rd-muted', dash: true, label: 'limit σ²/τ²' }], vlines: [{ x: c.eps, tok: '--rd-ink' }] });
    // ESS heatmap (Figure 1 right): n_eff/n of the reference hospital borrowing from the current network
    const T2 = [], EP = []; for (let i = 0; i < 22; i++) T2.push(10 ** (-4 + 4 * i / 21)); for (let j = 0; j < 18; j++) EP.push(10 ** (-1.5 + 2.5 * j / 17));
    const refs = n.concat([c.refN]);
    const z = EP.map((e) => T2.map((t2) => { const q = { tau2: t2, sigma2: c.sigma2, eps: e, delta: c.delta, C: c.C }; return 1 + borrowedPatients(t2, looMSE(refs, refs.length - 1, q), c.sigma2) / c.refN; }));
    fHeat.setCaption(`Gain in effective sample size n<sub>eff</sub>/n for a hospital of ${c.refN} patients borrowing from the current ${K} hospitals, over heterogeneity τ² and budget ε (Eq. 3 with M of Eq. 4; colour on a log scale; hover for values).`);
    fHeat.setHeatmap({ xs: T2, ys: EP, z, xName: 'τ²', yName: 'ε', zLabel: 'n_eff / n', logZ: true });
    const rho = zcdpRho(c.eps, c.delta), sNow = Math.sqrt(dpVar(c.refN, priv.eps, priv.delta, priv.C));
    const cc = Math.min(c.collude, K - 1), rhoC = collusionRho(rho, K, cc), epsC = zcdpToEps(rhoC, c.delta), sJ = c.noDP ? 0 : jointNoiseSd(c.C, c.eps, c.delta, r0);
    chips(chipEl, [
      ['Total budget', c.noDP ? 'none' : `ε ${fmt(c.eps, 3)}, δ ${fmt(c.delta)}`, 'noise'], ['Per release (basic)', `ε ${fmt(c.eps / r0, 3)}, δ ${fmt(c.delta / r0)}`],
      ['zCDP ρ total / release', `${fmt(rho, 4)} / ${fmt(rho / r0, 4)}`], ['s at n = ' + c.refN + ' (' + c.accounting + ')', fmt(sNow, 4), 'noise'],
      ['Noise-to-signal s²/v', fmt(sNow ** 2 / (c.sigma2 / c.refN), 3)], ['B now', fmt(Bof(c.accounting, r0), 1), 'ghb'], ['B without privacy', fmt(Bfree, 1)],
      ['GHB: noise sd of the sum', fmt(sJ, 3), 'noise'], ['GHB: B now', fmt(BJ(c.eps, r0), 1), 'ghb'], [`GHB with ${cc} colluders: ε for the others`, c.noDP ? 'none' : fmt(epsC, 3), cc > 0 ? 'risk' : ''],
      ['B joint (g = ' + c.group + ')', fmt(jointReleaseB(releaseUnits(n, c.group, cut), { tau2: c.tau2, sigma2: c.sigma2, eps: priv.eps, delta: priv.delta, C: c.C }), 1), 'ghb']
    ]);
    eqEl.innerHTML = EQ.mechanism(c.refN, c.C, priv.delta, priv.eps, sNow) + EQ.jointM(K, c.tau2, c.sigma2, sJ, c.refN, 0).replace(/<div class="live">.*?<\/div>/, '') + EQ.ghbPrivacy(rho, c.delta, c.eps, K, cc, rhoC, epsC) + EQ.zcdp(rho, c.delta, c.eps, r0);
    applyRelevance(document.getElementById('p-privacy'), c);
  }
  let t; store.subscribe(() => { clearTimeout(t); t = setTimeout(update, 60); });
  update();
}
