/**
 * ui/panels/real.js
 * Responsibility: the Real data tab (Section 7). Presents the paper's real-data results exactly, from the generated
 * module data/realResults.js (simulations/export_app_data.py): the pre-registered confirmation on five unseen TCGA
 * networks (Table 10), GHB tuned like its competitors (Table 11), the seven-network benchmark with every literature
 * method in three tuning arms (Table 12) and UCI Heart Disease by training size (Table 13), with charts, the best
 * values in bold, citations and a plain-language account of the tuning arms and of pre-registration.
 */
import { add, explain } from '../dom.js';
import { Figure } from '../../viz/chart.js';
import { TCGA_CONFIRM, TCGA_TUNED, REAL_NETWORKS, UCI_HEART, META } from '../../data/realResults.js';

const f3 = (x) => (x == null ? '' : x < 9.995 ? x.toFixed(3) : x.toFixed(1));
const cite = (c) => (c ? ` <span class="rd-cite">(${c})</span>` : '');
const TOK = { ghb: '--rd-ghb', lite: '--rd-lite', liteEB: '--rd-lite', local: '--rd-local', oracle: '--rd-muted', lit: '--rd-noise' };

export function initReal() {
  const rail = document.getElementById('rail-real'), stage = document.getElementById('stage-real'), nb = document.getElementById('nb-real');
  let eps = '1';

  /* ---------- rail ---------- */
  add(rail, '<div class="rd-ctl"><span class="rd-label" id="real-eps-lab">Privacy budget in the charts</span><div class="rd-seg" role="group" aria-labelledby="real-eps-lab"><button type="button" id="real-eps-1" data-e="1" aria-pressed="true">ε = 1</button><button type="button" id="real-eps-10" data-e="10" aria-pressed="false">ε = 10</button></div></div>');
  add(rail, `<div><h3 class="rd-h">In the paper</h3><ul class="rd-prose">
    <li><a href="#real-confirm">Section 7.1</a>: pre-registered confirmation, Table 10</li>
    <li><a href="#real-tuned">Section 7.1</a>: tuned against tuned, Table 11</li>
    <li><a href="#real-eight">Section 7.2</a>: seven real networks, Table 12, Figure 14</li>
    <li><a href="#real-heart">Section 7.2.1</a>: UCI Heart Disease, Table 13</li>
    <li>Section 7.3: what the real data say</li></ul>
    <p class="rd-msg">Numbers are read from ${META.sources.length} result files by ${META.script} (${META.generated}); they are the numbers of the paper's tables.</p></div>`);

  /* ---------- notebook ---------- */
  explain(nb, 'How to read these tables', [
    'Every number is a held-out log-loss <em>relative to local estimation</em>: each hospital is scored on its own held-out patients (repeated stratified five-fold cross-validation) and the total is divided by that of a model fitted on the hospital alone with the default prior N(0, 2.5²). Below 1, collaboration helped.',
    'GHB and GHB-lite have <strong>nothing to tune</strong>. Every literature method has a grid of hyperparameters, and how it is tuned decides how it compares.'
  ]);
  explain(nb, 'The tuning arms', [
    '<strong>Realistic, or tuned on development</strong>: each method uses the configuration with the lowest mean error on networks the consortium already has (twelve development networks in Section 7.1; UCI Heart Disease in Section 7.2). This is what a deployment can do.',
    '<strong>Leave-one-network-out</strong> (Section 7.2): the configuration best on the other networks, the most generous tuning a hospital could actually do.',
    '<strong>Oracle</strong>: the configuration best on the evaluated network\'s own held-out folds. No hospital can do this; it is an upper bound for the literature methods.'
  ]);
  explain(nb, 'What pre-registration means here', [
    'Before the confirmation ran, the rule that chose the five networks (every further TCGA project with at least four hospitals of five eligible patients and fifteen deaths), the three predictions, GHB\'s frozen configuration and every baseline\'s frozen configuration were committed to the repository. The run was made once. Its outcome could have contradicted the predictions; it did not (Section 7.1).',
    'Table 11 is exploratory, after the confirmation: GHB given a grid like the others. Its rows come from a rerun with new privacy noise, so the frozen configuration differs from Table 10 by a few thousandths, which is the noise of a single run.'
  ]);

  /* ---------- Table 10 ---------- */
  const C = TCGA_CONFIRM;
  add(stage, `<div id="real-confirm"><h2 class="rd-h2">A pre-registered test on five unseen cancer networks</h2><p class="rd-prose">The Cancer Genome Atlas, one network per cancer type, the hospital being the institution that contributed the tissue; death within two years; an eight-coefficient logistic head. Twelve networks served for development; these five (${C.nets.map((n) => `${n.label} ${n.K}`).join(', ')} hospitals of 5 to 109 patients) were fixed by a rule on counts before any model saw them (GHB paper, Section 7.1, Table 10).</p></div>`);
  const t10 = add(stage, '<div class="rd-table-wrap" style="max-height:none"><table class="rd-table static" id="real-t10"></table></div>');
  {
    const head = `<thead><tr><th class="lab">Method</th>${C.nets.map((n) => `<th>${n.label} (${n.K})</th>`).join('')}<th>Mean</th><th>GHB wins</th></tr></thead>`;
    let body = '';
    for (const e of ['1', '10']) {
      const rows = C.eps[e], cand = rows.filter((r) => r.kind !== 'local' && r.kind !== 'oracle'), best = C.nets.map((_, j) => Math.min(...cand.map((r) => r.values[j])));
      const bestMean = Math.min(...cand.map((r) => r.mean));
      body += `<tr><td class="sub" colspan="${C.nets.length + 3}">ε = ${e}${e === '1' ? ' (strong privacy: the joint releases of a few hundred patients are noisy)' : ' (weaker privacy: borrowing can pay)'}</td></tr>`;
      body += rows.map((r) => `<tr class="${r.kind === 'ghb' ? 'ghbrow' : ''}"><td class="lab"><span class="rd-dot" style="--c:var(${TOK[r.kind] || '--rd-noise'})"></span> ${r.label}${cite(r.cite)} <span class="rd-cite">${r.note}</span></td>${r.values.map((v, j) => `<td class="num${r.kind !== 'local' && r.kind !== 'oracle' && v === best[j] ? ' best' : ''} ${v < 1 ? 'gain' : v > 1 ? 'loss' : ''}">${f3(v)}</td>`).join('')}<td class="num${r.kind !== 'local' && r.kind !== 'oracle' && r.mean === bestMean ? ' best' : ''}">${f3(r.mean)}</td><td class="num">${r.wins == null ? '' : `${r.wins}/${C.nets.length}`}</td></tr>`).join('');
    }
    t10.querySelector('table').innerHTML = head + `<tbody>${body}</tbody>`;
  }
  add(stage, '<p class="rd-prose"><strong>Bold</strong>: the lowest per network among methods that use other hospitals (oracle excluded). Wins: networks in which GHB is more accurate than that row. The best of the six, oracle per network, is the best configuration of the best method chosen afresh for each network on its own test folds.</p>');
  const fConf = new Figure(add(stage, '<div></div>'), { id: 'fig-real-confirm', num: 'R1', caption: '', xLabel: 'held-out log-loss relative to local', yLabel: '', height: 240, canLogY: false, canLogX: false });
  add(stage, `<div class="rd-callout"><strong>The three predictions all held.</strong> GHB has a lower mean error than every literature method tuned on development, at both budgets (ε = 1: ${f3(row(C.eps['1'], 'ghb').mean)} against ${f3(Math.min(...C.eps['1'].filter((r) => r.kind === 'lit').map((r) => r.mean)))} for the best, DP-FedAvg with fine-tuning; ε = 10: ${f3(row(C.eps['10'], 'ghb').mean)} against ${f3(row(C.eps['10'], 'Ditto').mean)} for Ditto). At ε = 10 GHB improves on GHB-lite (${f3(row(C.eps['10'], 'lite').mean)} to ${f3(row(C.eps['10'], 'ghb').mean)}). At ε = 10 it beats the best realistic method, Ditto, in exactly ${row(C.eps['10'], 'Ditto').wins} of 5 networks. Against the oracle arm, GHB is ahead at ε = 1; at ε = 10 hindsight tuning puts pFedMe ahead (${f3(row(C.eps['10'], 'pFedMe').oracle_mean)}).</div>`);

  /* ---------- Table 11 ---------- */
  const T = TCGA_TUNED;
  add(stage, `<div id="real-tuned"><h2 class="rd-h2">Tuned against tuned</h2><p class="rd-prose">Exploratory, after the confirmation: GHB given a grid of ${T.configs || 36} configurations (scale of the prior on τ, weight of the non-borrowing components, split of the budget) and tuned exactly as the baselines were (GHB paper, Table 11). Mean over the five confirmation networks.</p></div>`);
  {
    const cols = [['1', 'dev'], ['1', 'oracle'], ['10', 'dev'], ['10', 'oracle']], best = cols.map(([e, a]) => Math.min(...T.rows.filter((r) => r.key !== 'ghb_frozen').map((r) => r[a][e])));
    add(stage, `<div class="rd-table-wrap" style="max-height:none"><table class="rd-table static" id="real-t11"><thead><tr><th class="lab">Method</th><th>ε = 1, development</th><th>ε = 1, oracle</th><th>ε = 10, development</th><th>ε = 10, oracle</th></tr></thead><tbody>${T.rows.map((r) => `<tr class="${r.kind === 'ghb' ? 'ghbrow' : ''}"><td class="lab"><span class="rd-dot" style="--c:var(${r.kind === 'ghb' ? '--rd-ghb' : '--rd-noise'})"></span> ${r.label}${cite(r.cite)}</td>${cols.map(([e, a], i) => `<td class="num${r.key !== 'ghb_frozen' && r[a][e] === best[i] ? ' best' : ''}">${f3(r[a][e])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
    add(stage, '<p class="rd-prose">Tuning changes GHB little: its frozen configuration is already close to the best it can reach. With the same tuning on both sides it stays ahead of every method at ε = 1 and under development tuning at ε = 10; under hindsight tuning at ε = 10, pFedMe is ahead.</p>');
  }

  /* ---------- Table 12 ---------- */
  const N = REAL_NETWORKS;
  add(stage, `<div id="real-eight"><h2 class="rd-h2">Seven real networks against twelve literature methods</h2><p class="rd-prose">Arizona Medicare (${N.nets[0].K} hospitals) and six TCGA cancer networks with a four- or five-coefficient head, each fitted on its own (GHB paper, Section 7.2, Table 12). GHB-lite was developed on these networks; GHB was added afterwards with its frozen configuration. UCI Heart Disease, the eighth network, served only to tune the baselines in the development arm.</p></div>`);
  {
    const e2 = ['1', '10'], arms = ['oracle', 'lono', 'uci'];
    const cand = N.rows.filter((r) => r.kind !== 'local'), best = {};
    for (const e of e2) for (const a of arms) best[e + a] = Math.min(...cand.map((r) => r.eps[e][a]));
    const head = `<thead><tr><th class="lab" rowspan="2">Method</th><th colspan="4">ε = 1</th><th colspan="4">ε = 10</th></tr><tr>${e2.map(() => '<th>Oracle</th><th>Other networks</th><th>Wins</th><th>UCI-tuned</th>').join('')}</tr></thead>`;
    const body = N.rows.map((r, i) => `${i === 0 ? `<tr><td class="sub" colspan="9">GHB and the local reference</td></tr>` : ''}${r.kind === 'lit' && N.rows[i - 1].kind !== 'lit' ? `<tr><td class="sub" colspan="9">Federated and personalized methods, same budget, ordered by their error at ε = 1 tuned on the other networks</td></tr>` : ''}<tr class="${r.key === 'ghb' ? 'ghbrow' : ''}"><td class="lab"><span class="rd-dot" style="--c:var(${r.key === 'ghb' ? '--rd-ghb' : r.kind === 'ghb' ? '--rd-lite' : r.kind === 'local' ? '--rd-local' : '--rd-noise'})"></span> ${r.label}${cite(r.cite)} <span class="rd-cite">${r.note}</span></td>${e2.map((e) => `${arms.slice(0, 2).map((a) => `<td class="num${r.kind !== 'local' && r.eps[e][a] === best[e + a] ? ' best' : ''}">${f3(r.eps[e][a])}</td>`).join('')}<td class="num">${r.eps[e].wins == null ? '' : r.eps[e].wins}</td><td class="num">${r.kind === 'lit' || r.key === 'liteTuned' || r.key === 'local' ? f3(r.eps[e].uci) : ''}</td>`).join('')}</tr>`).join('');
    add(stage, `<div class="rd-table-wrap" style="max-height:none"><table class="rd-table static" id="real-t12">${head}<tbody>${body}</tbody></table></div>`);
    add(stage, '<p class="rd-prose">Mean over the seven networks. Wins: in how many of the seven networks untuned GHB has a lower error than that method tuned on the other networks. UCI-tuned: tuned on UCI Heart Disease alone and frozen (Section 7.2, "Tuning is most of the baselines\' advantage"). Bold: the lowest per column among methods that use other hospitals.</p>');
  }
  const fEight = new Figure(add(stage, '<div></div>'), { id: 'fig-real-eight', num: 'R2', caption: '', xLabel: 'held-out log-loss relative to local', yLabel: '', height: 300, canLogY: false, canLogX: false });

  /* ---------- Table 13 ---------- */
  const H = UCI_HEART;
  add(stage, `<div id="real-heart"><h2 class="rd-h2">UCI Heart Disease: how much the size of a hospital matters</h2><p class="rd-prose">Four hospitals (${H.sites.map((s, i) => `${s} ${H.n[i]}`).join(', ')}; prevalence ${Math.round(100 * Math.min(...H.prevalence))}% to ${Math.round(100 * Math.max(...H.prevalence))}%), each trained on at most 20, 40 or 80 patients or on all of them, ${H.splits} splits. This analysis concerns GHB-lite and the baselines it was first compared with, tuned on the held-out folds; it explains the limit that GHB was built to remove (GHB paper, Section 7.2.1, Table 13).</p></div>`);
  {
    const lab = { 20: '≤ 20 patients', 40: '≤ 40', 80: '≤ 80', all: 'All' };
    const head = `<thead><tr><th class="lab" rowspan="2">Method</th>${H.tags.map((t) => `<th colspan="2">${lab[t]}</th>`).join('')}</tr><tr>${H.tags.map(() => '<th>Relative</th><th>Worst</th>').join('')}</tr></thead>`;
    const best = {};
    for (const t of H.tags) { best[t + 'rel'] = Math.min(...H.rows.filter((r) => r.key !== 'Local' && !r.label.includes('no DP') && !r.label.startsWith('Global')).map((r) => r.byTag[t].rel)); best[t + 'w'] = Math.min(...H.rows.filter((r) => r.key !== 'Local' && !r.label.includes('no DP') && !r.label.startsWith('Global')).map((r) => r.byTag[t].worst)); }
    const body = `<tr><td class="lab">Local log-loss (nats)</td>${H.tags.map((t) => `<td class="num" colspan="2" style="text-align:center">${f3(H.localLL[t])}</td>`).join('')}</tr>` + H.rows.filter((r) => r.key !== 'Local').map((r) => `<tr><td class="lab">${r.label}${cite(r.cite)}${r.label.includes('no DP') ? ' <span class="rd-tag">no DP</span>' : ''}</td>${H.tags.map((t) => { const x = r.byTag[t], ok = !r.label.includes('no DP') && !r.label.startsWith('Global'); return `<td class="num${ok && x.rel === best[t + 'rel'] ? ' best' : ''}">${f3(x.rel)}<small> ±${(2 * x.se).toFixed(3)}</small></td><td class="num${ok && x.worst === best[t + 'w'] ? ' best' : ''}">${x.worst.toFixed(2)}</td>`; }).join('')}</tr>`).join('');
    add(stage, `<div class="rd-table-wrap" style="max-height:none"><table class="rd-table static" id="real-t13">${head}<tbody>${body}</tbody></table></div>`);
    add(stage, '<p class="rd-prose">At ε = 1 the privacy noise of each per-hospital release is many times the hospital\'s own sampling error (about 26 times at 40 patients), so GHB-lite falls back to local estimation, doing no harm. Without privacy it borrows, most for the smallest training sets. The tuned DP-FedAvg + FT gains exactly as much as local gradient descent with tuned stopping: its gain is early stopping tuned on the test folds, not borrowing.</p>');
  }

  /* ---------- charts ---------- */
  function draw() {
    document.querySelectorAll('#real-eps-1, #real-eps-10').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.e === eps)));
    const rows = C.eps[eps], y = C.nets.map((n) => `${n.label} (${n.K})`);
    const lit = rows.filter((r) => r.kind === 'lit'), bestReal = C.nets.map((_, j) => Math.min(...lit.map((r) => r.values[j])));
    const pts = (vals) => vals.map((v, j) => [v, j]);
    fConf.setData({ yCats: y, xName: 'relative log-loss', yName: 'network', series: [
      { id: 'orc', label: 'best of six, oracle per network', tok: '--rd-muted', noLine: true, marker: 3.5, pts: pts(row(rows, 'best_oracle').values) },
      { id: 'real', label: 'best of six, tuned on development', tok: '--rd-noise', noLine: true, marker: 4, pts: pts(bestReal) },
      { id: 'loc', label: 'local, evidence prior', tok: '--rd-local', noLine: true, marker: 3, pts: pts(row(rows, 'local').values) },
      { id: 'lite', label: 'GHB-lite (evidence prior)', tok: '--rd-lite', noLine: true, marker: 3.5, pts: pts(row(rows, 'lite').values) },
      { id: 'ghb', label: 'GHB, no tuning', tok: '--rd-ghb', noLine: true, marker: 5, pts: pts(row(rows, 'ghb').values) }], vlines: [{ x: 1, tok: '--rd-local' }] });
    fConf.setCaption(`Pre-registered confirmation at ε = ${eps}: held-out log-loss relative to local estimation per network (hospitals in parentheses). Left of the line, collaboration helped. Teal: GHB; light teal: GHB-lite; orange: the best of the six literature methods tuned on the development networks; grey: the same tuned on the network's own test folds.`);
    const nets = N.nets.map((n) => `${n.label} (${n.K})`), R = (k) => N.rows.find((r) => r.key === k), lits = N.rows.filter((r) => r.kind === 'lit');
    const bestOf = (arm) => N.nets.map((_, j) => Math.min(...lits.map((r) => r.eps[eps].perNetwork[arm][j])));
    fEight.setData({ yCats: nets, xName: 'relative log-loss', yName: 'network', series: [
      { id: 'orc', label: 'best of twelve, oracle', tok: '--rd-muted', noLine: true, marker: 3.5, pts: pts(bestOf('oracle')) },
      { id: 'lono', label: 'best of twelve, tuned on the other networks', tok: '--rd-noise', noLine: true, marker: 4, pts: pts(bestOf('lono')) },
      { id: 'loc', label: 'one hospital alone, tuned prior', tok: '--rd-local', noLine: true, marker: 3, pts: pts(R('local').eps[eps].perNetwork.lono) },
      { id: 'eb', label: 'GHB-lite, evidence prior', tok: '--rd-lite', noLine: true, marker: 3.5, pts: pts(R('liteEB').eps[eps].perNetwork.oracle) },
      { id: 'ghb', label: 'GHB, no tuning', tok: '--rd-ghb', noLine: true, marker: 5, pts: pts(R('ghb').eps[eps].perNetwork.oracle) }], vlines: [{ x: 1, tok: '--rd-local' }] });
    fEight.setCaption(`Seven real networks at ε = ${eps} (as Figure 14 of the paper): GHB (teal) and GHB-lite with the evidence prior (light teal), neither tuned; orange: the best of twelve literature methods for that network, each tuned on the other networks; grey: the same tuned on the network's own folds (oracle); open: one hospital alone with a tuned prior. Choosing the best of twelve methods separately for every network is itself an advantage no hospital has.`);
  }
  rail.querySelectorAll('[data-e]').forEach((b) => b.addEventListener('click', () => { eps = b.dataset.e; draw(); }));
  return { onShow() { draw(); } };
}

function row(rows, key) { return rows.find((r) => r.key === key); }
