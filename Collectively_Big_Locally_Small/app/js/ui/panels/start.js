/**
 * ui/panels/start.js
 * Responsibility: the Start here tab: what the lab is for, a pointer to Simple mode, three one-click demos (GHB against GHB-lite under strong
 * privacy, every literature method at the paper's main setting, the real-data results) and a map of the tabs.
 */
import { add } from '../dom.js';

/**
 * Build the Start here tab.
 * @param {{demo:(id:string)=>void, tour:()=>void, open:(tab:string)=>void, simple?:()=>void}} act actions provided by main.js
 * @returns {{}}
 * @see Sections 4 to 7 of the GHB paper
 */
export function initStart(act) {
  const stage = document.getElementById('stage-start');
  add(stage, `<div><h2 class="rd-h2">Start here</h2><p class="rd-prose">Graph Hierarchical Borrowing lets small hospitals borrow strength from each other without sharing data, without a server and without tuning. Each hospital shrinks its own model toward a leave-one-out estimate of the network by an amount its own data decide. <strong>GHB</strong> makes two joint releases, sums over every patient privatized once for the whole network by secure summation (Section 5.2); <strong>GHB-lite</strong> has every hospital release on its own (Section 5.3). This lab lets you run both, against every method of the paper, and read the paper's real-data results.</p></div>`);
  if (act.simple) {
    const sm = add(stage, '<p class="rd-callout">New to privacy, statistics or AI? <button class="rd-btn" type="button" id="start-simple">Switch to Simple mode</button> for a one-page, plain-language version: set up a few hospitals and test GHB in a minute.</p>');
    sm.querySelector('#start-simple').onclick = () => act.simple();
  }
  const cards = add(stage, '<div class="rd-start"></div>');
  const demos = [
    ['ghb-vs-lite', '--rd-ghb', 'GHB against GHB-lite under strong privacy', 'Twenty-four small hospitals (median 30 patients) at ε = 0.1. Per-hospital releases drown in noise, so GHB-lite falls back to local estimation; GHB\'s noise is calibrated to the network sum and it still borrows. Opens Borrowing & metrics with both protocols side by side.', 'Run this demo'],
    ['arena-all', '--rd-noise', 'Every literature method at the paper\'s main setting', 'The logistic world of Simulation 15 (τ² = 0.05, ε = 1, 24 hospitals): GHB, GHB-lite, one-shot FedAvg, meta-analysis, Ditto, DP-FedAvg with and without fine-tuning and secure aggregation, decentralized and clustered DP-FedAvg, pFedMe, APFL, Per-FedAvg, FedRep and FedProx, all at the same budget. Runs in the background.', 'Run every method'],
    ['real', '--rd-ref', 'Real data results', 'The pre-registered confirmation on five unseen cancer networks, GHB tuned like its competitors, and the seven-network benchmark against twelve literature methods in three tuning arms, exactly as in the paper\'s tables.', 'Open the results']
  ];
  for (const [id, c, title, text, btn] of demos) {
    const el = add(cards, `<div class="rd-card" style="--c:var(${c})"><h3>${title}</h3><p>${text}</p><button class="rd-btn primary" type="button" id="start-${id}">${btn}</button></div>`);
    el.querySelector('button').onclick = () => act.demo(id);
  }
  add(stage, '<div class="rd-btns"><button class="rd-btn" type="button" id="start-tour">Take the guided tour</button><button class="rd-btn" type="button" id="start-lite-wins">When does GHB-lite win?</button></div>');
  stage.querySelector('#start-tour').onclick = () => act.tour();
  stage.querySelector('#start-lite-wins').onclick = () => act.demo('lite-wins');
  const tabs = [
    ['calc', 'Smallness', 'How small is a dataset for its task: effective sample size and Riley\'s criterion (survey Section 2; GHB paper Section 2.1, Eq. 1).'],
    ['setup', 'Setup', 'Every parameter of the Gaussian simulations in one place, with presets and JSON import and export.'],
    ['network', 'Network & gossip', 'Watch one replicate: per-hospital releases (GHB-lite) or masked shares gossiped to the network sum (GHB), then borrowing (Section 4.3, Proposition 4.3).'],
    ['borrow', 'Borrowing & metrics', 'Monte Carlo of GHB, GHB-lite and references per hospital, against Proposition 4.1 and Eq. (3).'],
    ['arena', 'Literature arena', 'GHB and GHB-lite against every method in the paper, in the logistic world, with paper-default, oracle or manual tuning (Section 6.3).'],
    ['real', 'Real data', 'Section 7: the pre-registered confirmation and the seven-network benchmark.'],
    ['stress', 'Stress tests', 'Atypical hospitals, dishonest hospitals and a wrong anchor (Section 6.4).'],
    ['privacy', 'Privacy', 'What privacy costs: per-hospital noise against the noise of a network sum, refits, collusion (Propositions 4.2 and 5.1).'],
    ['regions', 'Regions (GMRF)', 'Structured heterogeneity, Eq. (5).'],
    ['sweeps', 'Sweeps & figures', 'Sweep any parameter and reproduce the paper\'s simulation figures, including GHB against GHB-lite over ε and hospital size.'],
    ['about', 'About', 'Assumptions, limits, and the map from tabs to the paper.']
  ];
  add(stage, `<div><h3 class="rd-h">What each tab does</h3><table class="rd-table static"><tbody>${tabs.map(([id, t, d]) => `<tr><td class="lab"><a href="#${id}" data-open="${id}">${t}</a></td><td class="lab" style="font-family:var(--rd-font-prose)">${d}</td></tr>`).join('')}</tbody></table></div>`);
  stage.addEventListener('click', (e) => { const a = e.target.closest('a[data-open]'); if (a) { e.preventDefault(); act.open(a.dataset.open); } });
  return {};
}
