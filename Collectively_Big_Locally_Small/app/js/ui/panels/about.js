/**
 * ui/panels/about.js
 * Responsibility: the About tab: assumptions in plain words, the map from tabs to the paper, and limits.
 */
import { add } from '../dom.js';

export function initAbout() {
  const rail = document.getElementById('rail-about'), stage = document.getElementById('stage-about'), nb = document.getElementById('nb-about');
  add(rail, `<div><h3 class="rd-h">Map to the paper</h3><ul class="rd-prose">
    <li>Smallness: Section 2, Eq. (1), Eq. (2), Def. 2.3, Example 2.4</li>
    <li>Network &amp; gossip: Section 8.4, Prop. 8.4, Fig. 9, Sim. 4 and 13</li>
    <li>Borrowing: Props. 8.1 and 8.2, Eq. (4) to (6), Algorithm 1, Sims. 1 to 3</li>
    <li>Stress tests: Sims. 9, 11, 12</li>
    <li>Privacy: Prop. 8.2, Sims. 5 and 14</li>
    <li>Regions: Eq. (7), Sim. 6</li>
    <li>Sweeps: Table 6, every simulation figure</li></ul></div>`);
  add(stage, `<div><h2 class="rd-h2">What the simulations assume</h2><ul class="rd-prose">
    <li><strong>Gaussian model.</strong> Hospital k has a parameter θ<sub>k</sub> = μ + u<sub>k</sub> with u<sub>k</sub> of variance τ², and a local estimate with variance v<sub>k</sub> = σ²/n<sub>k</sub> (Eq. 4). Site effects can be Gaussian, heavy-tailed, contain an outlier, or follow regions.</li>
    <li><strong>Honest but curious.</strong> Every hospital follows the protocol and releases what it should, unless the dishonest-hospital stress test is switched on.</li>
    <li><strong>Public sizes.</strong> Site sizes n<sub>k</sub>, the per-patient variance σ² and the noise variances are public, so weights computed from them reveal nothing new.</li>
    <li><strong>ℓ<sub>2</sub> clipping.</strong> Per-patient contributions are clipped in ℓ<sub>2</sub> norm to C, so one patient moves a release by at most 2C/n<sub>k</sub>; clipping is assumed to add no bias.</li>
    <li><strong>Only releases leave a hospital.</strong> Gossip, the estimate of τ² and the leave-one-out mean are post-processing of privatized releases and cost no further privacy. A hospital uses its own estimate without noise.</li>
    <li><strong>Risk is averaged over hospitals.</strong> Propositions 8.1 and 8.2 hold for a hospital drawn from the population of hospitals. A particular atypical hospital can lose by borrowing.</li>
    <li><strong>A fixed reference.</strong> Releases are centred on a fixed public reference and never on the knowledge anchor, which would make the anchor part of what is released.</li>
    <li><strong>Plug-in versus oracle.</strong> The oracle knows τ² and uses the exact equal-weight leave-one-out mean. Plug-in GHB estimates τ² (DerSimonian–Laird or a half-Cauchy posterior) and forms an inverse-variance leave-one-out mean through T rounds of Metropolis–Hastings gossip (Algorithm 1).</li>
  </ul></div>`);
  add(stage, `<div><h2 class="rd-h2">Limits of this lab</h2><ul class="rd-prose">
    <li>Simulations 7, 8, 10 and 11 use logistic heads in the paper; their presets here are Gaussian analogues and say so.</li>
    <li>Simulation 15 (multi-round DP-FedAvg, fine-tuning, decentralized DP-SGD and clustered training at the same budget) is reproduced only by <code>simulations/exp15_literature.py</code>.</li>
    <li>Joint release (Simulation 5) and refits (Simulation 14) are shown in closed form, as in the paper's lines; the Monte Carlo engine simulates individual releases.</li>
    <li>The paper's size profile and the K = 24 regional and random graphs are exact copies of the Python seeds. Other sizes and graphs come from this page's own seeded generator, so they differ from numpy draws.</li>
    <li>With link failures, the engine averages over 8 fixed realizations of the failure pattern, as the Python code reuses a finite set of sequences.</li></ul></div>`);
  add(nb, '<div><h3 class="rd-h">Reading the figures</h3><p class="rd-prose">Relative risk R<sub>k</sub>/v<sub>k</sub> compares a method with local estimation: below 1 the hospital gains. Error bars are ±2 Monte Carlo standard errors. Hover any figure for exact values; use the method keys above a figure to show or hide series and the switches to change axis scales.</p></div>');
}
