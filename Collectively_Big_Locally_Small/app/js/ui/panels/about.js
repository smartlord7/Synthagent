/**
 * ui/panels/about.js
 * Responsibility: the About tab: the two protocols, the assumptions in plain words, the map from tabs to the GHB
 * paper (and to the survey for smallness), how the lab was checked against the paper's code, and its limits.
 */
import { add } from '../dom.js';

export function initAbout() {
  const rail = document.getElementById('rail-about'), stage = document.getElementById('stage-about'), nb = document.getElementById('nb-about');
  add(rail, `<div><h3 class="rd-h">Map to the GHB paper</h3><ul class="rd-prose">
    <li>Smallness: GHB paper Section 2.1, Eq. (1); survey Section 2 (Eq. (1), Eq. (2), Def. 2.3, Example 2.4)</li>
    <li>Borrowing: Section 4.1, Prop. 4.1, Eq. (2) and (3); Sims. 1 to 3</li>
    <li>Privacy: Section 4.2, Prop. 4.2, Eq. (4); Section 5.2 and Prop. 5.1 for GHB; Sims. 5 and 14</li>
    <li>Network &amp; gossip: Section 4.3, Prop. 4.3, Fig. 2, Algorithm 1; Sims. 4 and 13</li>
    <li>Unknown heterogeneity: Remark 4.4</li>
    <li>Regions: Section 4.5, Eq. (5); Sim. 6</li>
    <li>GHB (joint releases): Section 5.2, Section 5.4, Algorithm 1, Prop. 5.1</li>
    <li>GHB-lite (per-hospital releases): Section 5.3</li>
    <li>Literature arena: Section 6.3, Sims. 7, 8, 10 and 15; Section 7.2 for the personalized methods</li>
    <li>Stress tests: Section 6.4, Sims. 9, 11, 12</li>
    <li>Real data: Section 7, Tables 10 to 13</li>
    <li>Sweeps: Table 2, every simulation figure</li></ul></div>`);
  add(stage, `<div><h2 class="rd-h2">Two protocols</h2><ul class="rd-prose">
    <li><strong>GHB</strong> (Section 5.2). The network makes joint releases, sums over every patient of every hospital, privatized once for the whole network by secure summation over the agreement graph (pairwise masks that cancel, gossip to consensus, no server). In the Gaussian tabs the release is the sum S̃ = Σ n<sub>k</sub>θ̂<sub>k</sub> + N(0, s²), s = 2C/√(2ρ); for logistic heads, the sums of every patient's score and information. Each hospital removes its own data exactly and integrates the unknown heterogeneity out against its own evidence, with a quarter of the prior weight on not borrowing. Nothing is tuned.</li>
    <li><strong>GHB-lite</strong> (Section 5.3). Every hospital releases its own privatized estimate (Gaussian tabs) or mean score at the anchor (logistic heads), private on its own, so no trust in summation is needed. The network learns τ from the spread of the releases. Its noise grows as 1/n<sub>k</sub>.</li>
    <li>Which to run (Section 8.1): GHB when hospitals are small or privacy is strong and secure summation among non-colluding hospitals is available; GHB-lite when releases are informative (many hospitals, enough patients, a public feature distribution that matches) or collusion cannot be ruled out.</li></ul></div>`);
  add(stage, `<div><h2 class="rd-h2">What the simulations assume</h2><ul class="rd-prose">
    <li><strong>Gaussian model.</strong> Hospital k has a parameter θ<sub>k</sub> = μ + u<sub>k</sub> with u<sub>k</sub> of variance τ², and a local estimate with variance v<sub>k</sub> = σ²/n<sub>k</sub> (Eq. 2). Site effects can be Gaussian, heavy-tailed, contain an outlier, or follow regions.</li>
    <li><strong>Logistic world</strong> (arena). Five-coefficient logistic heads with θ<sub>k</sub> = μ + √τ² z<sub>k</sub>, features bounded by ‖x‖ ≤ √5, a published anchor with error, excess log-loss on fresh patients (Section 6.3).</li>
    <li><strong>Honest but curious.</strong> Every hospital follows the protocol, unless the dishonest-hospital stress test is on. GHB's sums assume hospitals do not collude to remove their noise shares; c colluders weaken ρ to ρK/(K − c) (Prop. 5.1).</li>
    <li><strong>Public sizes.</strong> Site sizes n<sub>k</sub>, the per-patient variance σ² and the noise variances are public.</li>
    <li><strong>ℓ<sub>2</sub> clipping.</strong> Per-patient contributions are clipped to C, so one patient moves a hospital mean by at most 2C/n<sub>k</sub> and a network sum by at most 2C; clipping is assumed to add no bias.</li>
    <li><strong>Only releases leave a hospital.</strong> Gossip, τ, the leave-one-out means and every evidence weight are post-processing and cost no further privacy. A hospital uses its own data without noise.</li>
    <li><strong>Risk is averaged over hospitals.</strong> Propositions 4.1 and 4.2 hold for a hospital drawn from the population of hospitals. A particular atypical hospital can lose by borrowing.</li>
    <li><strong>Plug-in versus oracle.</strong> The oracle knows τ². Plug-in GHB-lite estimates τ² (DerSimonian-Laird or a half-Cauchy posterior) from the releases; GHB integrates τ out over its half-Cauchy prior (Remark 4.4); "GHB, known τ²" is a reference.</li>
  </ul></div>`);
  add(stage, `<div><h2 class="rd-h2">How the lab was checked</h2><ul class="rd-prose">
    <li><code>tests/math.test.mjs</code>: closed forms, SLEM, zCDP, the Gaussian simulations' reference values.</li>
    <li><code>tests/gauss_ghb.test.mjs</code>: GHB's M<sub>k</sub> against simulation, and the regimes where GHB and GHB-lite each win.</li>
    <li><code>tests/logit.test.mjs</code>: every logistic method (GHB, GHB-lite, every literature method) against the Python code on fixed data, to 10<sup>−8</sup>.</li>
    <li><code>tests/arena.test.mjs</code>: the arena's Simulation 15 and 8 means against the paper's values, within Monte Carlo error.</li></ul></div>`);
  add(stage, `<div><h2 class="rd-h2">Limits of this lab</h2><ul class="rd-prose">
    <li>The Gaussian tabs run GHB with the network sum computed exactly (secure summation run to consensus, as in Algorithm 1); GHB-lite uses T gossip rounds, so with few rounds the comparison favours GHB.</li>
    <li>Gaussian GHB's no-borrowing components keep the local estimate, with evidence under N(0, s²), s in {0.5, 1, 2.5}; its borrowing components carry no default prior. The logistic GHB of the arena is the paper's.</li>
    <li>The arena's random draws come from this page's generator, so its numbers match the paper's within Monte Carlo error, not digit for digit; GHB's pan-network step is not needed for one network. The personalized methods start from the origin, as in personalized.py.</li>
    <li>Joint release in groups (Simulation 5) and refits (Simulation 14) are shown in closed form, as in the paper's lines.</li>
    <li>The paper's size profile (exactly the numpy draws of <code>site_sizes()</code>) and the K = 24 regional and random graphs are exact copies of the Python seeds; other Gaussian-tab sizes and graphs come from this page's generator.</li>
    <li>The real-data tab shows the paper's saved results; it does not rerun them (the data and the runs take hours).</li></ul></div>`);
  add(nb, '<div><h3 class="rd-h">Reading the figures</h3><p class="rd-prose">Relative risk R<sub>k</sub>/v<sub>k</sub> compares a method with local estimation: below 1 the hospital gains. In the arena and the real-data tab the error is the excess or held-out log-loss, again often relative to local estimation. Error bars are ±2 Monte Carlo standard errors. Hover any figure for exact values; use the method keys above a figure to show or hide series and the switches to change axis scales. Dark teal is GHB, light teal GHB-lite, throughout.</p></div>');
}
