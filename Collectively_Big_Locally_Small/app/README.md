# GHB Simulation Lab

An interactive companion to the paper *Borrowing Strength Without Borrowing Data: Private, Serverless and Tuning-Free Collaboration with Graph Hierarchical Borrowing* (the GHB paper, `Graph_Hierarchical_Borrowing/ghb.tex`), and to the survey *Collectively Big, Locally Small* for the notion of small data. In GHB, hospitals borrow strength from each other over a sparse graph of agreements, with differential privacy, no central server and nothing to tune.

The lab has two protocols side by side:

- **GHB** (paper Section 5.2): the network makes joint releases, sums over every patient privatized once for the whole network by secure summation over the graph. Each hospital removes its own data exactly and integrates the unknown heterogeneity out against its own evidence. The lab runs GHB as released (v5). Compared with the first version (v4, still selectable in the arena as "GHB v4 (earlier version)") it does not borrow the intercept, because base rates differ across hospitals; it releases the information sum with the tighter sensitivity √2 G²/4 (one patient changes it by the difference of two positive semidefinite rank-one terms); and it releases only the diagonal of that sum when a rule on public quantities, fixed before the release, says the full matrix would be mostly noise.
- **GHB-lite** (Section 5.3): every hospital releases its own privatized estimate or score, private on its own.

It measures how small a dataset is, animates both protocols, runs GHB and GHB-lite against every literature method of the paper with one click, and shows the paper's real-data results.

A hosted copy is at https://claude.ai/artifact/7ZYuDy8A9eur912XQNfELs (private until shared from its Share menu).

## Run it

The app is plain HTML, CSS and native ES modules. There is no build step and no npm. ES modules need HTTP, so serve the folder rather than opening the file directly. It works offline: only the fonts come from the network, with local fallbacks.

```sh
cd app
python3 -m http.server 8000      # then open http://localhost:8000
```

Tests (Node 18 or later), from `app/`:

```sh
node --test tests/
```

This takes about a minute, most of it the arena check against Simulation 15.

The real-data tab reads `js/data/realResults.js`. That file is generated from the paper's saved results; regenerate it after rerunning the real-data experiments:

```sh
cd simulations
python3 export_app_data.py
```

## Start here

The first tab offers three one-click demos:

1. **GHB against GHB-lite under strong privacy.** Small hospitals at ε = 0.1, both protocols side by side.
2. **Every literature method at the paper's main setting.** The arena at τ² = 0.05, ε = 1, runs in the background.
3. **Real data results.** Opens the real-data tab.

## Tabs and the paper

| Tab | What it does | Paper |
|---|---|---|
| Start here | Three one-click demos and a map of the tabs | |
| Smallness | Effective sample size, effective events, Riley's n*, smallness index σ and its regime, deflation chart | GHB §2.1, Eq. (1); survey §2, Eq. (1)-(2), Def. 2.3, Example 2.4 |
| Setup | Every parameter in one place: network (generated or per-hospital sizes, topology, editable graph, link failures), model, privacy, estimation, safeguards, adversaries, Monte Carlo; presets; JSON export and import | §6.1 design |
| Network & gossip | Animated network for either protocol. GHB-lite: releases and gossip. GHB: masked shares (pairwise masks plus noise shares) gossiped to the network sum. Then borrowing. Shows λ, spectral gap, rounds to tolerance, consensus error and a full readout for any hospital | §4.3, Prop. 4.3, Fig. 2, Algorithm 1; §5.2; Sims. 4, 13 |
| Borrowing & metrics | Monte Carlo of local, pooled, oracle, GHB (joint release), GHB with known τ², plug-in GHB-lite and its options, with a protocol switch; per-hospital risk, n_eff gain against 1 + B/n_k, sortable metrics table | Props. 4.1-4.2, Eq. (2)-(4), Remark 4.4, §5.2; Sims. 1-3 |
| Literature arena | The logistic world of the paper's simulations. GHB and GHB-lite against one-shot FedAvg, meta-analysis, Ditto-style, DP-FedAvg (with fine-tuning, with secure aggregation), decentralized DP-SGD, clustered DP-FedAvg, Ditto, pFedMe, APFL, Per-FedAvg, FedRep and FedProx, all at the same (ε, δ). Paper-default, oracle or manual tuning; presets for Simulations 8, 10 and 15 with the paper's values beside yours; CSV/JSON export | §6.3, Sims. 7, 8, 10, 15; §7.2 |
| Real data | The pre-registered TCGA confirmation, GHB tuned like its competitors, the seven-network benchmark in three tuning arms, and UCI Heart Disease, exactly as in the paper's tables | §7, Tables 10-13, Fig. 14 |
| Stress tests | Atypical hospitals, dishonest hospitals (GHB and GHB-lite) and a wrong anchor, with the safeguards | §6.4; Sims. 9, 11, 12 |
| Privacy | Per-hospital noise against the noise of a network sum, B against ε for individual, group and joint releases, refits under basic composition and zCDP, collusion | Props. 4.2, 5.1; Eq. (4); Sims. 5, 14 |
| Regions (GMRF) | Regional site effects; exchangeable against GMRF prior | §4.5, Eq. (5); Sim. 6 |
| Sweeps & figures | Sweep any parameter (linear or log) and plot any metric with ±2 standard errors; one-click reproductions of the paper's figures, including GHB against GHB-lite over ε and hospital size | Sims. 2-14 |
| About | The two protocols, assumptions and limits in plain words | §5, §8 |

## Architecture

```
index.html ── css/researchdata.css
     │
     └─ js/main.js  (wiring only)
          ├─ js/config.js       schema of every parameter, defaults = the paper's design (incl. the arena's)
          ├─ js/scenarios.js    presets: paper defaults, Simulations 1-14, GHB vs GHB-lite, arena presets (Sims. 8, 10, 15)
          ├─ js/data/           realResults.js, generated by simulations/export_app_data.py
          ├─ js/ui/             store, tabs, controls, panels/*, inspector, metrics table, tour, equations
          │     └─ uses js/viz/ (canvas figures, network renderer, theme tokens)
          │     └─ uses js/sim/ (engine, gossip, sweep, scheduler, arena + its worker)
          └─ js/sim/  ──► js/math/ (pure functions, no DOM)
```

- **`math/`** holds pure, side-effect-free functions.
  - `smallness.js`: n_eff, Riley's criterion, σ.
  - `dp.js`: Gaussian mechanism, zCDP accounting, the noise of a network sum, collusion.
  - `graph.js`: topologies, Metropolis-Hastings weights, SLEM, link failures.
  - `ghb.js`: Gaussian GHB-lite (DerSimonian-Laird, half-Cauchy τ², inverse-variance leave-one-out mean, shrinkage, robust Student-t prior, opt-out, screening, anchor check, GMRF Jacobi) and Gaussian GHB with a joint release (closed-form M_k, τ integrated out over its prior).
  - `linalg.js`: small dense linear algebra for model heads.
  - `logit.js`: the logistic world (features, MAP fits, Laplace-evidence mixtures, excess log-loss, AUC, public moments).
  - `ghbLogit.js`: GHB (two joint releases, `ghb_v4.py` with the frozen v5 configuration `real_v3.GHB_V5`; `GHB_V4` selects the earlier version) and GHB-lite (`ghb_logit.py`) for logistic heads.
  - `literature.js`: every literature method (`literature.py`, `personalized.py`, `ghb_sa.py`, the one-shot baselines of `logit_world.py`).
  - `random.js`: seeded generator, the exact numpy site-size profile, log-normal sizes.
- **`sim/`** runs Monte Carlo jobs.
  - `engine.js` prepares a configuration once and runs replicates of the Gaussian model.
  - `arena.js` runs the logistic arena: world, every method, oracle tuning on separate replicates, summaries. It is DOM-free.
  - `arena.worker.js` runs the arena in a module Web Worker with progress; cancelling terminates it. Without module workers the page falls back to the main thread.
  - `scheduler.js` chunks work into animation frames so the page stays responsive.
  - `gossip.js` steps the gossip process for animation.
  - `sweep.js` varies one parameter.
- **`viz/`** draws on canvas, with colours read from the CSS tokens so the theme switch redraws correctly.
  - `chart.js`: journal-style figures with method toggles, log/linear axes and hover readouts.
  - `network.js`: the animated network.
  - `diagrams.js`: special drawings.
  - `theme.js`: token reader.
- **`ui/`** builds the interface.
  - `store.js` holds the live configuration and notifies subscribers of changed keys.
  - `controls.js` renders parameters from the schema (sliders, selects, switches, segmented switches).
  - `panels/` has one module per tab, including `start.js`, `arena.js` and `real.js`.
  - `equations.js` renders the paper's equations as native MathML with live values.
  - `inspector.js` is the clicked-hospital readout for either protocol.
  - `metricsTable.js` is the sortable per-hospital table.
  - `tour.js` is the guided tour.

Every module opens with a header comment stating its responsibility. Every exported function has a JSDoc block naming the paper equation or proposition it implements.

## ResearchData design system

The look of a scientific instrument panel crossed with a printed journal figure. The tokens are defined once at the top of `css/researchdata.css`, all prefixed `--rd-`.

- **Colour.** The paper's figure palette, so app and paper read as one system.
  - Dark teal for GHB, light teal for GHB-lite.
  - Amber for privacy noise, red for harm or loss, blue for baselines, violet for robust variants, slate for local estimation.
  - In the arena, warm colours for the federated family and cool ones for personalized learning.
  - The paper theme is cool paper with dark ink; the dark theme is a blackboard. The same semantic colour marks the same method everywhere.
- **Type.**
  - IBM Plex Sans Condensed for labels and controls.
  - IBM Plex Mono with tabular numerals for every number.
  - Literata for explanations and captions.
  - Labels are uppercase with letter-spacing.
- **Form.**
  - 1px hairlines, at most 2px radius, no shadows or gradients.
  - Canvases on a millimetre grid.
  - Sliders drawn as instrument scales with ticks and unit readouts.
  - Every figure has a journal caption (Fig. A1, N1, B1, L1, R1, ...).
  - Equations carry the GHB paper's numbering.
- **Layout.** A parameter rail on the left, the stage in the centre, a lab notebook on the right with explanations and the hospital readout. It stacks to one column at phone width.
- **Motion.** Only to show a process: releases and masked shares travelling along edges, estimates easing during shrinkage, Monte Carlo figures filling in. `prefers-reduced-motion` turns it off.

## Tests

Run all of them with `node --test tests/`. Tests import `js/math/` and the DOM-free `js/sim/arena.js`.

`tests/math.test.mjs` checks the math modules against the paper and the Python code:

1. ICU example: n_eff ≈ 414 and 33 effective events.
2. Riley's n* = 2,707.
3. SLEM 0.977 (ring) and 0.933 (ring with two-hop links) for K = 24.
4. `risk_hb` and `loo_mse` match `scripts/blueprint.py`.
5. zCDP accounting round-trips and beats basic composition.
6. Plug-in GHB-lite mean relative risk ≈ 0.72 on the paper's 24 hospitals.
7. The robust prior shrinks less for a far-away estimate; weighted median; GMRF fixed point.
8. The joint-release closed form matches Simulation 5.

`tests/gauss_ghb.test.mjs` checks Gaussian GHB:

- The noise of the sum, refits and collusion.
- The closed form of M_k against 40,000 simulated replicates.
- On fixed seeds, GHB beats GHB-lite at ε = 0.1 with small hospitals.
- On fixed seeds, GHB-lite beats GHB with 48 large hospitals at ε = 10, and with 48 hospitals of median 100 at ε = 3.

`tests/logit.test.mjs` checks every logistic method against the paper's Python code on a fixed small network, given the same data and the same noise draws (`tests/fixtures/logit_reference.json`; the generating script is in the test's header). It covers:

- The zCDP budget and the MAP fit, including a closed-form case.
- GHB-lite, GHB (v5, including the diagonal release at ε = 0.3) and GHB v4.
- One-shot FedAvg and meta-analysis.
- DP-FedAvg, with and without secure aggregation, decentralized DP-SGD and fine-tuning.
- Ditto, pFedMe, APFL, Per-FedAvg, FedRep and FedProx.

All agree to 10⁻⁸ or better.

`tests/arena.test.mjs` runs the arena headless on Simulation 15 (τ² = 0.05, ε = 1, K = 24, oracle tuning on 10 replicates, 20 replicates). It checks every mean against the paper's values within max(20%, 2.5 combined SE); GHB v4 against the Simulation 15 run made with v4 before the v5 rerun. Excess log-loss in 10⁻³ nats, JS against the paper:

| Method | JS | Paper |
|---|---|---|
| Local | 26.7 | 27.1 |
| GHB | 18.5 | 19.2 |
| GHB v4 (earlier version) | 15.6 | 15.9 |
| GHB-lite | 13.0 | 13.8 |
| Ditto-style | 14.4 | 15.4 |
| DP-FedAvg + FT | 15.5 | 15.3 |
| Decentralized DP-SGD | 19.0 | 19.1 |
| Clustered | 19.1 | 21.4 |
| One-shot FedAvg (Sim. 8) | 32.3 | 36.5 |
| Meta-analysis (Sim. 8) | 27.8 | 30.3 |
| Pooled, no DP | 16.8 | 17.4 |

The same test checks that each method's numbers do not depend on which other methods are selected, and that manual configurations are honoured.
