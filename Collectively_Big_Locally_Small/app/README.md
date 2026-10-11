# GHB Simulation Lab

An interactive companion to the survey *Collectively Big, Locally Small*. It measures how small a dataset is for a task (Section 2), then lets you run, watch and stress-test Graph Hierarchical Borrowing (GHB, Section 8). In GHB, hospitals borrow strength from each other over a sparse graph of agreements, with differential privacy and no central server. Every parameter is configurable, every plot of the paper's simulations can be reproduced with your own setup, and every number is explained next to the equation it comes from.

A hosted copy is at https://claude.ai/artifact/7ZYuDy8A9eur912XQNfELs (private until shared from its Share menu).

## Run it

The app is plain HTML, CSS and native ES modules. There is no build step and no npm. ES modules need HTTP, so serve the folder rather than opening the file directly:

```sh
cd app
python3 -m http.server 8000      # then open http://localhost:8000
```

Tests (Node 18 or later):

```sh
node --test tests/math.test.mjs
```

## Tabs and the paper

| Tab | What it does | Paper |
|---|---|---|
| Smallness | Effective sample size, effective events, Riley's n*, smallness index σ and its regime, deflation chart | §2, Eq. (1)–(2), Def. 2.3, Example 2.4 |
| Setup | Every parameter, in one place: network (generated or per-hospital sizes, topology, editable graph, link failures), model, privacy, estimation, safeguards, adversaries, Monte Carlo; presets; JSON export and import | §8.7 design |
| Network & gossip | Animated network: local estimates, privatized releases, gossip rounds with particles on the edges, shrinkage; λ, spectral gap, rounds to tolerance, consensus error; click a hospital for its full readout | §8.4, Prop. 8.4, Fig. 9; Simulations 4, 13 |
| Borrowing & metrics | Monte Carlo of local, pooled, oracle, plug-in GHB, robust prior, opt-out and screened GHB; per-hospital risk, n_eff gain against 1 + B/n_k, sortable metrics table | Props. 8.1–8.2, Eq. (5)–(6); Simulations 1–3 |
| Stress tests | Atypical hospitals, dishonest hospitals, a wrong anchor, with the safeguards | Simulations 9, 11, 12 |
| Privacy | Noise per hospital, B against ε, joint release in groups, refits under basic composition and zCDP | Prop. 8.2; Simulations 5, 14 |
| Regions (GMRF) | Regional site effects; exchangeable against GMRF prior | Eq. (7); Simulation 6 |
| Sweeps & figures | Sweep any parameter (linear or log) and plot any metric with ±2 standard errors; one-click reproductions of the paper's figures | Simulations 2–14 |
| About | Assumptions and limits in plain words | §8.6, §8.7 |

The logistic-regression simulations (7, 8 and 10) need iterative model fitting on thousands of synthetic patients per hospital. They are reproduced by `simulations/` in Python; the app offers their Gaussian analogues. Simulation 15 (iterative federated baselines) is in Python only.

## Architecture

```
index.html ── css/researchdata.css
     │
     └─ js/main.js  (wiring only)
          ├─ js/config.js       schema of every parameter, defaults = the paper's design
          ├─ js/scenarios.js    presets: paper defaults, Simulations 1–14
          ├─ js/ui/             store, tabs, controls, panels/*, inspector, metrics table, tour, equations
          │     └─ uses js/viz/ (canvas figures, network renderer, theme tokens)
          │     └─ uses js/sim/ (engine, gossip, sweep, scheduler)
          └─ js/sim/  ──► js/math/ (pure functions, no DOM)
```

- **`math/`** holds pure, side-effect-free functions; it is the only layer the tests import.
  - `smallness.js`: n_eff, Riley's criterion, σ.
  - `dp.js`: Gaussian mechanism, zCDP accounting.
  - `graph.js`: topologies, Metropolis–Hastings weights, SLEM, link failures.
  - `ghb.js`: DerSimonian–Laird, half-Cauchy τ², leave-one-out inverse-variance mean, shrinkage, robust Student-t prior, opt-out, release screening, anchor conflict check, GMRF Jacobi.
  - `random.js`: seeded generator, normal, Student-t, log-normal sizes.
- **`sim/`** runs Monte Carlo jobs.
  - `engine.js` prepares a configuration once and runs replicates.
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
  - `controls.js` renders parameters from the schema.
  - `panels/` has one module per tab.
  - `equations.js` renders the paper's equations as native MathML with live values.
  - `inspector.js` is the clicked-hospital readout.
  - `metricsTable.js` is the sortable per-hospital table.
  - `tour.js` is the guided tour.

Every module opens with a header comment stating its responsibility. Every exported function has a JSDoc block naming the paper equation or proposition it implements.

## ResearchData design system

The look of a scientific instrument panel crossed with a printed journal figure. The tokens are defined once at the top of `css/researchdata.css`, all prefixed `--rd-`.

- **Colour.** The paper's figure palette, so app and paper read as one system: teal for GHB, amber for privacy noise, red for harm or loss, blue for baselines, violet for robust variants, slate for local estimation. The paper theme is cool paper with dark ink; the dark theme is a blackboard. The same semantic colour marks the same method everywhere.
- **Type.**
  - IBM Plex Sans Condensed for labels and controls.
  - IBM Plex Mono with tabular numerals for every number.
  - Literata for explanations and captions.
  - Labels are uppercase with letter-spacing.
- **Form.**
  - 1px hairlines, at most 2px radius, no shadows or gradients.
  - Canvases on a millimetre grid.
  - Sliders drawn as instrument scales with ticks and unit readouts.
  - Every figure has a journal caption (Fig. A1, N1, B1, …).
  - Equations carry the paper's numbering.
- **Layout.** A parameter rail on the left, the stage in the centre, a lab notebook on the right with explanations and the hospital readout. It stacks to one column at phone width.
- **Motion.** Only to show a process: releases travelling along edges, estimates easing during shrinkage, Monte Carlo figures filling in. `prefers-reduced-motion` turns it off.

## Tests

`tests/math.test.mjs` checks the math modules against the paper and the Python code:

1. ICU example: n_eff ≈ 414 and 33 effective events.
2. Riley's n* = 2,707.
3. SLEM 0.977 (ring) and 0.933 (ring with two-hop links) for K = 24.
4. `risk_hb` and `loo_mse` match `scripts/blueprint.py`.
5. zCDP accounting round-trips and beats basic composition.
6. Plug-in GHB mean relative risk ≈ 0.72 on the paper's 24 hospitals.
7. The robust prior shrinks less for a far-away estimate; weighted median; GMRF fixed point.
8. The joint-release closed form matches Simulation 5.
