/**
 * ui/inspector.js
 * Responsibility: the clicked-hospital readout in the lab notebook: n_k, v_k, the network estimate, M_k, alpha_k,
 * the estimate and its Monte Carlo risk, for GHB (joint release) or GHB-lite (per-hospital release), with one plain
 * sentence on what borrowing does for that hospital.
 */
import { fmt } from '../viz/theme.js';

/**
 * Plain-language fate of a hospital.
 * @param {{n:number, alpha:number, B:number, bad?:boolean}} h hospital record (alpha and B of the protocol shown)
 * @param {number|null} rel Monte Carlo relative risk (or null while running)
 * @returns {{text:string, loses:boolean}}
 * @see Proposition 4.1, Eq. (3)
 */
export function fate(h, rel) {
  const share = Math.round(100 * (1 - h.alpha)), size = h.alpha < 0.4 ? 'Small hospital' : h.alpha < 0.75 ? 'Mid-sized hospital' : 'Large hospital';
  if (h.bad) return { text: `Dishonest hospital: it contributes a fabricated value; its own estimate is still shrunk with weight ${fmt(h.alpha, 2)}.`, loses: false };
  let t = h.alpha < 0.75
    ? `${size}: borrows ${share}% of its estimate from the network and gains about ${fmt(h.B, 0)} virtual patients on top of its ${h.n}.`
    : `${size}: keeps ${100 - share}% of its own estimate; the ${fmt(h.B, 0)} borrowed patients barely change its ${h.n}.`;
  const loses = rel != null && rel > 1.02;
  if (rel != null) t += loses ? ` On average it does worse than alone (relative risk ${fmt(rel, 2)}), which happens to hospitals unlike the others.` : ` Its risk is ${fmt(rel, 2)} times that of going alone, so its effective sample is ${fmt(h.n / rel, 0)} patients.`;
  return { text: t, loses };
}

/**
 * Render the inspector.
 * @param {HTMLElement} el container
 * @param {Object|null} h hospital record (engine.summarize perHospital entry or an equivalent from one replicate)
 * @param {number|null} rel relative risk of the protocol shown
 * @param {'ghb'|'lite'} [protocol='lite'] which protocol's numbers to show
 * @see Proposition 4.1, Section 5.2, Section 5.3
 */
export function renderInspector(el, h, rel, protocol = 'lite') {
  if (!h) { el.innerHTML = '<h3 class="rd-h">Hospital readout</h3><p class="rd-prose">Click a hospital on the network (or a row of the metrics table) to read its numbers.</p>'; return; }
  const joint = protocol === 'ghb';
  const f = fate(joint ? { ...h, alpha: h.alphaJ, B: h.BJ } : h, rel);
  const rows = joint ? [
    ['n<sub>k</sub> patients', h.n], ['v<sub>k</sub> = σ²/n<sub>k</sub>', fmt(h.v, 5)], ['local estimate θ̂<sub>k</sub>', fmt(h.th, 4)],
    ['noise sd of the sum s', fmt(h.sJ, 4)], ['leave-one-out μ<sub>−k</sub>', fmt(h.muJ, 4)], ['M<sub>k</sub> (joint)', fmt(h.MJ, 5)],
    ['τ² (evidence-weighted)', fmt(h.t2J, 4)], ['effective α<sub>k</sub>', fmt(h.alphaJ, 3)], ['weight on not borrowing', fmt(h.wSafe, 3)],
    ['GHB estimate', fmt(h.estJ, 4)], ['true θ<sub>k</sub>', fmt(h.theta, 4)], ['B<sub>k</sub> borrowed (τ² known)', fmt(h.BJ, 1)], ['risk R<sub>k</sub>/v<sub>k</sub>', rel != null ? fmt(rel, 3) : 'running...']
  ] : [
    ['n<sub>k</sub> patients', h.n], ['v<sub>k</sub> = σ²/n<sub>k</sub>', fmt(h.v, 5)], ['local estimate θ̂<sub>k</sub>', fmt(h.th, 4)],
    ['noise sd s<sub>k</sub>', fmt(h.s, 4)], ['release x<sub>k</sub>', fmt(h.x, 4)], ['network estimate μ̃<sub>k</sub>', fmt(h.mu, 4)],
    ['M<sub>k</sub>', fmt(h.M, 5)], ['τ̂²', fmt(h.t2, 4)], ['α<sub>k</sub>', fmt(h.alpha, 3)], ['GHB-lite estimate', fmt(h.est, 4)], ['true θ<sub>k</sub>', fmt(h.theta, 4)],
    ['B<sub>k</sub> borrowed', fmt(h.B, 1)], ['risk R<sub>k</sub>/v<sub>k</sub>', rel != null ? fmt(rel, 3) : 'running...']
  ];
  el.innerHTML = `<h3 class="rd-h">Hospital ${h.k + 1} readout, ${joint ? 'GHB' : 'GHB-lite'}</h3><dl>${rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join('')}</dl><p class="rd-prose rd-fate ${f.loses ? 'loses' : ''}">${f.text}</p>`;
}
