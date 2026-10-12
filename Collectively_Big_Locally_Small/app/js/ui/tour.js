/**
 * ui/tour.js
 * Responsibility: the guided tour. Steps switch tabs, outline the element under discussion and explain it
 * in a small card with Back / Next / Close (Escape closes). Steps may run an action (e.g. play the gossip).
 */

/** Tour steps: smallness, the network, what leaves a hospital, gossip, borrowing, the literature arena, real data, stress tests. */
export const TOUR_STEPS = [
  { tab: 'calc', target: '#fig-deflate', title: '1. How small is small?', text: 'A dataset is small for a task when its effective sample size n_eff falls below the sample the task needs (GHB paper Section 2.1, Eq. 1). Clustering, missing labels, label noise and distribution shift each deflate n; Riley\'s criterion gives the requirement n*. The ICU example: 2,000 stays shrink to about 414 effective records against 2,707 needed.' },
  { tab: 'network', target: '#net-canvas', title: '2. The network', text: 'Each disc is a hospital, its area proportional to its patients. Lines are data-sharing agreements: there is no central server, and hospitals only talk to neighbours.' },
  { tab: 'network', target: '#net-canvas', action: 'noise', title: '3. What leaves a hospital', text: 'Under GHB, each hospital sends a masked share of the network sum: pairwise masks that cancel in the total plus a share of the privacy noise (amber halo). Under GHB-lite (switch the protocol), each hospital releases its own estimate plus noise that grows as 1/n_k. A hospital always keeps its own noise-free estimate for its final step.' },
  { tab: 'network', target: '#fig-consensus', action: 'gossip', title: '4. Gossip', text: 'In each round every hospital averages with its neighbours (the particles). Values mix and the consensus error falls like λ^t, where λ is the second largest eigenvalue modulus of the gossip matrix (Proposition 4.3). Gossip uses only privatized or masked values, so it costs no extra privacy.' },
  { tab: 'network', target: '#net-strip', action: 'shrink', title: '5. Borrowing', text: 'Each hospital shrinks its own estimate toward its leave-one-out network estimate, by an amount its own evidence decides (Proposition 4.1, Remark 4.4). Small hospitals move far; large ones barely move. Click any hospital to read its numbers.' },
  { tab: 'borrow', target: '#fig-rel', title: '6. Who gains, under each protocol', text: 'Monte Carlo over thousands of replicates: relative risk below 1 means a hospital does better than alone. Dark teal is GHB, light teal GHB-lite; the protocol switch shows one or both. Every hospital receives about the same B virtual patients, a large share for a small hospital.' },
  { tab: 'arena', target: '#arena-run', title: '7. Against the literature', text: 'The Literature arena runs GHB, GHB-lite and every method of the paper (FedAvg, meta-analysis, Ditto, DP-FedAvg with fine-tuning and secure aggregation, decentralized and clustered DP-SGD, pFedMe, APFL, Per-FedAvg, FedRep, FedProx) at the same privacy budget, with paper-default, oracle or manual tuning.' },
  { tab: 'real', target: '#real-t10', title: '8. Real hospitals', text: 'On five cancer networks that no part of its development had seen, and under pre-registration, GHB was more accurate than every federated and personalized method tuned as a consortium could tune it (Section 7.1).' },
  { tab: 'stress', target: '#fig-atyp', title: '9. Stress tests', text: 'Propositions hold on average over hospitals. A hospital far from the others can lose; evidence weighting, a robust Student-t prior and an opt-out rule limit the damage. Dishonest hospitals and failing links are tested on this tab and on the Sweeps tab.' }
];

/**
 * Create the tour controller.
 * @param {{show:(tab:string)=>void}} tabs tab controller
 * @param {(action:string)=>void} runAction performs a step's action
 * @returns {{start:()=>void, stop:()=>void}}
 */
export function createTour(tabs, runAction) {
  let i = -1, card = null, target = null;
  function stop() { card?.remove(); card = null; target?.classList.remove('rd-tour-target'); target = null; i = -1; document.removeEventListener('keydown', onKey); }
  function onKey(e) { if (e.key === 'Escape') stop(); }
  function go(k) {
    if (k < 0 || k >= TOUR_STEPS.length) { stop(); return; }
    i = k; const s = TOUR_STEPS[i];
    tabs.show(s.tab);
    target?.classList.remove('rd-tour-target');
    target = document.querySelector(s.target);
    if (!card) {
      card = document.createElement('div'); card.className = 'rd-tour'; card.setAttribute('role', 'dialog'); card.setAttribute('aria-label', 'Guided tour');
      document.body.appendChild(card); document.addEventListener('keydown', onKey);
    }
    card.innerHTML = `<div class="step">step ${i + 1} of ${TOUR_STEPS.length}</div><h3>${s.title}</h3><p>${s.text}</p><div class="rd-btns"><button class="rd-btn" id="tour-back" ${i === 0 ? 'disabled' : ''}>Back</button><button class="rd-btn primary" id="tour-next">${i === TOUR_STEPS.length - 1 ? 'Finish' : 'Next'}</button><button class="rd-btn" id="tour-close">Close</button></div>`;
    card.querySelector('#tour-back').onclick = () => go(i - 1);
    card.querySelector('#tour-next').onclick = () => go(i + 1);
    card.querySelector('#tour-close').onclick = stop;
    requestAnimationFrame(() => {
      if (target) { target.classList.add('rd-tour-target'); target.scrollIntoView({ block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }); }
      const r = target?.getBoundingClientRect(), cw = card.offsetWidth, ch = card.offsetHeight;
      let left = 16, top = window.innerHeight - ch - 16;
      if (r && window.innerWidth > 760) { left = Math.min(window.innerWidth - cw - 16, Math.max(16, r.right - cw)); top = Math.min(window.innerHeight - ch - 16, Math.max(16, r.top + 16)); }
      card.style.left = left + 'px'; card.style.top = top + 'px';
      card.querySelector('#tour-next').focus();
      if (s.action) runAction(s.action);
    });
  }
  return { start: () => go(0), stop };
}
