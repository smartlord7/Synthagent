/**
 * ui/equations.js
 * Responsibility: the paper's display equations as native MathML, numbered as in the paper, each with a
 * second line that substitutes the current values. Pure string builders; no library is needed.
 */
import { fmt } from '../viz/theme.js';

const mi = (x) => `<mi>${x}</mi>`, mo = (x) => `<mo>${x}</mo>`, mnum = (x, d = 3) => `<mn>${fmt(x, d)}</mn>`;
const row = (...a) => `<mrow>${a.join('')}</mrow>`, frac = (a, b) => `<mfrac>${a}${b}</mfrac>`;
const sub = (a, b) => `<msub>${a}${b}</msub>`, sup = (a, b) => `<msup>${a}${b}</msup>`, subsup = (a, b, c) => `<msubsup>${a}${b}${c}</msubsup>`;
const paren = (x) => row(mo('('), x, mo(')'));
const math = (x, block = true) => `<math${block ? ' display="block"' : ''}>${x}</math>`;

/**
 * Display equation block.
 * @param {string} no equation label as in the paper, e.g. 'Eq. (5)' or 'Prop. 8.1'
 * @param {string} body MathML content of the symbolic equation
 * @param {string} [live] MathML content with current values
 * @param {string} [cap] short caption
 * @returns {string} HTML
 */
export function block(no, body, live, cap) {
  return `<div class="rd-eq"><div class="row">${math(body)}<span class="no">${no}</span></div>${live ? `<div class="live">${math(live, false)}</div>` : ''}${cap ? `<p class="cap">${cap}</p>` : ''}</div>`;
}

/** Riley's first criterion with values. @see Eq. (1) */
export function riley(p, R2, S, n) {
  return block('Eq. (1)', row(sup(mi('n'), mo('⋆')), mo('≥'), frac(mi('p'), row(paren(row(mi('S'), mo('−'), mn1())), mi('ln'), paren(row(mn1(), mo('−'), frac(subsup(mi('R'), mi('CS'), mn2()), mi('S'))))))),
    row(sup(mi('n'), mo('⋆')), mo('='), frac(mnum(p, 0), row(paren(row(mnum(S, 2), mo('−'), mn1())), mi('ln'), paren(row(mn1(), mo('−'), frac(mnum(R2, 4), mnum(S, 2)))))), mo('='), mnum(Math.ceil(n), 0)),
    'Sample size at which a p-parameter model reaches global shrinkage S.');
}
const mn1 = () => '<mn>1</mn>', mn2 = () => '<mn>2</mn>';

/** Effective sample size with values. @see Eq. (2) */
export function neff(n, deff, rhoL, eta, rhoS, ne) {
  return block('Eq. (2)', row(sup(mi('n'), mi('eff')), mo('='), mi('n'), mo('·'), frac(mn1(), mi('DEFF')), mo('·'), sub(mi('ρ'), mi('L')), mo('·'), sup(paren(row(mn1(), mo('−'), mn2(), mi('η'))), mn2()), mo('·'), sub(mi('ρ'), mi('S'))),
    row(mnum(n, 0), mo('·'), frac(mn1(), mnum(deff, 2)), mo('·'), mnum(rhoL, 2), mo('·'), sup(paren(mnum(1 - 2 * eta, 2)), mn2()), mo('·'), mnum(rhoS, 2), mo('='), mnum(ne, 1)),
    'DEFF = 1 + (m̄ − 1) ICC is the design effect of repeated records.');
}

/** Smallness index with values. @see Definition 2.3 */
export function smallness(nstar, ne, s) {
  return block('Def. 2.3', row(mi('σ'), mo('='), frac(sup(mi('n'), mo('⋆')), sup(mi('n'), mi('eff')))), row(mi('σ'), mo('='), frac(mnum(Math.ceil(nstar), 0), mnum(ne, 1)), mo('='), mnum(s, 2)),
    'Sufficient ≤ 1 < strained ≤ 10 < small ≤ 100 < extreme.');
}

/** The Gaussian model. @see Eq. (4) */
export function model(mu, tau2, sigma2, nk) {
  return block('Eq. (4)', row(sub(mi('θ'), mi('k')), mo('='), mi('μ'), mo('+'), sub(mi('u'), mi('k')), mo(','), sub(mi('u'), mi('k')), mo('∼'), mi('N'), paren(row('<mn>0</mn>', mo(','), sup(mi('τ'), mn2()))), mo(';'), sub(mover('θ'), mi('k')), mo('='), sub(mi('θ'), mi('k')), mo('+'), sub(mi('e'), mi('k')), mo(','), sub(mi('v'), mi('k')), mo('='), frac(sup(mi('σ'), mn2()), sub(mi('n'), mi('k')))),
    row(mi('μ'), mo('='), mnum(mu, 2), mo(','), sup(mi('τ'), mn2()), mo('='), mnum(tau2, 4), mo(','), sup(mi('σ'), mn2()), mo('='), mnum(sigma2, 2), mo(','), sub(mi('v'), mi('k')), mo('='), frac(mnum(sigma2, 2), mnum(nk, 0)), mo('='), mnum(sigma2 / nk, 4)));
}
const mover = (x) => `<mover>${mi(x)}${mo('^')}</mover>`;

/** Optimal shrinkage for hospital k. @see Proposition 8.1 */
export function alpha(t2, M, v, a, k) {
  return block('Prop. 8.1', row(subsup(mi('α'), mi('k'), mo('⋆')), mo('='), frac(row(sup(mi('τ'), mn2()), mo('+'), mi('M')), row(sup(mi('τ'), mn2()), mo('+'), mi('M'), mo('+'), sub(mi('v'), mi('k')))), mo(','), sub(mover('θ'), mi('k')), mo('←'), sub(mi('α'), mi('k')), sub(mover('θ'), mi('k')), mo('+'), paren(row(mn1(), mo('−'), sub(mi('α'), mi('k')))), sub(mover('μ'), mi('k'))),
    row(sub(mi('α'), mn(k)), mo('='), frac(row(mnum(t2, 4), mo('+'), mnum(M, 4)), row(mnum(t2, 4), mo('+'), mnum(M, 4), mo('+'), mnum(v, 4))), mo('='), mnum(a, 3)),
    `Hospital ${k} keeps ${(100 * a).toFixed(0)}% of its own estimate and borrows ${(100 * (1 - a)).toFixed(0)}% from the network.`);
}
const mn = (x) => `<mn>${x}</mn>`;

/** Borrowed patients. @see Eq. (5) */
export function borrowed(sigma2, t2, M, B, nk) {
  return block('Eq. (5)', row(subsup(mi('n'), mi('k'), mi('eff')), mo('='), frac(sup(mi('σ'), mn2()), subsup(mi('R'), mi('k'), mo('⋆'))), mo('='), sub(mi('n'), mi('k')), mo('+'), frac(sup(mi('σ'), mn2()), row(sup(mi('τ'), mn2()), mo('+'), mi('M')))),
    row(mi('B'), mo('='), frac(mnum(sigma2, 2), row(mnum(t2, 4), mo('+'), mnum(M, 4))), mo('='), mnum(B, 1), mo(';'), subsup(mi('n'), mi('k'), mi('eff')), mo('='), mnum(nk, 0), mo('+'), mnum(B, 1), mo('='), mnum(nk + B, 1)),
    'Every hospital receives about the same number B of virtual patients.');
}

/** Leave-one-out MSE with privacy noise. @see Proposition 8.2, Eq. (6) */
export function looM(K, tau2, sigma2, C, delta, eps, M) {
  return block('Eq. (6)', row(mi('M'), mo('='), frac(mn1(), sup(paren(row(mi('K'), mo('−'), mn1())), mn2())), `<munder>${mo('∑')}${row(mi('j'), mo('≠'), mi('k'))}</munder>`, paren(row(sup(mi('τ'), mn2()), mo('+'), frac(sup(mi('σ'), mn2()), sub(mi('n'), mi('j'))), mo('+'), frac(row('<mn>8</mn>', sup(mi('C'), mn2()), mi('ln'), paren(row('<mn>1.25</mn>', mo('/'), mi('δ')))), row(subsup(mi('n'), mi('j'), mn2()), sup(mi('ε'), mn2())))))),
    row(mi('K'), mo('='), mn(K), mo(','), mi('C'), mo('='), mnum(C, 2), mo(','), mi('δ'), mo('='), mnum(delta), mo(','), mi('ε'), mo('='), isFinite(eps) ? mnum(eps, 3) : mi('∞'), mo('⇒'), sub(mi('M'), mi('k')), mo('='), mnum(M, 5)),
    'Privacy noise is divided by (K − 1) and scales as 1/n_j², so it is expensive only for tiny contributors.');
}

/** Gossip convergence. @see Proposition 8.4 */
export function gossip(lambda, T, gamma, rounds) {
  return block('Prop. 8.4', row(sub(mrow2('‖', sup(mi('x'), mi('t')), mo('−'), mover('x'), '<mn mathvariant="bold">1</mn>', '‖'), mn2()), mo('≤'), sup(mi('λ'), mi('t')), sub(mrow2('‖', sup(mi('x'), '<mn>0</mn>'), mo('−'), mover('x'), '<mn mathvariant="bold">1</mn>', '‖'), mn2())),
    row(mi('λ'), mo('='), mnum(lambda, 3), mo(','), sup(mi('λ'), mn(T)), mo('='), mnum(lambda ** T, 4), mo(';'), frac(row(mi('ln'), paren(row(mn1(), mo('/'), mi('γ')))), row(mn1(), mo('−'), mi('λ'))), mo('='), mnum(rounds, 1), mi(' rounds')),
    'Messages are functions of privatized releases only, so gossip costs no extra privacy.');
}
const mrow2 = (...a) => row(...a.map((x) => (x === '‖' ? mo('‖') : x)));

/** Gaussian mechanism. @see Proposition 8.2 */
export function mechanism(n, C, delta, eps, s) {
  return block('Prop. 8.2', row(sub(mi('s'), mi('j')), mo('='), frac(row(sub(mi('Δ'), mi('j')), msqrt(row(mn2(), mi('ln'), paren(row('<mn>1.25</mn>', mo('/'), mi('δ')))))), mi('ε')), mo(','), sub(mi('Δ'), mi('j')), mo('='), frac(row(mn2(), mi('C')), sub(mi('n'), mi('j')))),
    row(mi('n'), mo('='), mnum(n, 0), mo('⇒'), mi('s'), mo('='), frac(row(mnum(2 * C / n, 4), msqrt(row(mn2(), mi('ln'), paren(mnum(1.25 / delta, 0))))), isFinite(eps) ? mnum(eps, 3) : mi('∞')), mo('='), mnum(s, 4)));
}
const msqrt = (x) => `<msqrt>${x}</msqrt>`;

/** zCDP accounting. @see Simulation 14 */
export function zcdp(rho, delta, eps, r) {
  return block('Sim. 14', row(mi('ρ'), mo('='), frac(sup(mi('Δ'), mn2()), row(mn2(), sup(mi('s'), mn2()))), mo(','), mi('ε'), mo('='), mi('ρ'), mo('+'), mn2(), msqrt(row(mi('ρ'), mi('ln'), paren(row(mn1(), mo('/'), mi('δ')))))),
    row(sub(mi('ρ'), mi('total')), mo('='), mnum(rho, 5), mo(','), sub(mi('ρ'), mi('release')), mo('='), mnum(rho / r, 5), mo(','), mi('ε'), mo('='), mnum(eps, 3)),
    'zCDP budgets add over releases; basic composition splits ε and δ into r equal parts.');
}

/** GMRF Jacobi update. @see Eq. (7) */
export function jacobi(beta, gamma) {
  return block('Eq. (7)', row(sub(mi('θ'), mi('k')), mo('←'), frac(row(sub(mover('θ'), mi('k')), mo('/'), sub(mi('v'), mi('k')), mo('+'), mi('β'), `<munder>${mo('∑')}${mi('l')}</munder>`, sub(mi('a'), mi('kl')), sub(mi('θ'), mi('l')), mo('+'), mi('γ'), sub(mi('μ'), '<mn>0</mn>')), row(mn1(), mo('/'), sub(mi('v'), mi('k')), mo('+'), mi('β'), sub(mi('d'), mi('k')), mo('+'), mi('γ')))),
    row(mi('β'), mo('='), mnum(beta, 1), mo(','), mi('γ'), mo('='), mnum(gamma, 1)),
    'Neighbours exchange smoothed privatized values; each hospital applies the last update with its own noise-free estimate.');
}
