/**
 * math/smallness.js
 * Responsibility: the task-relative definition of small data of Section 2:
 * effective sample size (Eq. 2), Riley's first criterion (Eq. 1) and the smallness index.
 */

/**
 * Kish design effect for repeated records per patient.
 * @param {number} mbar mean records per patient
 * @param {number} icc intra-class correlation
 * @returns {number} DEFF = 1 + (mbar - 1) ICC
 * @see Section 2, survey Eq. (2)
 */
export function designEffect(mbar, icc) { return 1 + (mbar - 1) * icc; }

/**
 * Effective sample size and the sample left after each deflating mechanism, in the order of survey Eq. (2).
 * @param {number} n nominal number of records
 * @param {number} mbar records per patient
 * @param {number} icc intra-class correlation
 * @param {number} rhoL labeled fraction (labels missing completely at random)
 * @param {number} eta symmetric label-noise rate, < 1/2
 * @param {number} rhoS Kish efficiency of the shift importance weights
 * @returns {{deff:number, steps:number[], neff:number}} steps = [n, after clustering, after labels, after noise, after shift]
 * @see Definition 2.2, survey Eq. (2): n_eff = n (1/DEFF) rho_L (1 - 2 eta)^2 rho_S
 */
export function neffParts(n, mbar, icc, rhoL, eta, rhoS) {
  const deff = designEffect(mbar, icc);
  const s1 = n / deff, s2 = s1 * rhoL, s3 = s2 * (1 - 2 * eta) ** 2, s4 = s3 * rhoS;
  return { deff, steps: [n, s1, s2, s3, s4], neff: s4 };
}

/**
 * Effective number of events for a binary outcome.
 * @param {number} neff effective sample size
 * @param {number} prev outcome prevalence
 * @returns {number} n_eff min(pi, 1 - pi)
 * @see Section 2, text after survey Eq. (2)
 */
export function effectiveEvents(neff, prev) { return neff * Math.min(prev, 1 - prev); }

/**
 * Maximum Cox-Snell R^2 of a logistic model at a given prevalence: 1 - exp(2 lnL0 / n).
 * @param {number} prev outcome prevalence
 * @returns {number} max R^2_CS
 * @see Riley et al. (2019), used with survey Eq. (1)
 */
export function r2csMax(prev) { return 1 - Math.exp(2 * (prev * Math.log(prev) + (1 - prev) * Math.log(1 - prev))); }

/**
 * Riley's first criterion: sample size at which a p-parameter model reaches global shrinkage S.
 * @param {number} p candidate parameters
 * @param {number} frac anticipated R^2_CS as a share of its maximum
 * @param {number} prev outcome prevalence
 * @param {number} S target shrinkage (typically 0.9)
 * @returns {number} n* (not rounded)
 * @see survey Eq. (1): n* >= p / ((S - 1) ln(1 - R^2_CS / S))
 */
export function rileyN(p, frac, prev, S) {
  const R2 = frac * r2csMax(prev);
  return p / ((S - 1) * Math.log(1 - R2 / S));
}

/**
 * Smallness regime of Definition 2.3.
 * @param {number} sigma smallness index n* / n_eff
 * @returns {{name:string, level:'good'|'warn'|'bad'}} sufficient (<=1), strained (<=10), small (<=100), extreme
 * @see Definition 2.3
 */
export function regime(sigma) {
  if (sigma <= 1) return { name: 'sufficient', level: 'good' };
  if (sigma <= 10) return { name: 'strained', level: 'warn' };
  if (sigma <= 100) return { name: 'small', level: 'bad' };
  return { name: 'extreme', level: 'bad' };
}
