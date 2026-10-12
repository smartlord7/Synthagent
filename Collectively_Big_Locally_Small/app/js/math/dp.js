/**
 * math/dp.js
 * Responsibility: differential privacy arithmetic: the Gaussian mechanism of Proposition 4.2
 * and composition over repeated releases (basic composition and zero-concentrated DP, Simulation 14).
 */

/**
 * l2 sensitivity of a mean of per-patient contributions clipped to norm C.
 * @param {number} n hospital size
 * @param {number} C clipping norm
 * @returns {number} Delta = 2C / n
 * @see Proposition 4.2
 */
export function sensitivity(n, C) { return 2 * C / n; }

/**
 * Noise standard deviation of the classical Gaussian mechanism.
 * @param {number} Delta l2 sensitivity
 * @param {number} eps privacy budget epsilon (Infinity means no privacy)
 * @param {number} delta privacy parameter delta
 * @returns {number} s = Delta sqrt(2 ln(1.25/delta)) / eps
 * @see Proposition 4.2
 */
export function gaussianSigma(Delta, eps, delta) {
  if (!isFinite(eps)) return 0;
  return Delta * Math.sqrt(2 * Math.log(1.25 / delta)) / eps;
}

/**
 * Release noise variance s_j^2 for a hospital of size n (blueprint.dp_noise_var).
 * @param {number} n hospital size
 * @param {number} eps epsilon (Infinity: no privacy, returns 0)
 * @param {number} delta delta
 * @param {number} C clipping norm
 * @returns {number} s_j^2 = 8 C^2 ln(1.25/delta) / (n^2 eps^2)
 * @see Proposition 4.2, Eq. (4)
 */
export function dpVar(n, eps, delta, C) { return gaussianSigma(sensitivity(n, C), eps, delta) ** 2; }

/**
 * Per-release noise under basic composition: r releases each with (eps/r, delta/r).
 * @param {number} Delta sensitivity
 * @param {number} eps total epsilon
 * @param {number} delta total delta
 * @param {number} r number of releases
 * @returns {number} noise sd per release
 * @see Section 3 (threat model), Simulation 14
 */
export function noiseBasic(Delta, eps, delta, r) { return gaussianSigma(Delta, eps / r, delta / r); }

/**
 * Total zCDP budget rho whose conversion rho + 2 sqrt(rho ln(1/delta)) equals eps.
 * @param {number} eps total epsilon
 * @param {number} delta delta
 * @returns {number} rho
 * @see Bun and Steinke (2016), Simulation 14
 */
export function zcdpRho(eps, delta) {
  const L = Math.log(1 / delta), a = -Math.sqrt(L) + Math.sqrt(L + eps);
  return a * a;
}

/**
 * Convert rho-zCDP to (eps, delta)-DP.
 * @param {number} rho zCDP parameter
 * @param {number} delta delta
 * @returns {number} eps = rho + 2 sqrt(rho ln(1/delta))
 * @see Simulation 14
 */
export function zcdpToEps(rho, delta) { return rho + 2 * Math.sqrt(rho * Math.log(1 / delta)); }

/**
 * Per-release noise when rho adds over r Gaussian releases, each with rho_r = Delta^2 / (2 s^2).
 * @param {number} Delta sensitivity
 * @param {number} eps total epsilon
 * @param {number} delta delta
 * @param {number} r number of releases
 * @returns {number} noise sd per release
 * @see Simulation 14
 */
export function noiseZcdp(Delta, eps, delta, r) { return Delta / Math.sqrt(2 * zcdpRho(eps, delta) / r); }

/**
 * Noise sd of a Gaussian mechanism that spends the zCDP budget rho on one release of sensitivity Delta.
 * @param {number} Delta l2 sensitivity
 * @param {number} rho zCDP budget (Infinity: no privacy, returns 0)
 * @returns {number} s = Delta / sqrt(2 rho)
 * @see Proposition 5.1 (rho = Delta^2 / (2 s^2) per release)
 */
export function zcdpSigma(Delta, rho) { return isFinite(rho) ? Delta / Math.sqrt(2 * rho) : 0; }

/**
 * Noise sd of GHB's joint release in the Gaussian model: the sum of every hospital's clipped per-patient
 * contributions has sensitivity 2C whatever the hospital sizes, and receives the whole budget (rho / refits per refit).
 * @param {number} C clipping bound
 * @param {number} eps total epsilon (Infinity: no privacy)
 * @param {number} delta delta
 * @param {number} [refits=1] releases on the same patients (zCDP composition)
 * @returns {number} s = 2C / sqrt(2 rho / refits)
 * @see Section 5.2 (noise calibrated to a sum, not to each hospital), Proposition 5.1
 */
export function jointNoiseSd(C, eps, delta, refits = 1) { return isFinite(eps) ? zcdpSigma(2 * C, zcdpRho(eps, delta) / refits) : 0; }

/**
 * The guarantee left for honest hospitals' patients when c of K hospitals collude and remove their noise shares.
 * @param {number} rho zCDP budget of the joint releases
 * @param {number} K hospitals
 * @param {number} c colluding hospitals (c < K)
 * @returns {number} rho K / (K - c)
 * @see Proposition 5.1 (collusion)
 */
export function collusionRho(rho, K, c) { return rho * K / Math.max(K - c, 1e-9); }
