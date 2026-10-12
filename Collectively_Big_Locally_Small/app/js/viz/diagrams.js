/**
 * viz/diagrams.js
 * Responsibility: special-purpose drawings used through Figure.setCustom: the deflation bars of
 * Figure 2 (n to n_eff) and the shrinkage strip that shows each hospital's local estimate, network
 * estimate, GHB estimate and true value on one axis.
 */
import { tok, fmt } from './theme.js';

/**
 * Deflation bars: the sample left after each mechanism of survey Eq. (2), with Riley's n* as a dashed line.
 * @param {CanvasRenderingContext2D} c context
 * @param {number} w width
 * @param {number} h height
 * @param {number[]} steps [n, after clustering, after labels, after noise, after shift]
 * @param {number} nstar Riley n*
 * @see survey Eq. (1), survey Eq. (2), Figure 2
 */
export function drawDeflation(c, w, h, steps, nstar) {
  const labels = ['nominal n', '÷ DEFF (clustering)', '× ρ_L (labels)', '× (1−2η)² (noise)', '× ρ_S (shift) = n_eff'];
  const max = Math.max(steps[0], nstar) * 1.06, L = Math.min(168, w * 0.4), R = 62, rowH = (h - 34) / steps.length;
  const X = (v) => L + (v / max) * (w - L - R), ui = tok('--rd-font-ui'), mono = tok('--rd-font-mono');
  c.textBaseline = 'middle';
  for (let i = 0; i < steps.length; i++) {
    const y = 8 + i * rowH, bh = rowH * 0.56, top = y + (rowH - bh) / 2, last = i === steps.length - 1;
    if (i > 0) { c.strokeStyle = tok('--rd-risk'); c.setLineDash([2, 3]); c.strokeRect(X(steps[i]) + 0.5, top + 0.5, X(steps[i - 1]) - X(steps[i]), bh); c.setLineDash([]); }
    c.fillStyle = last ? tok('--rd-ghb') : tok('--rd-local'); c.fillRect(L, top, X(steps[i]) - L, bh);
    c.fillStyle = tok('--rd-muted'); c.font = `12px ${ui}`; c.textAlign = 'right'; c.fillText(labels[i], L - 8, y + rowH / 2);
    c.fillStyle = tok('--rd-ink'); c.font = `12px ${mono}`; c.textAlign = 'left'; c.fillText(Math.round(steps[i]).toLocaleString('en-US'), X(steps[i]) + 5, y + rowH / 2);
  }
  c.strokeStyle = tok('--rd-risk'); c.setLineDash([5, 4]); c.beginPath(); c.moveTo(X(nstar), 4); c.lineTo(X(nstar), h - 24); c.stroke(); c.setLineDash([]);
  c.fillStyle = tok('--rd-risk'); c.font = `12px ${mono}`; c.textAlign = 'center'; c.fillText('Riley n* = ' + Math.ceil(nstar).toLocaleString('en-US'), Math.min(Math.max(X(nstar), 70), w - 70), h - 12);
}

/**
 * Shrinkage strip: one row per hospital (sorted by size): local estimate, network estimate, current
 * estimate after a fraction t of the shrinkage, and the true value.
 * @param {CanvasRenderingContext2D} c context
 * @param {number} w width
 * @param {number} h height
 * @param {{n:number[], th:ArrayLike<number>, mu:ArrayLike<number>, est:ArrayLike<number>, theta:ArrayLike<number>, alpha:ArrayLike<number>}} d hospitals
 * @param {number} t shrinkage progress in [0, 1]
 * @param {number} sel selected hospital or -1
 * @see Proposition 4.1: estimate = alpha th + (1 - alpha) mu
 */
export function drawShrinkStrip(c, w, h, d, t, sel) {
  const K = d.n.length, L = 46, R = 12, T = 8, B = 26, rowH = (h - T - B) / K, mono = tok('--rd-font-mono');
  let lo = Infinity, hi = -Infinity;
  for (let k = 0; k < K; k++) for (const v of [d.th[k], d.mu[k], d.theta[k]]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = (hi - lo) * 0.05 || 0.1; lo -= pad; hi += pad;
  const X = (v) => L + (v - lo) / (hi - lo) * (w - L - R);
  c.font = `10px ${mono}`; c.textBaseline = 'middle';
  for (let k = 0; k < K; k++) {
    const y = T + (k + 0.5) * rowH, cur = d.th[k] + t * (d.est[k] - d.th[k]);
    if (k === sel) { c.fillStyle = tok('--rd-sel'); c.fillRect(0, y - rowH / 2, w, rowH); }
    c.strokeStyle = tok('--rd-rule'); c.lineWidth = 1; c.beginPath(); c.moveTo(X(d.th[k]), y); c.lineTo(X(d.mu[k]), y); c.stroke();
    c.fillStyle = tok('--rd-local'); c.fillRect(X(d.th[k]) - 1, y - rowH * 0.35, 2, rowH * 0.7);
    c.fillStyle = tok('--rd-ref'); c.fillRect(X(d.mu[k]) - 1, y - rowH * 0.35, 2, rowH * 0.7);
    c.strokeStyle = tok('--rd-ink'); c.beginPath(); c.moveTo(X(d.theta[k]), y - rowH * 0.42); c.lineTo(X(d.theta[k]), y + rowH * 0.42); c.stroke();
    c.fillStyle = tok('--rd-ghb'); c.beginPath(); c.arc(X(cur), y, Math.min(4, rowH * 0.38), 0, 2 * Math.PI); c.fill();
    c.fillStyle = tok('--rd-muted'); c.textAlign = 'right'; if (rowH >= 7 || k % 3 === 0) c.fillText(String(d.n[k]), L - 6, y);
  }
  c.fillStyle = tok('--rd-muted'); c.textAlign = 'center'; c.fillText(fmt(lo, 2), X(lo) + 12, h - 10); c.fillText(fmt(hi, 2), X(hi) - 12, h - 10);
  c.fillText('parameter value θ', w / 2, h - 10);
}
