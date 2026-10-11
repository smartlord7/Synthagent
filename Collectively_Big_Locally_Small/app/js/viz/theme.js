/**
 * viz/theme.js
 * Responsibility: read ResearchData colour and type tokens from CSS so every canvas follows the active
 * theme, notify renderers when the theme changes, and provide the colour scales used on canvases.
 */

/**
 * Current value of a CSS custom property on the root element.
 * @param {string} name token name, e.g. '--rd-ghb'
 * @returns {string} colour or font value
 */
export function tok(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888888';
}

/**
 * Parse '#rrggbb', '#rgb' or 'rgb(r, g, b)' into channels.
 * @param {string} c colour
 * @returns {number[]} [r, g, b]
 */
export function rgb(c) {
  if (c.startsWith('rgb')) return c.match(/[\d.]+/g).slice(0, 3).map(Number);
  let h = c.replace('#', ''); if (h.length === 3) h = h.split('').map((x) => x + x).join('');
  const n = parseInt(h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Linear blend of two colours.
 * @param {string} a colour at t = 0
 * @param {string} b colour at t = 1
 * @param {number} t blend in [0, 1]
 * @returns {string} rgb() colour
 */
export function mix(a, b, t) { const A = rgb(a), B = rgb(b); return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(',')})`; }

/**
 * Colour with alpha.
 * @param {string} c colour
 * @param {number} a alpha
 * @returns {string} rgba() colour
 */
export function alpha(c, a) { const [r, g, b] = rgb(c); return `rgba(${r},${g},${b},${a})`; }

/**
 * Diverging scale for node values: below-centre values toward --rd-div-neg, above toward --rd-div-pos.
 * @param {number} t standardized value in [-1, 1]
 * @returns {string}
 */
export function diverging(t) {
  const m = tok('--rd-node'), c = Math.max(-1, Math.min(1, t));
  return c < 0 ? mix(m, tok('--rd-div-neg'), -c) : mix(m, tok('--rd-div-pos'), c);
}

/**
 * Sequential scale for heatmaps (paper to data teal).
 * @param {number} t value in [0, 1]
 * @returns {string}
 */
export function sequential(t) { return mix(tok('--rd-surface'), tok('--rd-ghb'), Math.max(0, Math.min(1, t))); }

const listeners = new Set();
/**
 * Register a redraw callback for theme changes (system preference or the host's data-theme attribute).
 * @param {()=>void} cb callback
 */
export function onThemeChange(cb) { listeners.add(cb); }
if (typeof window !== 'undefined') {
  const fire = () => setTimeout(() => listeners.forEach((f) => f()), 30);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', fire);
  new MutationObserver(fire).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });
}

/** Whether the viewer asked for reduced motion. */
export const reducedMotion = typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * Prepare a canvas for drawing at device pixel ratio.
 * @param {HTMLCanvasElement} cv canvas (its CSS width is the layout width; data-h is the height in px)
 * @returns {{c:CanvasRenderingContext2D, w:number, h:number}|null} null when the canvas is not visible
 */
export function setupCanvas(cv) {
  const w = cv.getBoundingClientRect().width; if (!w) return null;
  const h = Math.round((+cv.dataset.h || 260) * (window.innerWidth < 560 ? 0.85 : 1)), dpr = window.devicePixelRatio || 1;
  cv.style.height = h + 'px'; cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { c, w, h };
}

/**
 * Faint millimetre-paper grid (1 mm minor, 5 mm major at 96 dpi) inside a rectangle.
 * @param {CanvasRenderingContext2D} c context
 * @param {number} x left
 * @param {number} y top
 * @param {number} w width
 * @param {number} h height
 */
export function mmGrid(c, x, y, w, h) {
  const mm = 3.78, minor = tok('--rd-mm'), major = tok('--rd-mm-major');
  c.save(); c.beginPath(); c.rect(x, y, w, h); c.clip(); c.lineWidth = 1;
  for (let i = 0, gx = x; gx <= x + w; i++, gx += mm) { c.strokeStyle = i % 5 ? minor : major; c.beginPath(); c.moveTo(Math.round(gx) + 0.5, y); c.lineTo(Math.round(gx) + 0.5, y + h); c.stroke(); }
  for (let i = 0, gy = y; gy <= y + h; i++, gy += mm) { c.strokeStyle = i % 5 ? minor : major; c.beginPath(); c.moveTo(x, Math.round(gy) + 0.5); c.lineTo(x + w, Math.round(gy) + 0.5); c.stroke(); }
  c.restore();
}

/**
 * Compact number formatting for readouts (scientific notation for very small or large values).
 * @param {number} x value
 * @param {number} [d=3] significant decimals
 * @returns {string}
 */
export function fmt(x, d = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return '–';
  if (!isFinite(x)) return '∞';
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(2);
  if (a >= 100) return x.toFixed(0);
  if (a >= 10) return x.toFixed(Math.max(0, d - 2));
  return x.toFixed(d);
}
