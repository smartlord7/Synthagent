/**
 * viz/chart.js
 * Responsibility: journal-style canvas figures. A Figure owns its DOM (toolbar with method toggles and
 * log/linear switches, canvas on millimetre paper, hover readout, numbered caption) and draws line/point
 * series with Monte Carlo error bars, reference lines, markers, categorical axes, heatmaps or a custom
 * drawing. Colours come from ResearchData tokens and are re-read on every draw, so theme changes apply.
 */
import { tok, setupCanvas, mmGrid, fmt, onThemeChange, sequential, alpha } from './theme.js';

const figures = new Set();
onThemeChange(() => figures.forEach((f) => f.draw()));

function niceTicks(a, b, n = 5) {
  const span = b - a || 1, st = 10 ** Math.floor(Math.log10(span / n)), err = span / n / st;
  const step = st * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1), t = [];
  for (let x = Math.ceil(a / step) * step; x <= b + step * 1e-9; x += step) t.push(+x.toPrecision(12));
  return t;
}
function logTicks(a, b) {
  const t = [], lo = Math.floor(Math.log10(a)), hi = Math.ceil(Math.log10(b)), mult = hi - lo <= 2 ? [1, 2, 5] : [1];
  for (let e = lo; e <= hi; e++) for (const m of mult) { const x = m * 10 ** e; if (x >= a * 0.999 && x <= b * 1.001) t.push(x); }
  return t.length >= 2 ? t : [a, b];
}
const tickLabel = (t) => { const a = Math.abs(t); if (a !== 0 && (a < 1e-3 || a >= 1e5)) return t.toExponential(0); return String(+t.toPrecision(4)); };

/** A numbered journal figure on canvas. */
export class Figure {
  /**
   * @param {HTMLElement} host element to render into
   * @param {{id:string, num:string, caption:string, xLabel:string, yLabel:string, height?:number, logX?:boolean, logY?:boolean,
   *          canLogX?:boolean, canLogY?:boolean, toggles?:boolean}} o options; ids of controls derive from `id`
   */
  constructor(host, o) {
    this.o = { height: 260, canLogX: true, canLogY: true, toggles: true, ...o };
    this.logX = !!o.logX; this.logY = !!o.logY; this.hidden = new Set(); this.data = null; this.hover = null;
    const f = document.createElement('figure'); f.className = 'rd-fig'; f.id = o.id;
    f.innerHTML = `<div class="rd-fig-tools"><div class="rd-fig-series" id="${o.id}-series"></div><div class="rd-fig-axes">
      ${this.o.canLogX ? `<label class="rd-switch"><input type="checkbox" id="${o.id}-logx" ${this.logX ? 'checked' : ''}> log x</label>` : ''}
      ${this.o.canLogY ? `<label class="rd-switch"><input type="checkbox" id="${o.id}-logy" ${this.logY ? 'checked' : ''}> log y</label>` : ''}</div></div>
      <div class="rd-canvas-wrap"><canvas class="rd-canvas" id="${o.id}-cv" data-h="${this.o.height}" tabindex="0" aria-label="${o.caption.replace(/"/g, '')}"></canvas><div class="rd-tip" hidden></div></div>
      <figcaption><span class="rd-fignum">Fig. ${o.num}</span> ${o.caption}</figcaption>`;
    host.appendChild(f);
    this.el = f; this.cv = f.querySelector('canvas'); this.tip = f.querySelector('.rd-tip'); this.seriesBox = f.querySelector('.rd-fig-series');
    f.querySelector(`#${o.id}-logx`)?.addEventListener('change', (e) => { this.logX = e.target.checked; this.draw(); });
    f.querySelector(`#${o.id}-logy`)?.addEventListener('change', (e) => { this.logY = e.target.checked; this.draw(); });
    this.cv.addEventListener('mousemove', (e) => { const r = this.cv.getBoundingClientRect(); this.hover = [e.clientX - r.left, e.clientY - r.top]; this.draw(); });
    this.cv.addEventListener('mouseleave', () => { this.hover = null; this.tip.hidden = true; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(f.querySelector('.rd-canvas-wrap'));
    figures.add(this);
  }

  /** Set the caption text (after the figure number). */
  setCaption(text) { this.el.querySelector('figcaption').innerHTML = `<span class="rd-fignum">Fig. ${this.o.num}</span> ${text}`; }

  /**
   * Set line/point data.
   * @param {{series:Array<{id:string,label:string,tok:string,pts:number[][],dash?:boolean,marker?:number,width?:number,noLine?:boolean,toggle?:boolean}>,
   *          hlines?:Array<{y:number,tok?:string,dash?:boolean,label?:string}>, vlines?:Array<{x:number,tok?:string,label?:string}>,
   *          marks?:Array<{x:number,y:number,tok:string,label?:string}>, xCats?:string[], yMin?:number, yMax?:number, xMin?:number, xMax?:number,
   *          xLabel?:string, yLabel?:string}} d data; pts are [x, y] or [x, y, se]
   */
  setData(d) { this.data = d; this.mode = 'xy'; this.buildToggles(); this.draw(); }

  /**
   * Set heatmap data.
   * @param {{xs:number[], ys:number[], z:number[][], zLabel:string, xLabel?:string, yLabel?:string, logZ?:boolean}} d z[iy][ix]
   */
  setHeatmap(d) { this.data = d; this.mode = 'heat'; this.seriesBox.innerHTML = ''; this.draw(); }

  /**
   * Custom drawing callback (used for bar diagrams and the shrinkage strip).
   * @param {(c:CanvasRenderingContext2D, w:number, h:number)=>void} fn drawing function
   */
  setCustom(fn) { this.data = fn; this.mode = 'custom'; this.seriesBox.innerHTML = ''; this.draw(); }

  buildToggles() {
    if (!this.o.toggles) { this.seriesBox.innerHTML = ''; return; }
    const ser = this.data.series.filter((s) => s.toggle !== false), sig = ser.map((s) => s.id).join('|');
    if (sig === this.sig) return;
    this.sig = sig;
    this.seriesBox.innerHTML = ser.map((s) => `<label class="rd-key" style="--c:var(${s.tok})"><input type="checkbox" id="${this.o.id}-s-${s.id}" ${this.hidden.has(s.id) ? '' : 'checked'}><span class="rd-swatch ${s.dash ? 'dash' : ''}"></span>${s.label}</label>`).join('');
    ser.forEach((s) => this.seriesBox.querySelector(`#${this.o.id}-s-${s.id}`).addEventListener('change', (e) => { if (e.target.checked) this.hidden.delete(s.id); else this.hidden.add(s.id); this.draw(); }));
  }

  draw() {
    if (!this.data) return;
    const S = setupCanvas(this.cv); if (!S) return;
    const { c, w, h } = S;
    c.clearRect(0, 0, w, h);
    if (this.mode === 'custom') { mmGrid(c, 0, 0, w, h); this.data(c, w, h); return; }
    if (this.mode === 'heat') return this.drawHeat(c, w, h);
    this.drawXY(c, w, h);
  }

  frame(c, w, h, xl, yl) {
    const L = 58, R = 14, T = 12, B = 44;
    return { L, R, T, B, pw: w - L - R, ph: h - T - B };
  }

  drawXY(c, w, h) {
    const d = this.data, ink = tok('--rd-ink'), muted = tok('--rd-muted'), rule = tok('--rd-rule'), mono = tok('--rd-font-mono'), ui = tok('--rd-font-ui');
    const { L, T, pw, ph } = this.frame(c, w, h);
    mmGrid(c, L, T, pw, ph);
    const vis = d.series.filter((s) => !this.hidden.has(s.id));
    const xs = [], ys = [];
    const logY = this.logY && !d.yCats; this.logYeff = logY;
    for (const s of vis) for (const p of s.pts) if (isFinite(p[0]) && isFinite(p[1]) && (!logY || p[1] > 0) && (!this.logX || p[0] > 0)) {
      xs.push(p[0]); ys.push(p[1]);
      if (p[2] && s.xerr) { xs.push(p[0] + 2 * p[2]); if (!this.logX || p[0] - 2 * p[2] > 0) xs.push(p[0] - 2 * p[2]); }
      else if (p[2]) { ys.push(p[1] + 2 * p[2]); if (!logY || p[1] - 2 * p[2] > 0) ys.push(p[1] - 2 * p[2]); }
    }
    for (const l of d.vlines || []) xs.push(l.x);
    for (const l of d.hlines || []) if (!logY || l.y > 0) ys.push(l.y);
    if (!xs.length) { c.fillStyle = muted; c.font = `13px ${ui}`; c.fillText(d.empty || 'Running the simulation…', L + 10, T + 22); return; }
    let x0 = d.xMin ?? Math.min(...xs), x1 = d.xMax ?? Math.max(...xs), y0 = d.yMin ?? Math.min(...ys), y1 = d.yMax ?? Math.max(...ys);
    if (d.xCats) { x0 = -0.5; x1 = d.xCats.length - 0.5; }
    if (this.logX && !d.xCats) { x0 = Math.max(x0, 1e-12); if (x1 <= x0) x1 = x0 * 10; } else if (x1 === x0) { x0 -= 0.5; x1 += 0.5; }
    if (d.yCats) { y0 = -0.6; y1 = d.yCats.length - 0.4; }
    else if (logY) { y0 = Math.max(Math.min(y0, y1) / 1.25, 1e-12); y1 = y1 * 1.25; if (y1 <= y0) y1 = y0 * 10; }
    else { if (d.yZero !== false && y0 > 0 && y0 < (y1 - y0) * 0.6) y0 = 0; const pad = (y1 - y0) * 0.07 || 0.1; y1 += pad; if (d.yMin == null && y0 !== 0) y0 -= pad; }
    const lx = this.logX && !d.xCats;
    const X = (v) => L + (lx ? (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) : (v - x0) / (x1 - x0)) * pw;
    const Y = (v) => T + ph - (logY ? (Math.log(Math.max(v, y0)) - Math.log(y0)) / (Math.log(y1) - Math.log(y0)) : (v - y0) / (y1 - y0)) * ph;
    this.map = { X, Y };
    c.font = `11px ${mono}`; c.lineWidth = 1;
    c.textAlign = 'right'; c.textBaseline = 'middle';
    if (d.yCats) d.yCats.forEach((lab, i) => { c.fillStyle = muted; c.font = `11px ${ui}`; c.fillText(lab, L - 7, Y(i)); });
    else for (const t of logY ? logTicks(y0, y1) : niceTicks(y0, y1)) { const yy = Math.round(Y(t)) + 0.5; c.strokeStyle = rule; c.beginPath(); c.moveTo(L - 4, yy); c.lineTo(L, yy); c.stroke(); c.fillStyle = muted; c.fillText(tickLabel(t), L - 7, yy); }
    c.textAlign = 'center'; c.textBaseline = 'top';
    if (d.xCats) { const step = Math.max(1, Math.ceil(d.xCats.length / (pw / 30))); for (let i = 0; i < d.xCats.length; i += step) { c.fillStyle = muted; c.fillText(d.xCats[i], X(i), T + ph + 6); } }
    else for (const t of lx ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(3, Math.floor(pw / 80)))) { const xx = Math.round(X(t)) + 0.5; c.strokeStyle = rule; c.beginPath(); c.moveTo(xx, T + ph); c.lineTo(xx, T + ph + 4); c.stroke(); c.fillStyle = muted; c.fillText(tickLabel(t), xx, T + ph + 7); }
    c.strokeStyle = ink; c.lineWidth = 1; c.beginPath(); c.moveTo(L + 0.5, T); c.lineTo(L + 0.5, T + ph + 0.5); c.lineTo(L + pw, T + ph + 0.5); c.stroke();
    c.fillStyle = muted; c.font = `12px ${ui}`; c.fillText(d.xLabel || this.o.xLabel, L + pw / 2, T + ph + 24);
    c.save(); c.translate(13, T + ph / 2); c.rotate(-Math.PI / 2); c.textBaseline = 'middle'; c.fillText(d.yLabel || this.o.yLabel, 0, 0); c.restore();
    c.save(); c.beginPath(); c.rect(L, T - 2, pw + 2, ph + 4); c.clip();
    for (const l of d.hlines || []) { if (logY && l.y <= 0) continue; c.strokeStyle = tok(l.tok || '--rd-muted'); c.setLineDash(l.dash ? [5, 4] : []); c.beginPath(); c.moveTo(L, Y(l.y)); c.lineTo(L + pw, Y(l.y)); c.stroke(); if (l.label) { c.setLineDash([]); c.fillStyle = muted; c.font = `11px ${ui}`; c.textAlign = 'left'; c.textBaseline = 'bottom'; c.fillText(l.label, L + 6, Y(l.y) - 2); } }
    for (const l of d.vlines || []) { c.strokeStyle = tok(l.tok || '--rd-muted'); c.setLineDash([4, 4]); c.beginPath(); c.moveTo(X(l.x), T); c.lineTo(X(l.x), T + ph); c.stroke(); }
    c.setLineDash([]);
    const pts = [];
    for (const s of vis) {
      const col = tok(s.tok); c.strokeStyle = col; c.fillStyle = col; c.lineWidth = s.width || 1.6; c.setLineDash(s.dash ? [5, 4] : []);
      if (!s.noLine) { c.beginPath(); let pen = false; for (const p of s.pts) { if (!isFinite(p[1]) || (logY && p[1] <= 0)) { pen = false; continue; } const px = X(p[0]), py = Y(p[1]); if (pen) c.lineTo(px, py); else { c.moveTo(px, py); pen = true; } } c.stroke(); }
      c.setLineDash([]); c.lineWidth = 1;
      for (const p of s.pts) {
        if (!isFinite(p[1]) || (logY && p[1] <= 0)) continue;
        const px = X(p[0]), py = Y(p[1]);
        if (p[2] > 0 && s.xerr) { const a = X(Math.max(p[0] - 2 * p[2], this.logX ? x0 : -Infinity)), b = X(p[0] + 2 * p[2]); c.beginPath(); c.moveTo(a, py); c.lineTo(b, py); c.moveTo(a, py - 3); c.lineTo(a, py + 3); c.moveTo(b, py - 3); c.lineTo(b, py + 3); c.stroke(); }
        else if (p[2] > 0) { const a = Y(p[1] + 2 * p[2]), b = Y(Math.max(p[1] - 2 * p[2], logY ? y0 : -Infinity)); c.beginPath(); c.moveTo(px, a); c.lineTo(px, b); c.moveTo(px - 3, a); c.lineTo(px + 3, a); c.moveTo(px - 3, b); c.lineTo(px + 3, b); c.stroke(); }
        if (s.marker) { c.beginPath(); c.arc(px, py, s.marker, 0, 2 * Math.PI); c.fill(); }
        pts.push({ px, py, s, p });
      }
    }
    for (const m of d.marks || []) { c.fillStyle = tok(m.tok); c.strokeStyle = tok('--rd-surface'); c.lineWidth = 2; c.beginPath(); c.arc(X(m.x), Y(m.y), 5, 0, 2 * Math.PI); c.fill(); c.stroke(); pts.push({ px: X(m.x), py: Y(m.y), s: { label: m.label || 'marker', tok: m.tok }, p: [m.x, m.y] }); }
    c.restore();
    this.hoverReadout(c, pts, d, L, T, pw, ph);
  }

  hoverReadout(c, pts, d, L, T, pw, ph) {
    if (!this.hover) { this.tip.hidden = true; return; }
    const [hx, hy] = this.hover; let best = null, bd = 28 * 28;
    for (const q of pts) { const dd = (q.px - hx) ** 2 + (q.py - hy) ** 2; if (dd < bd) { bd = dd; best = q; } }
    if (!best) { this.tip.hidden = true; return; }
    c.strokeStyle = tok('--rd-ink'); c.lineWidth = 1; c.beginPath(); c.arc(best.px, best.py, 6, 0, 2 * Math.PI); c.stroke();
    const xs = d.xCats ? d.xCats[best.p[0]] : fmt(best.p[0], 4), ys = d.yCats ? d.yCats[best.p[1]] : fmt(best.p[1], 4);
    const pm = best.p[2] > 0 ? ` ± ${fmt(2 * best.p[2], 3)} (2 SE)` : '';
    this.tip.innerHTML = `<b style="color:var(${best.s.tok})">${best.s.label}</b><br>${d.xName || 'x'} = ${xs}${best.s.xerr ? pm : ''}<br>${d.yName || 'y'} = ${ys}${best.s.xerr ? '' : pm}${best.p[3] ? '<br>' + best.p[3] : ''}`;
    this.tip.hidden = false;
    const tw = this.tip.offsetWidth, left = best.px + 12 + tw > L + pw ? best.px - tw - 12 : best.px + 12;
    this.tip.style.left = Math.max(0, left) + 'px'; this.tip.style.top = Math.max(0, best.py - 20) + 'px';
  }

  drawHeat(c, w, h) {
    const d = this.data, muted = tok('--rd-muted'), mono = tok('--rd-font-mono'), ui = tok('--rd-font-ui');
    const L = 58, R = 64, T = 12, B = 44, pw = w - L - R, ph = h - T - B, nx = d.xs.length, ny = d.ys.length;
    let zmin = Infinity, zmax = -Infinity; for (const row of d.z) for (const v of row) { if (isFinite(v)) { zmin = Math.min(zmin, v); zmax = Math.max(zmax, v); } }
    const tz = (v) => (d.logZ ? (Math.log(v) - Math.log(zmin)) / (Math.log(zmax) - Math.log(zmin) || 1) : (v - zmin) / (zmax - zmin || 1));
    const cw = pw / nx, ch = ph / ny;
    for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) { c.fillStyle = sequential(tz(d.z[iy][ix])); c.fillRect(L + ix * cw, T + ph - (iy + 1) * ch, Math.ceil(cw), Math.ceil(ch)); }
    c.font = `11px ${mono}`; c.fillStyle = muted; c.textAlign = 'center'; c.textBaseline = 'top';
    const xstep = Math.max(1, Math.ceil(nx / 6)), ystep = Math.max(1, Math.ceil(ny / 6));
    for (let ix = 0; ix < nx; ix += xstep) c.fillText(tickLabel(+d.xs[ix].toPrecision(2)), L + (ix + 0.5) * cw, T + ph + 6);
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (let iy = 0; iy < ny; iy += ystep) c.fillText(tickLabel(+d.ys[iy].toPrecision(2)), L - 6, T + ph - (iy + 0.5) * ch);
    for (let i = 0; i <= 20; i++) { c.fillStyle = sequential(i / 20); c.fillRect(L + pw + 14, T + ph - (i + 1) * ph / 21, 12, Math.ceil(ph / 21)); }
    c.fillStyle = muted; c.textAlign = 'left'; c.fillText(fmt(zmax, 2), L + pw + 30, T + 6); c.fillText(fmt(zmin, 2), L + pw + 30, T + ph - 6);
    c.font = `12px ${ui}`; c.textAlign = 'center'; c.textBaseline = 'top'; c.fillText(d.xLabel || this.o.xLabel, L + pw / 2, T + ph + 24);
    c.save(); c.translate(13, T + ph / 2); c.rotate(-Math.PI / 2); c.textBaseline = 'middle'; c.fillText(d.yLabel || this.o.yLabel, 0, 0); c.restore();
    if (this.hover) {
      const ix = Math.floor((this.hover[0] - L) / cw), iy = Math.floor((T + ph - this.hover[1]) / ch);
      if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
        c.strokeStyle = tok('--rd-ink'); c.strokeRect(L + ix * cw + 0.5, T + ph - (iy + 1) * ch + 0.5, cw, ch);
        this.tip.innerHTML = `${d.xName} = ${fmt(d.xs[ix], 3)}<br>${d.yName} = ${fmt(d.ys[iy], 3)}<br><b>${d.zLabel} = ${fmt(d.z[iy][ix], 3)}</b>`;
        this.tip.hidden = false; this.tip.style.left = Math.min(this.hover[0] + 12, w - 160) + 'px'; this.tip.style.top = Math.max(0, this.hover[1] - 30) + 'px';
      } else this.tip.hidden = true;
    } else this.tip.hidden = true;
  }
}

/**
 * Convert method summaries into chart series.
 * @param {string[]} ids method ids in drawing order
 * @param {Object} METHOD method registry (engine.METHOD)
 * @param {(id:string)=>number[][]} ptsFn points for a method
 * @param {Object} [style] extra style per series
 * @returns {Array} series
 */
export function methodSeries(ids, METHOD, ptsFn, style = {}) {
  return ids.map((id) => ({ id, label: METHOD[id].label, tok: METHOD[id].tok, pts: ptsFn(id), marker: 2.6, width: id === 'ghb' ? 2.2 : 1.4, ...style }));
}

export { alpha };
