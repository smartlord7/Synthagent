/**
 * viz/network.js
 * Responsibility: the animated hospital network. Nodes sit on a circle (regions stay contiguous), their
 * area proportional to hospital size and their fill set by the value they hold. It animates the three
 * phases of Algorithm 1: noise being added to each release (amber halo sized by s_k), gossip rounds
 * (particles travelling along working edges while values mix) and shrinkage (values easing from the local
 * estimate to the GHB estimate). Nodes are selectable by click or keyboard; in edit mode two clicks
 * add or remove an edge.
 */
import { tok, setupCanvas, mmGrid, diverging, alpha, onThemeChange, reducedMotion, fmt } from './theme.js';

const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

export class NetworkView {
  /**
   * @param {HTMLCanvasElement} canvas target canvas (data-h sets its height)
   * @param {{onSelect?:(k:number)=>void, onEdgeToggle?:(i:number, j:number)=>void}} [cb] callbacks
   */
  constructor(canvas, cb = {}) {
    this.cv = canvas; this.cb = cb; this.K = 0; this.edges = []; this.sizes = []; this.values = null; this.selected = -1;
    this.failed = new Set(); this.particles = []; this.halos = null; this.edit = false; this.pending = -1; this.bad = new Set();
    this.centre = 0; this.extent = 1; this.anim = null; this.label = '';
    canvas.addEventListener('click', (e) => this.onClick(e));
    canvas.addEventListener('keydown', (e) => this.onKey(e));
    new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
    onThemeChange(() => this.draw());
  }

  /**
   * Set the graph.
   * @param {{K:number, edges:number[][], sizes:number[], bad?:number[]}} g graph and hospital sizes
   */
  setGraph(g) {
    this.K = g.K; this.edges = g.edges; this.sizes = g.sizes; this.bad = new Set(g.bad || []);
    if (this.selected >= this.K) this.selected = -1;
    this.draw();
  }

  /**
   * Values held by the hospitals and the colour scale.
   * @param {ArrayLike<number>} v values
   * @param {number} [centre] value drawn neutral
   * @param {number} [extent] value distance drawn at full colour
   */
  setValues(v, centre, extent) {
    this.values = Float64Array.from(v);
    if (centre !== undefined) this.centre = centre;
    if (extent !== undefined) this.extent = extent || 1;
    this.draw();
  }

  /** Select a hospital (or -1). */
  select(k) { this.selected = k; this.draw(); }

  pos(i, w, h) {
    const cx = w / 2, cy = h / 2, rad = Math.min(w, h) / 2 - 34, a = 2 * Math.PI * i / this.K - Math.PI / 2;
    return [cx + rad * Math.cos(a), cy + rad * Math.sin(a)];
  }
  radius(i, w, h) {
    const maxN = Math.max(...this.sizes), rad = Math.min(w, h) / 2 - 34, room = Math.min(24, rad * Math.PI / this.K * 0.95);
    return Math.max(3.5, Math.sqrt(this.sizes[i] / maxN) * room);
  }

  /**
   * Run an animation for `ms` milliseconds; frame(t) receives eased progress in [0, 1].
   * Reduced motion jumps to the end state.
   * @returns {Promise<void>}
   */
  animate(ms, frame) {
    if (this.anim) { this.anim.cancel(); }
    return new Promise((resolve) => {
      if (reducedMotion || ms <= 0) { frame(1); this.draw(); resolve(); return; }
      let stop = false; const t0 = performance.now();
      this.anim = { cancel: () => { stop = true; frame(1); this.draw(); resolve(); } };
      const step = (now) => {
        if (stop) return;
        const t = Math.min(1, (now - t0) / ms); frame(ease(t)); this.draw();
        if (t < 1) requestAnimationFrame(step); else { this.anim = null; resolve(); }
      };
      requestAnimationFrame(step);
    });
  }

  /**
   * Animate privatization: each node's value moves from its local estimate to its release while an amber halo
   * proportional to the noise sd s_k pulses.
   * @param {ArrayLike<number>} from local estimates
   * @param {ArrayLike<number>} to releases
   * @param {ArrayLike<number>} s noise sd per hospital
   * @param {number} ms duration
   * @see Proposition 8.2
   */
  animateNoise(from, to, s, ms) {
    const smax = Math.max(...s) || 1;
    return this.animate(ms, (t) => {
      const wob = Math.sin(t * Math.PI * 3) * (1 - t);
      this.values = Float64Array.from(from, (f, k) => f + (to[k] - f) * t + wob * s[k] * 0.6);
      this.halos = Array.from(s, (x) => ({ r: x / smax, a: Math.sin(t * Math.PI) }));
      if (t >= 1) this.halos = null;
    });
  }

  /**
   * Animate one gossip round: a particle per direction of every working edge, values mixing.
   * @param {ArrayLike<number>} prev values before the round
   * @param {ArrayLike<number>} next values after the round
   * @param {Set<string>} failed failed edges ("i-j")
   * @param {number} ms duration
   * @see Proposition 8.4
   */
  animateGossip(prev, next, failed, ms) {
    this.failed = failed;
    const parts = [];
    for (const [i, j] of this.edges) if (!failed.has(i + '-' + j)) { parts.push({ a: i, b: j, v: prev[i] }, { a: j, b: i, v: prev[j] }); }
    return this.animate(ms, (t) => {
      this.particles = t < 1 ? parts.map((p) => ({ ...p, t })) : [];
      this.values = Float64Array.from(prev, (p, k) => p + (next[k] - p) * t);
    });
  }

  /**
   * Animate shrinkage from local estimates toward the GHB estimates.
   * @param {ArrayLike<number>} from local estimates
   * @param {ArrayLike<number>} to GHB estimates
   * @param {number} ms duration
   * @param {(t:number)=>void} [onFrame] progress callback (drives the shrinkage slider)
   * @see Proposition 8.1
   */
  animateShrink(from, to, ms, onFrame) {
    return this.animate(ms, (t) => { this.values = Float64Array.from(from, (f, k) => f + (to[k] - f) * t); onFrame?.(t); });
  }

  hit(e) {
    const r = this.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, w = r.width, h = r.height;
    for (let i = 0; i < this.K; i++) { const [px, py] = this.pos(i, w, h); if ((px - x) ** 2 + (py - y) ** 2 <= (this.radius(i, w, h) + 6) ** 2) return i; }
    return -1;
  }
  onClick(e) { const k = this.hit(e); if (k < 0) return; this.choose(k); }
  choose(k) {
    if (this.edit) {
      if (this.pending < 0) { this.pending = k; this.draw(); return; }
      if (this.pending !== k) this.cb.onEdgeToggle?.(Math.min(k, this.pending), Math.max(k, this.pending));
      this.pending = -1; this.draw(); return;
    }
    this.selected = k; this.draw(); this.cb.onSelect?.(k);
  }
  onKey(e) {
    if (!this.K) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); this.focusK = ((this.focusK ?? this.selected) + 1 + this.K) % this.K; this.draw(); }
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); this.focusK = ((this.focusK ?? (this.selected < 0 ? 0 : this.selected)) - 1 + this.K) % this.K; this.draw(); }
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.choose(this.focusK ?? 0); }
  }

  draw() {
    const S = setupCanvas(this.cv); if (!S || !this.K) return;
    const { c, w, h } = S, ink = tok('--rd-ink'), rule = tok('--rd-rule'), muted = tok('--rd-muted');
    mmGrid(c, 0, 0, w, h);
    for (const [i, j] of this.edges) {
      const f = this.failed.has(i + '-' + j), [ax, ay] = this.pos(i, w, h), [bx, by] = this.pos(j, w, h);
      c.strokeStyle = f ? alpha(rule, 0.5) : rule; c.lineWidth = f ? 1 : 1.3; c.setLineDash(f ? [3, 4] : []);
      c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
    }
    c.setLineDash([]);
    for (const p of this.particles) {
      const [ax, ay] = this.pos(p.a, w, h), [bx, by] = this.pos(p.b, w, h);
      c.fillStyle = diverging((p.v - this.centre) / this.extent); c.strokeStyle = ink; c.lineWidth = 0.6;
      c.beginPath(); c.arc(ax + (bx - ax) * p.t, ay + (by - ay) * p.t, 2.6, 0, 2 * Math.PI); c.fill(); c.stroke();
    }
    const noise = tok('--rd-noise'), risk = tok('--rd-risk');
    for (let i = 0; i < this.K; i++) {
      const [x, y] = this.pos(i, w, h), r = this.radius(i, w, h);
      if (this.halos) { const hl = this.halos[i]; c.strokeStyle = alpha(noise, 0.85 * hl.a); c.lineWidth = 2; c.beginPath(); c.arc(x, y, r + 3 + 16 * hl.r * hl.a, 0, 2 * Math.PI); c.stroke(); }
      const v = this.values ? this.values[i] : this.centre;
      c.fillStyle = diverging((v - this.centre) / this.extent);
      c.strokeStyle = this.bad.has(i) ? risk : ink; c.lineWidth = this.bad.has(i) ? 2 : 0.8;
      c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fill(); c.stroke();
      if (i === this.selected || i === this.pending) { c.strokeStyle = i === this.pending ? noise : ink; c.lineWidth = 2; c.beginPath(); c.arc(x, y, r + 4, 0, 2 * Math.PI); c.stroke(); }
      if (i === this.focusK && document.activeElement === this.cv) { c.strokeStyle = tok('--rd-focus'); c.setLineDash([2, 2]); c.lineWidth = 1.5; c.beginPath(); c.arc(x, y, r + 7, 0, 2 * Math.PI); c.stroke(); c.setLineDash([]); }
    }
    c.font = `10px ${tok('--rd-font-mono')}`; c.fillStyle = muted; c.textAlign = 'center'; c.textBaseline = 'middle';
    if (this.K <= 30) for (let i = 0; i < this.K; i++) { const [x, y] = this.pos(i, w, h), cx = w / 2, cy = h / 2, dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1, off = this.radius(i, w, h) + 11; c.fillText(String(this.sizes[i]), x + dx / d * off, y + dy / d * off); }
    c.fillStyle = ink; c.font = `11px ${tok('--rd-font-mono')}`; c.fillText(this.label, w / 2, h / 2);
    if (this.selected >= 0 && this.values) { c.fillStyle = muted; c.fillText(`hospital ${this.selected + 1}: ${fmt(this.values[this.selected], 4)}`, w / 2, h / 2 + 16); }
  }
}
