/**
 * math/graph.js
 * Responsibility: collaboration graphs and gossip linear algebra (Section 4.3, Proposition 4.3):
 * topologies, Metropolis-Hastings weights, the second largest eigenvalue modulus, matrix powers
 * and per-round link failures (Simulation 13). Matrices are arrays of Float64Array rows.
 */
import { makeRng } from './random.js';

/**
 * Seeded K = 24 graphs of scripts/fig_gossip.py (numpy seeds 5 and 577), exported as edge lists so that
 * their spectral properties match Figure 2 exactly (lambda = 0.964 and 0.937).
 */
export const PAPER_EDGES = Object.freeze({
  regional: [[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5],[2,4],[2,5],[3,4],[3,20],[5,10],[6,8],[6,9],[6,10],[6,11],[7,8],[7,9],[7,11],[8,9],[8,11],[9,14],[10,11],[12,13],[12,14],[12,15],[12,16],[12,17],[13,14],[13,16],[13,17],[14,15],[14,16],[14,17],[15,16],[15,17],[16,17],[17,19],[18,19],[18,20],[18,21],[18,22],[19,20],[19,22],[20,21],[20,23],[21,22],[21,23],[22,23]],
  random: [[0,1],[0,3],[0,14],[0,17],[0,18],[1,3],[1,4],[1,16],[2,8],[2,12],[2,18],[2,21],[3,8],[4,5],[4,9],[4,15],[4,21],[4,22],[5,6],[5,8],[5,18],[6,16],[7,10],[7,23],[8,16],[8,20],[9,20],[9,22],[10,23],[11,12],[11,20],[12,15],[12,18],[12,23],[13,22],[14,15],[17,19],[18,21],[18,23],[19,22],[20,21],[21,22]]
});

/** Topology identifiers and labels used across the app. */
export const TOPOLOGIES = Object.freeze([
  ['ring', 'Ring (degree 2)'], ['ring2', 'Ring + 2-hop (degree 4)'],
  ['regional', 'Regional clusters + bridges'], ['random', 'Sparse random']
]);

/** @returns {Float64Array[]} K x K zero matrix */
export function zeros(K) { return Array.from({ length: K }, () => new Float64Array(K)); }

/**
 * Symmetric adjacency matrix from an edge list.
 * @param {number} K nodes
 * @param {number[][]} edges pairs [i, j]
 * @returns {Float64Array[]} adjacency
 */
export function edgesToAdj(K, edges) {
  const A = zeros(K);
  for (const [i, j] of edges) { A[i][j] = 1; A[j][i] = 1; }
  return A;
}

/**
 * Upper-triangle edge list of an adjacency matrix.
 * @param {Float64Array[]} A adjacency
 * @returns {number[][]} edges [i, j] with i < j
 */
export function edgeList(A) {
  const E = [];
  for (let i = 0; i < A.length; i++) for (let j = i + 1; j < A.length; j++) if (A[i][j]) E.push([i, j]);
  return E;
}

/**
 * Build a collaboration graph (fig_gossip.ring / regional / erdos_renyi).
 * Seed 0 with K = 24 returns the paper's seeded graphs; other settings use this module's generator.
 * @param {'ring'|'ring2'|'regional'|'random'} kind topology
 * @param {number} K number of hospitals
 * @param {number} [p=0.15] edge probability of the sparse random graph
 * @param {number} [seed=0] graph seed
 * @param {number} [clusters=4] regions of the clustered graph
 * @param {number} [pIn=0.8] within-region edge probability
 * @returns {Float64Array[]} adjacency
 * @see Section 4.3, Figure 2
 */
export function topology(kind, K, p = 0.15, seed = 0, clusters = 4, pIn = 0.8) {
  const E = [];
  if (kind === 'ring' || kind === 'ring2') {
    const h = kind === 'ring' ? 1 : 2;
    for (let i = 0; i < K; i++) for (let s = 1; s <= h; s++) { const j = (i + s) % K; if (j !== i) E.push([i, j]); }
  } else if (kind === 'regional') {
    if (K === 24 && seed === 0 && clusters === 4 && pIn === 0.8) return edgesToAdj(K, PAPER_EDGES.regional);
    const r = makeRng(1000 + seed), size = Math.max(1, Math.floor(K / clusters));
    const grp = (i) => Math.min(clusters - 1, Math.floor(i / size));
    const members = (c) => { const m = []; for (let i = 0; i < K; i++) if (grp(i) === c) m.push(i); return m; };
    for (let c = 0; c < clusters; c++) {
      const m = members(c);
      for (let a = 0; a < m.length; a++) for (let b = a + 1; b < m.length; b++) if (r.u() < pIn) E.push([m[a], m[b]]);
    }
    for (let c = 0; c < clusters; c++) {
      const m = members(c), m2 = members((c + 1) % clusters);
      if (m.length && m2.length) E.push([m[Math.floor(r.u() * m.length)], m2[Math.floor(r.u() * m2.length)]]);
    }
  } else {
    if (K === 24 && seed === 0 && Math.abs(p - 0.15) < 1e-9) return edgesToAdj(K, PAPER_EDGES.random);
    const r = makeRng(2000 + seed + Math.round(p * 1000) * 31);
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) if (r.u() < p) E.push([i, j]);
  }
  return edgesToAdj(K, E.filter(([i, j]) => i !== j));
}

/**
 * Whether the graph is connected (required for lambda < 1).
 * @param {Float64Array[]} A adjacency
 * @returns {boolean}
 * @see Proposition 4.3 (ii)
 */
export function connected(A) {
  const K = A.length, seen = new Uint8Array(K), st = [0];
  seen[0] = 1; let c = 1;
  while (st.length) { const i = st.pop(); for (let j = 0; j < K; j++) if (A[i][j] && !seen[j]) { seen[j] = 1; c++; st.push(j); } }
  return c === K;
}

/**
 * Metropolis-Hastings gossip matrix W_kl = 1/(1 + max(d_k, d_l)) on edges, diagonal fills the row to 1.
 * @param {Float64Array[]} A adjacency
 * @returns {Float64Array[]} symmetric doubly stochastic W
 * @see Section 4.3
 */
export function metropolisHastings(A) {
  const K = A.length, deg = A.map((r) => r.reduce((s, x) => s + x, 0)), W = zeros(K);
  for (let k = 0; k < K; k++) {
    let s = 0;
    for (let l = 0; l < K; l++) if (A[k][l]) { W[k][l] = 1 / (1 + Math.max(deg[k], deg[l])); s += W[k][l]; }
    W[k][k] = 1 - s;
  }
  return W;
}

/**
 * Eigenvalues of a symmetric matrix by cyclic Jacobi rotations.
 * @param {Float64Array[]} M symmetric matrix
 * @returns {number[]} eigenvalues, ascending
 */
export function symEig(M) {
  const n = M.length, a = M.map((r) => Float64Array.from(r));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
    }
  }
  return a.map((r, i) => r[i]).sort((x, y) => x - y);
}

/**
 * Second largest eigenvalue modulus lambda = max(|lambda_2|, |lambda_K|) (blueprint.slem).
 * @param {Float64Array[]} W gossip matrix
 * @returns {number} lambda
 * @see Proposition 4.3 (ii)
 */
export function slem(W) { const ev = symEig(W); return Math.max(Math.abs(ev[ev.length - 2]), Math.abs(ev[0])); }

/**
 * Rounds needed for consensus accuracy gamma.
 * @param {number} lambda SLEM
 * @param {number} gamma target relative error
 * @returns {number} log(1/gamma) / (1 - lambda)
 * @see Section 4.3, after Proposition 4.3
 */
export function predictedRounds(lambda, gamma) { return Math.log(1 / gamma) / (1 - lambda); }

/** @returns {Float64Array[]} matrix product A B */
export function matMul(A, B) {
  const n = A.length, C = zeros(n);
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) {
    const a = A[i][k]; if (a === 0) continue;
    const Bk = B[k], Ci = C[i];
    for (let j = 0; j < n; j++) Ci[j] += a * Bk[j];
  }
  return C;
}

/**
 * T-th power of the gossip matrix: the averaging operator after T rounds.
 * @param {Float64Array[]} W gossip matrix
 * @param {number} T rounds (>= 1)
 * @returns {Float64Array[]} W^T
 * @see Proposition 4.3
 */
export function matPow(W, T) {
  let R = null, B = W;
  while (T > 0) { if (T & 1) R = R ? matMul(R, B) : B; T >>= 1; if (T) B = matMul(B, B); }
  return R;
}

/**
 * Matrix-vector product W x.
 * @param {Float64Array[]} W matrix
 * @param {ArrayLike<number>} x vector
 * @returns {Float64Array}
 */
export function matVec(W, x) {
  const K = W.length, o = new Float64Array(K);
  for (let k = 0; k < K; k++) { let s = 0; const Wk = W[k]; for (let l = 0; l < K; l++) s += Wk[l] * x[l]; o[k] = s; }
  return o;
}

/**
 * Remove each edge independently with probability p (one gossip round of Simulation 13).
 * Recomputing Metropolis-Hastings weights on the result keeps the round doubly stochastic.
 * @param {number[][]} edges edge list
 * @param {number} K nodes
 * @param {number} p failure probability
 * @param {() => number} u uniform generator
 * @returns {{A: Float64Array[], failed: Set<string>}} working adjacency and the failed edges as "i-j"
 * @see Simulation 13
 */
export function failLinks(edges, K, p, u) {
  const A = zeros(K), failed = new Set();
  for (const [i, j] of edges) { if (u() >= p) { A[i][j] = A[j][i] = 1; } else failed.add(i + '-' + j); }
  return { A, failed };
}
