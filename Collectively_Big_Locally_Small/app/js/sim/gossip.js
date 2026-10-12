/**
 * sim/gossip.js
 * Responsibility: a step-by-step gossip process for animation and the consensus-error curve
 * (Proposition 4.3). Each round applies x <- W_t x, where W_t are Metropolis-Hastings weights recomputed
 * on the links that work in that round (Simulation 13).
 */
import { makeRng } from '../math/random.js';
import { metropolisHastings, failLinks, matVec } from '../math/graph.js';

/**
 * Create a gossip process.
 * @param {{K:number, edges:number[][], W:Float64Array[]}} net graph (from engine.prepare)
 * @param {ArrayLike<number>} x0 initial values (the privatized releases)
 * @param {number} pFail link failure probability per round
 * @param {number} [seed=99] seed of the failure pattern
 * @returns {{x:Float64Array, x0:Float64Array, xbar:number, t:number, errors:number[][], failed:Set<string>, step:()=>{prev:Float64Array, next:Float64Array, failed:Set<string>, W:Float64Array[]}}}
 * @see Proposition 4.3 (ii): ||x^t - xbar 1|| <= lambda^t ||x^0 - xbar 1||
 */
export function createGossip(net, x0, pFail, seed = 99) {
  const K = net.K, rng = makeRng(seed);
  const xbar = Array.from(x0).reduce((a, b) => a + b, 0) / K;
  const norm = (x) => Math.sqrt(Array.from(x).reduce((s, v) => s + (v - xbar) ** 2, 0));
  const e0 = norm(x0) || 1;
  const g = { x: Float64Array.from(x0), x0: Float64Array.from(x0), xbar, t: 0, errors: [[0, 1]], failed: new Set() };
  g.step = () => {
    let W = net.W, failed = new Set();
    if (pFail > 0) { const f = failLinks(net.edges, K, pFail, rng.u); W = metropolisHastings(f.A); failed = f.failed; }
    const prev = g.x, next = matVec(W, prev);
    g.x = next; g.t++; g.failed = failed;
    g.errors.push([g.t, Math.max(norm(next) / e0, 1e-16)]);
    return { prev, next, failed, W };
  };
  return g;
}

/**
 * Leave-one-out estimate held by hospital k after t rounds of plain averaging.
 * @param {ArrayLike<number>} xt values after t rounds
 * @param {ArrayLike<number>} x0 initial releases
 * @param {number} k hospital
 * @returns {number} (K x_k^t - x_k^0) / (K - 1)
 * @see Proposition 4.3 (iii)
 */
export function looFromGossip(xt, x0, k) { const K = xt.length; return (K * xt[k] - x0[k]) / (K - 1); }
