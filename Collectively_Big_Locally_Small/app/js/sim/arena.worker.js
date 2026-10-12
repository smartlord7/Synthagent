/**
 * sim/arena.worker.js
 * Responsibility: run the Literature arena (sim/arena.js) in a module Web Worker so the page never freezes.
 * Receives {spec}; posts {type:'progress', phase, done, total} after every replicate, then {type:'done', result},
 * or {type:'error', message}. Cancelling is done by the page terminating the worker.
 */
import { runArena } from './arena.js';

/**
 * Message handler: run one arena specification to completion.
 * @param {MessageEvent<{spec:Object}>} e message
 * @see Section 6.3 (Simulations 8 and 15)
 */
self.onmessage = (e) => {
  try {
    const g = runArena(e.data.spec);
    let r = g.next();
    while (!r.done) { self.postMessage({ type: 'progress', ...r.value }); r = g.next(); }
    self.postMessage({ type: 'done', result: r.value });
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
