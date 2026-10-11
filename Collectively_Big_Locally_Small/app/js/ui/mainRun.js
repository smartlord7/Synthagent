/**
 * ui/mainRun.js
 * Responsibility: keep one Monte Carlo run of the current configuration up to date. When a parameter that
 * affects the simulation changes, the run restarts (debounced); every consumer (metrics, figures, inspector)
 * subscribes to its progressive results, so all tabs show the same replicates.
 */
import * as store from './store.js';
import { PARAM } from '../config.js';
import { runMonteCarlo } from '../sim/engine.js';
import { runJob } from '../sim/scheduler.js';

const subs = new Set();
let last = null, timer = null, overrides = {};

/**
 * Subscribe to progressive results.
 * @param {(r:{ctx:Object, summary:Object, done:boolean, total:number})=>void} fn callback
 */
export function onResult(fn) { subs.add(fn); if (last) fn(last); }

/** @returns {Object|null} the latest result */
export function latest() { return last; }

/** Restart the run now with the current configuration. */
export function restart() {
  clearTimeout(timer);
  const c = { ...store.get(), ...overrides }, total = c.R;
  runJob('main', runMonteCarlo(c, { onProgress: (summary, done, ctx) => { last = { ctx, summary, done, total }; subs.forEach((f) => f(last)); } }));
}

store.subscribe((keys) => {
  if (keys.every((k) => PARAM[k]?.group === 'calc' || k === 'refN')) return;
  clearTimeout(timer); timer = setTimeout(restart, 280);
});
