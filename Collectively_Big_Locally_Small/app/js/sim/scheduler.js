/**
 * sim/scheduler.js
 * Responsibility: run long Monte Carlo jobs in small chunks inside requestAnimationFrame so the page stays
 * responsive and charts can fill in progressively. Jobs are generators; each `yield` ends a chunk.
 * Starting a job under a name that is already running cancels the old one.
 */

const jobs = new Map();
let scheduled = false;
const BUDGET_MS = 12;

function frame() {
  scheduled = false;
  const t0 = performance.now();
  while (jobs.size && performance.now() - t0 < BUDGET_MS) {
    for (const [name, job] of jobs) {
      let r;
      try { r = job.gen.next(); } catch (e) { jobs.delete(name); console.error(`job ${name} failed`, e); job.onError?.(e); continue; }
      if (r.done) { jobs.delete(name); job.onDone?.(r.value); }
      if (performance.now() - t0 >= BUDGET_MS) break;
    }
  }
  if (jobs.size) schedule();
}
function schedule() {
  if (scheduled) return;
  scheduled = true;
  if (typeof document !== 'undefined' && document.hidden) setTimeout(frame, 30); else requestAnimationFrame(frame);
}

/**
 * Start (or restart) a chunked job.
 * @param {string} name job name; an existing job with this name is cancelled
 * @param {Generator} gen generator doing a chunk of work per `next()`; its return value is passed to onDone
 * @param {{onDone?:(v:any)=>void, onError?:(e:Error)=>void}} [cb] callbacks
 */
export function runJob(name, gen, cb = {}) { jobs.set(name, { gen, ...cb }); schedule(); }

/**
 * Cancel a job by name (no callback fires).
 * @param {string} name job name
 */
export function cancelJob(name) { jobs.delete(name); }

/**
 * Whether a job is running.
 * @param {string} name job name
 * @returns {boolean}
 */
export function isRunning(name) { return jobs.has(name); }
