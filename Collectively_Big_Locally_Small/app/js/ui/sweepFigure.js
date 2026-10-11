/**
 * ui/sweepFigure.js
 * Responsibility: run a sweep (sim/sweep.js) into a Figure with progress, so that the Sweeps tab and the
 * canned figures of other tabs share one code path.
 */
import { runSweep, grid, METRIC } from '../sim/sweep.js';
import { runJob } from '../sim/scheduler.js';
import { METHOD } from '../sim/engine.js';
import { PARAM } from '../config.js';

/**
 * Convert a sweep result into chart series (one per metric and method).
 * @param {Object} res sweep result
 * @returns {Array} series
 */
export function sweepSeries(res) {
  const multi = new Set(res.series.map((s) => s.metric)).size > 1;
  return res.series.map((s, i) => {
    const m = s.method ? METHOD[s.method] : null, met = METRIC[s.metric];
    return {
      id: s.metric + '-' + (s.method || i), label: (m ? m.label : met.label) + (multi && m ? ` (${met.label.toLowerCase()})` : ''),
      tok: m ? m.tok : ['--rd-ghb', '--rd-ref', '--rd-noise', '--rd-robust'][i % 4], dash: multi && s.metric !== res.metrics?.[0] && !!m,
      pts: s.pts, marker: 3, width: s.method === 'ghb' ? 2.2 : 1.5
    };
  });
}

/**
 * Start a sweep into a figure.
 * @param {string} name job name (restarting cancels the previous run under that name)
 * @param {import('../viz/chart.js').Figure} fig target figure
 * @param {{param:string, start:number, stop:number, steps:number, log:boolean, metrics:string[], methods:string[], R:number, title?:string}} spec sweep
 * @param {Object} cfg base configuration
 * @param {HTMLElement} [progEl] progress element
 * @param {Object} [extra] extra figure data (hlines etc.)
 */
export function startSweep(name, fig, spec, cfg, progEl, extra = {}) {
  const p = PARAM[spec.param], values = grid(spec.start, spec.stop, spec.steps, spec.log, !!p?.int);
  const metricUnits = [...new Set(spec.metrics.map((m) => METRIC[m].unit))].join(', ');
  const xLabel = `${p ? p.label : spec.param}${p?.unit ? ` (${p.unit})` : ''}`;
  fig.logX = spec.log; fig.logY = !!spec.logY;
  const lx = fig.el.querySelector(`#${fig.o.id}-logx`); if (lx) lx.checked = spec.log;
  const ly = fig.el.querySelector(`#${fig.o.id}-logy`); if (ly) ly.checked = !!spec.logY;
  const yLabel = spec.metrics.length === 1 ? `${METRIC[spec.metrics[0]].label} (${metricUnits})` : metricUnits;
  const ref = spec.metrics.some((m) => ['mean', 'worst', 'honest'].includes(m)) ? [{ y: 1, tok: '--rd-local', dash: true, label: 'local = 1' }] : [];
  fig.setData({ series: [], xLabel, yLabel, empty: 'Running the sweep…' });
  if (progEl) progEl.textContent = `Sweeping ${values.length} values…`;
  runJob(name, runSweep(cfg, { ...spec, values }, (res, done) => {
    res.metrics = spec.metrics;
    fig.setData({ series: sweepSeries(res), xLabel, yLabel, xName: p ? p.label : spec.param, yName: METRIC[spec.metrics[0]].label, hlines: [...ref, ...(extra.hlines || [])], vlines: extra.vlines });
    if (progEl) progEl.textContent = done ? `${values.length} values${spec.R && spec.metrics.some((m) => METRIC[m].kind === 'mc') ? `, ${spec.R} replicates each` : ', closed form'}.` : `Sweeping… ${res.series[0]?.pts.length || 0} / ${values.length} values`;
  }));
}
