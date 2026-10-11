/**
 * ui/store.js
 * Responsibility: hold the live configuration and notify subscribers of changed keys. Every control,
 * preset, import and sweep reads and writes the configuration through this store only.
 */
import { defaultConfig, normalizeConfig } from '../config.js';

const subs = new Set();
let config = defaultConfig();

/** @returns {Object} the live configuration (treat as read-only; write through set/replace) */
export function get() { return config; }

/**
 * Set one key and notify subscribers.
 * @param {string} key configuration key
 * @param {*} value new value
 */
export function set(key, value) {
  if (config[key] === value) return;
  config = { ...config, [key]: value };
  subs.forEach((f) => f([key], config));
}

/**
 * Replace the configuration (presets, import, randomize); values are normalized first.
 * @param {Object} next configuration
 * @returns {string[]} normalization warnings
 */
export function replace(next) {
  const { config: c, warnings } = normalizeConfig(next);
  config = c;
  subs.forEach((f) => f(Object.keys(c), config));
  return warnings;
}

/**
 * Subscribe to changes.
 * @param {(keys:string[], config:Object)=>void} fn called with the changed keys
 * @returns {()=>void} unsubscribe
 */
export function subscribe(fn) { subs.add(fn); return () => subs.delete(fn); }
