// tests/index.js
// Entry point so that `node --test tests/` works on every Node version: recent Node resolves a directory argument to
// this file, which loads every *.test.mjs here (node:test collects their tests in this process).
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

(async () => {
  for (const f of fs.readdirSync(__dirname).filter((x) => x.endsWith('.test.mjs')).sort()) await import(pathToFileURL(path.join(__dirname, f)).href);
})();
