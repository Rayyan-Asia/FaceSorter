/**
 * Launches Electron with ELECTRON_RUN_AS_NODE removed from the environment.
 * This variable, if set (e.g. by Claude Code or CI), causes Electron to skip
 * its own API setup, breaking require('electron') in the main process.
 */
const { spawn } = require('child_process');
const path = require('path');

const electronPath = require('electron');

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const args = [path.resolve(__dirname, '..'), ...process.argv.slice(2)];

const proc = spawn(electronPath, args, {
  stdio: 'inherit',
  env,
});

proc.on('close', (code) => process.exit(code ?? 0));
proc.on('error', (err) => {
  console.error('Failed to start Electron:', err.message);
  process.exit(1);
});
