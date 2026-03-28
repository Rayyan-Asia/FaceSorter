const { execSync } = require('child_process');
const path = require('path');

const scriptsDir = __dirname;
const cmd = process.platform === 'win32'
  ? `"${path.join(scriptsDir, 'setup.bat')}"`
  : `bash "${path.join(scriptsDir, 'setup.sh')}"`;

execSync(cmd, { stdio: 'inherit', cwd: scriptsDir });
