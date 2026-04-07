const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const os = require('os');

const isDev = process.env.NODE_ENV === 'development';

function getPythonExecutable() {
  const venvPython = process.platform === 'win32'
    ? path.join(__dirname, '../../scripts', 'venv', 'Scripts', 'python.exe')
    : path.join(__dirname, '../../scripts', 'venv', 'bin', 'python3');

  if (isDev) {
    const fs = require('fs');
    if (fs.existsSync(venvPython)) return venvPython;
    return process.platform === 'win32' ? 'python' : 'python3';
  }
  if (process.platform === 'win32') {
    return path.join(process.resourcesPath, 'scripts', 'venv', 'Scripts', 'python.exe');
  }
  return path.join(process.resourcesPath, 'scripts', 'venv', 'bin', 'python3');
}

function getScriptPath(scriptName) {
  if (isDev) {
    return path.join(__dirname, '../../scripts', scriptName);
  }
  return path.join(process.resourcesPath, 'scripts', scriptName);
}

function getLocalIpAddress() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return '127.0.0.1';
}

const PHOTO_SERVER_PORT = 4567;
const ALLOWED_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']);

const photoServer = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET');

  const reqUrl = new URL(req.url, `http://localhost:${PHOTO_SERVER_PORT}`);
  if (reqUrl.pathname !== '/photo') {
    res.writeHead(404);
    res.end();
    return;
  }

  const filePath = reqUrl.searchParams.get('path');
  if (!filePath) {
    res.writeHead(400);
    res.end();
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  if (!ALLOWED_EXTENSIONS.has(ext)) {
    res.writeHead(403);
    res.end();
    return;
  }

  const fs = require('fs');
  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      res.writeHead(404);
      res.end();
      return;
    }

    const mimeTypes = {
      '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
      '.png': 'image/png', '.bmp': 'image/bmp',
      '.tiff': 'image/tiff', '.tif': 'image/tiff',
      '.webp': 'image/webp',
    };

    const widthParam = reqUrl.searchParams.get('w');
    const thumbWidth = widthParam ? parseInt(widthParam, 10) : null;

    if (thumbWidth && thumbWidth > 0) {
      // Serve a resized thumbnail using sharp
      try {
        const sharp = require('sharp');
        res.writeHead(200, { 'Content-Type': 'image/jpeg', 'Cache-Control': 'max-age=86400' });
        sharp(filePath).resize({ width: thumbWidth, withoutEnlargement: true }).jpeg({ quality: 80 }).pipe(res);
      } catch (e) {
        res.writeHead(500);
        res.end();
      }
    } else {
      res.writeHead(200, {
        'Content-Type': mimeTypes[ext] || 'application/octet-stream',
        'Cache-Control': 'max-age=3600',
      });
      fs.createReadStream(filePath).pipe(res);
    }
  });
});

photoServer.listen(PHOTO_SERVER_PORT, () => {
  console.log(`Photo file server listening on port ${PHOTO_SERVER_PORT}`);
});

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    titleBarStyle: 'hiddenInset',
    show: false,
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../../dist/index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC: Open folder picker dialog
ipcMain.handle('dialog:openDirectory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });
  if (result.canceled) return null;
  return result.filePaths[0];
});

// Track active processing processes keyed by eventId
const activeProcesses = new Map();

// IPC: Spawn the Python processing pipeline for an event
ipcMain.handle('python:processEvent', async (event, { eventId, photosDir, apiBaseUrl, token }) => {
  return new Promise((resolve, reject) => {
    const scriptPath = getScriptPath('process_event.py');

    const proc = spawn(getPythonExecutable(), [
      scriptPath,
      '--event-id', String(eventId),
      '--photos-dir', photosDir,
      '--api-url', apiBaseUrl,
      '--token', token,
    ]);

    activeProcesses.set(eventId, proc);

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      const text = data.toString();
      stdout += text;
      mainWindow.webContents.send('python:processOutput', { eventId, text });
    });

    proc.stderr.on('data', (data) => {
      const text = data.toString();
      stderr += text;
      mainWindow.webContents.send('python:processOutput', { eventId, text });
    });

    proc.on('close', (code) => {
      activeProcesses.delete(eventId);
      if (code === 0) {
        resolve({ success: true, stdout });
      } else {
        resolve({ success: false, code, stderr });
      }
    });

    proc.on('error', (err) => {
      activeProcesses.delete(eventId);
      resolve({ success: false, error: err.message });
    });
  });
});

// IPC: Stop an active processing pipeline for an event
ipcMain.handle('python:stopProcessEvent', async (event, { eventId }) => {
  const proc = activeProcesses.get(eventId);
  if (proc) {
    proc.kill('SIGTERM');
    activeProcesses.delete(eventId);
    return { stopped: true };
  }
  return { stopped: false };
});

// IPC: Extract embedding from a single photo (for walk-in retrieval)
ipcMain.handle('python:extractEmbedding', async (event, { imagePath }) => {
  return new Promise((resolve, reject) => {
    const scriptPath = getScriptPath('extract_embedding.py');

    const proc = spawn(getPythonExecutable(), [scriptPath, '--image', imagePath]);

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code === 0) {
        try {
          // InsightFace prints model-loading lines to stdout before the JSON — find the JSON line
          const jsonLine = stdout.split('\n').map(l => l.trim()).find(l => l.startsWith('{'));
          if (!jsonLine) throw new Error('No JSON line in output');
          const parsed = JSON.parse(jsonLine);
          const embedding = Array.isArray(parsed) ? parsed : parsed.embedding;
          if (!embedding) throw new Error('No embedding field in JSON');
          resolve({ success: true, embedding });
        } catch (err) {
          resolve({ success: false, error: `Failed to parse embedding output: ${err.message}. stdout: ${stdout.trim()}` });
        }
      } else {
        resolve({ success: false, error: `Python exit code ${code}. stderr: ${stderr.trim()} stdout: ${stdout.trim()}` });
      }
    });

    proc.on('error', (err) => {
      resolve({ success: false, error: err.message });
    });
  });
});

// IPC: Save a captured photo from the camera to a temp file
ipcMain.handle('file:saveTempPhoto', async (event, { dataUrl }) => {
  const fs = require('fs');
  const os = require('os');
  const tempDir = os.tmpdir();
  const filePath = path.join(tempDir, `facesorter_capture_${Date.now()}.jpg`);

  const base64Data = dataUrl.replace(/^data:image\/\w+;base64,/, '');
  fs.writeFileSync(filePath, Buffer.from(base64Data, 'base64'));

  return filePath;
});

// IPC: Get this device's base URL for the photo file server
ipcMain.handle('device:getBaseUrl', () => {
  return `http://${getLocalIpAddress()}:${PHOTO_SERVER_PORT}`;
});

// IPC: Scan a directory and return all image files
ipcMain.handle('fs:scanDirectory', async (event, { dirPath }) => {
  const fs = require('fs');
  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    return entries
      .filter((e) => e.isFile() && ALLOWED_EXTENSIONS.has(path.extname(e.name).toLowerCase()))
      .map((e) => ({
        filename: e.name,
        localPath: path.join(dirPath, e.name),
      }));
  } catch (err) {
    return [];
  }
});
