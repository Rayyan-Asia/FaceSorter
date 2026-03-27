const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

const isDev = process.env.NODE_ENV === 'development';

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

// IPC: Spawn the Python processing pipeline for an event
ipcMain.handle('python:processEvent', async (event, { eventId, photosDir, apiBaseUrl }) => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, '../../scripts/process_event.py');

    const proc = spawn('python3', [
      scriptPath,
      '--event-id', String(eventId),
      '--photos-dir', photosDir,
      '--api-url', apiBaseUrl,
    ]);

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
      if (code === 0) {
        resolve({ success: true, stdout });
      } else {
        resolve({ success: false, code, stderr });
      }
    });

    proc.on('error', (err) => {
      resolve({ success: false, error: err.message });
    });
  });
});

// IPC: Extract embedding from a single photo (for walk-in retrieval)
ipcMain.handle('python:extractEmbedding', async (event, { imagePath }) => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, '../../scripts/extract_embedding.py');

    const proc = spawn('python3', [scriptPath, '--image', imagePath]);

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
          const embedding = JSON.parse(stdout.trim());
          resolve({ success: true, embedding });
        } catch {
          resolve({ success: false, error: 'Failed to parse embedding output' });
        }
      } else {
        resolve({ success: false, error: stderr });
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
