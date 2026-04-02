const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openDirectory: () => ipcRenderer.invoke('dialog:openDirectory'),

  processEvent: (params) => ipcRenderer.invoke('python:processEvent', params),
  stopProcessEvent: (params) => ipcRenderer.invoke('python:stopProcessEvent', params),

  onProcessOutput: (callback) => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on('python:processOutput', handler);
    return () => ipcRenderer.removeListener('python:processOutput', handler);
  },

  extractEmbedding: (params) => ipcRenderer.invoke('python:extractEmbedding', params),

  saveTempPhoto: (params) => ipcRenderer.invoke('file:saveTempPhoto', params),

  getDeviceBaseUrl: () => ipcRenderer.invoke('device:getBaseUrl'),
  scanDirectory: (params) => ipcRenderer.invoke('fs:scanDirectory', params),
});
