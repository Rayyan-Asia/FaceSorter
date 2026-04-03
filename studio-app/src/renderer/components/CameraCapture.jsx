import React, { useRef, useState, useCallback, useEffect } from 'react';

export default function CameraCapture({ onCapture, onCancel }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const dropZoneRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);
  const [dragging, setDragging] = useState(false);

  // Enumerate video input devices
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((all) => {
      const videoDevices = all.filter((d) => d.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
    });
  }, []);

  // Start/restart camera when selectedDeviceId changes
  useEffect(() => {
    if (!selectedDeviceId) return;

    let cancelled = false;

    // Stop existing stream first
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setReady(false);
    setError(null);

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } },
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => setReady(true);
        }
      } catch {
        if (!cancelled) setError('Camera access denied or unavailable.');
      }
    }

    startCamera();

    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, [selectedDeviceId]);

  const handleCapture = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    onCapture(dataUrl);
  }, [onCapture]);

  const handleFileRead = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => onCapture(e.target.result);
    reader.readAsDataURL(file);
  }, [onCapture]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileRead(file);
  }, [handleFileRead]);

  const handleDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const handleDragLeave = () => setDragging(false);

  const handleFileInput = (e) => handleFileRead(e.target.files[0]);

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-600 mb-4">{error}</p>
        <button onClick={onCancel} className="btn-secondary">Go Back</button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Camera selector */}
      {devices.length > 1 && (
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-600 whitespace-nowrap">Camera source:</label>
          <select
            className="input-field"
            value={selectedDeviceId || ''}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
          >
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Camera ${devices.indexOf(d) + 1}`}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Live camera feed */}
      <div className="relative rounded-xl overflow-hidden bg-black aspect-video">
        <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
      </div>
      <canvas ref={canvasRef} className="hidden" />

      <div className="flex gap-3 justify-center">
        <button onClick={onCancel} className="btn-secondary">Cancel</button>
        <button onClick={handleCapture} disabled={!ready} className="btn-primary">
          Capture Photo
        </button>
      </div>

      {/* Drag-and-drop alternative */}
      <div className="relative">
        <div className="flex items-center gap-3 my-2">
          <div className="flex-1 border-t border-gray-200" />
          <span className="text-xs text-gray-400">or use an existing photo</span>
          <div className="flex-1 border-t border-gray-200" />
        </div>

        <div
          ref={dropZoneRef}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
            dragging ? 'border-primary-400 bg-primary-50' : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
          }`}
          onClick={() => document.getElementById('photo-file-input').click()}
        >
          <svg className="w-8 h-8 text-gray-400 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="text-sm text-gray-500">
            {dragging ? 'Drop photo here' : 'Drag & drop a photo here, or click to browse'}
          </p>
        </div>

        <input
          id="photo-file-input"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileInput}
        />
      </div>
    </div>
  );
}
