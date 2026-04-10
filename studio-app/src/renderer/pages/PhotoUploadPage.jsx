import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { photosApi } from '../services/api';
import Spinner from '../components/Spinner';

const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']);

export default function PhotoUploadPage() {
  const { eventId } = useParams();
  const queryClient = useQueryClient();

  const [selectedDir, setSelectedDir] = useState(null);
  const [photoFiles, setPhotoFiles] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [deviceBaseUrl, setDeviceBaseUrl] = useState(null);

  const registerMutation = useMutation({
    mutationFn: (photos) => photosApi.register(Number(eventId), photos),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['events', eventId] });
    },
  });

  const handleSelectFolder = async () => {
    if (!window.electronAPI) {
      alert('Folder selection requires the Electron desktop app.');
      return;
    }

    const dirPath = await window.electronAPI.openDirectory();
    if (!dirPath) return;

    setSelectedDir(dirPath);
    setScanning(true);
    registerMutation.reset();

    const [files, baseUrl] = await Promise.all([
      window.electronAPI.scanDirectory({ dirPath }),
      window.electronAPI.getDeviceBaseUrl(),
    ]);

    setPhotoFiles(files);
    setDeviceBaseUrl(baseUrl);
    setScanning(false);
  };

  const handleRegister = () => {
    if (!photoFiles.length || !deviceBaseUrl) return;

    const photos = photoFiles.map((f) => ({
      filename: f.filename,
      localPath: f.localPath,
      url: `${deviceBaseUrl}/photo?path=${encodeURIComponent(f.localPath)}`,
      fileHash: f.fileHash,
    }));

    registerMutation.mutate(photos);
  };

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <Link to={`/events/${eventId}`} className="text-gray-400 hover:text-gray-600 text-sm">
          Event
        </Link>
        <span className="text-gray-300">/</span>
        <h1 className="page-title">Upload Photos</h1>
      </div>

      <div className="card">
        <h2 className="text-lg font-semibold mb-4">Select Photo Folder</h2>
        <p className="text-sm text-gray-500 mb-4">
          Choose the local folder containing event photos. Photos are served directly
          from this device — make sure customers can reach it on the network.
        </p>

        <div className="flex items-center gap-4">
          <button onClick={handleSelectFolder} className="btn-primary">
            Choose Folder
          </button>
          {selectedDir && (
            <span className="text-sm text-gray-600 font-mono truncate max-w-xs">{selectedDir}</span>
          )}
        </div>

        {scanning && (
          <div className="flex items-center gap-2 mt-4">
            <Spinner size="sm" />
            <span className="text-sm text-gray-500">Scanning folder...</span>
          </div>
        )}

        {!scanning && photoFiles.length > 0 && (
          <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm text-gray-600">
            Found <span className="font-semibold">{photoFiles.length}</span> image file{photoFiles.length !== 1 ? 's' : ''}.
            {deviceBaseUrl && (
              <span className="ml-2 text-gray-400">Serving from <span className="font-mono">{deviceBaseUrl}</span></span>
            )}
          </div>
        )}

        {!scanning && selectedDir && photoFiles.length === 0 && !scanning && (
          <p className="mt-4 text-sm text-amber-600">No image files found in this folder.</p>
        )}

        {selectedDir && !scanning && photoFiles.length > 0 && (
          <div className="mt-6 pt-4 border-t border-gray-200">
            <button
              onClick={handleRegister}
              disabled={registerMutation.isPending}
              className="btn-primary"
            >
              {registerMutation.isPending ? 'Registering...' : `Register ${photoFiles.length} Photo${photoFiles.length !== 1 ? 's' : ''}`}
            </button>

            {registerMutation.isSuccess && (
              <p className="mt-3 text-sm text-green-600">
                Photos registered successfully. You can now process the event.
              </p>
            )}

            {registerMutation.isError && (
              <p className="mt-3 text-sm text-red-600">
                Registration failed: {registerMutation.error?.message}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
