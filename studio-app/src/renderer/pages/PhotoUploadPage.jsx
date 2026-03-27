import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { photosApi } from '../services/api';
import Spinner from '../components/Spinner';

export default function PhotoUploadPage() {
  const { eventId } = useParams();
  const queryClient = useQueryClient();
  const [selectedDir, setSelectedDir] = useState(null);
  const [photoFiles, setPhotoFiles] = useState([]);
  const [scanning, setScanning] = useState(false);

  const registerMutation = useMutation({
    mutationFn: (photos) => photosApi.register(eventId, photos),
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

    // We scan on the renderer side for display only; actual file reading happens during processing.
    // The backend just needs filenames and local paths.
    try {
      // Use a simple fetch to list files - in Electron we'd use Node fs via IPC
      // For now, we'll let the user confirm and send the directory path to the backend
      setPhotoFiles([]);
      setScanning(false);
    } catch {
      setScanning(false);
    }
  };

  const handleRegister = () => {
    if (!selectedDir) return;

    // Register the photo directory with the backend.
    // The backend will record the directory path and the event association.
    registerMutation.mutate([
      {
        directory: selectedDir,
        eventId: Number(eventId),
      },
    ]);
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
          Choose the local folder containing event photos. The photos remain on this computer;
          only metadata (filenames and paths) is registered with the server.
        </p>

        <div className="flex items-center gap-4">
          <button onClick={handleSelectFolder} className="btn-primary">
            Choose Folder
          </button>

          {selectedDir && (
            <span className="text-sm text-gray-600 font-mono">{selectedDir}</span>
          )}
        </div>

        {scanning && (
          <div className="flex items-center gap-2 mt-4">
            <Spinner size="sm" />
            <span className="text-sm text-gray-500">Scanning folder...</span>
          </div>
        )}

        {selectedDir && !scanning && (
          <div className="mt-6 pt-4 border-t border-gray-200">
            <button
              onClick={handleRegister}
              disabled={registerMutation.isPending}
              className="btn-primary"
            >
              {registerMutation.isPending ? 'Registering...' : 'Register Photos'}
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
