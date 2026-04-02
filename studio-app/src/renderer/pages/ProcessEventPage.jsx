import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { eventsApi, photosApi } from '../services/api';
import { API_BASE_URL } from '../services/api';
import { useAuthStore } from '../store/authStore';
import Spinner from '../components/Spinner';

function deriveDirectory(localPath) {
  if (!localPath) return null;
  const lastSlash = Math.max(localPath.lastIndexOf('/'), localPath.lastIndexOf('\\'));
  return lastSlash > 0 ? localPath.substring(0, lastSlash) : null;
}

export default function ProcessEventPage() {
  const { eventId } = useParams();
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef(null);

  const token = useAuthStore((s) => s.token);

  const { data: event } = useQuery({
    queryKey: ['events', eventId],
    queryFn: () => eventsApi.get(eventId),
  });

  const { data: unprocessedPhotos } = useQuery({
    queryKey: ['photos', eventId, 'unprocessed'],
    queryFn: () => photosApi.listUnprocessed(eventId),
  });

  const photosDir = deriveDirectory(unprocessedPhotos?.[0]?.localPath);
  const canProcess = !!photosDir && unprocessedPhotos?.length > 0;

  useEffect(() => {
    if (!window.electronAPI) return;

    const unsubscribe = window.electronAPI.onProcessOutput(({ eventId: eid, text }) => {
      if (String(eid) === String(eventId)) {
        setLogs((prev) => [...prev, text]);
      }
    });

    return unsubscribe;
  }, [eventId]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleProcess = async () => {
    if (!window.electronAPI) {
      alert('Processing requires the Electron desktop app.');
      return;
    }

    setProcessing(true);
    setResult(null);
    setLogs([]);

    const res = await window.electronAPI.processEvent({
      eventId: Number(eventId),
      photosDir,
      apiBaseUrl: API_BASE_URL,
      token,
    });

    setResult(res);
    setProcessing(false);
  };

  const handleStop = async () => {
    if (!window.electronAPI) return;
    await window.electronAPI.stopProcessEvent({ eventId: Number(eventId) });
    setProcessing(false);
    setResult({ success: false, code: -1 });
    setLogs((prev) => [...prev, 'Processing stopped by user.']);
  };

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <Link to={`/events/${eventId}`} className="text-gray-400 hover:text-gray-600 text-sm">
          Event
        </Link>
        <span className="text-gray-300">/</span>
        <h1 className="page-title">Process Event</h1>
      </div>

      <div className="card mb-6">
        <h2 className="text-lg font-semibold mb-2">
          {event?.name || `Event #${eventId}`}
        </h2>
        <p className="text-sm text-gray-500 mb-4">
          This will run the face detection and embedding extraction pipeline on all
          unprocessed photos. Already-processed photos are skipped.
        </p>

        {unprocessedPhotos === undefined ? (
          <div className="flex items-center gap-2 mb-4">
            <Spinner size="sm" />
            <span className="text-sm text-gray-500">Loading photos...</span>
          </div>
        ) : canProcess ? (
          <p className="text-sm text-gray-600 mb-4">
            <span className="font-semibold">{unprocessedPhotos.length}</span> unprocessed photo{unprocessedPhotos.length !== 1 ? 's' : ''} in{' '}
            <span className="font-mono">{photosDir}</span>
          </p>
        ) : (
          <p className="text-sm text-yellow-600 mb-4">
            No unprocessed photos found.{' '}
            <Link to={`/events/${eventId}/upload`} className="underline">
              Upload photos first
            </Link>.
          </p>
        )}

        <div className="flex items-center gap-3">
          <button
            onClick={handleProcess}
            disabled={processing || !canProcess}
            className="btn-primary"
          >
            {processing ? (
              <span className="flex items-center gap-2">
                <Spinner size="sm" /> Processing...
              </span>
            ) : (
              'Start Processing'
            )}
          </button>

          {processing && (
            <button
              onClick={handleStop}
              className="btn-secondary text-red-600 border-red-300 hover:bg-red-50"
            >
              Stop
            </button>
          )}
        </div>

        {result && (
          <div className={`mt-4 p-3 rounded-lg text-sm ${
            result.success ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
          }`}>
            {result.success
              ? 'Processing completed successfully.'
              : result.code === -1
              ? 'Processing was stopped.'
              : `Processing failed (exit code ${result.code}). Check the logs below.`}
          </div>
        )}
      </div>

      {logs.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Processing Log</h3>
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 max-h-96 overflow-y-auto font-mono text-xs leading-relaxed">
            {logs.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}
