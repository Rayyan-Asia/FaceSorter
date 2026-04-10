import React, { useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ordersApi, matchingApi } from '../services/api';
import CameraCapture from '../components/CameraCapture';
import Spinner from '../components/Spinner';

export default function WalkInRetrievalPage() {
  const { orderId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [step, setStep] = useState('camera'); // camera | searching | results | confirm
  const [capturedEmbedding, setCapturedEmbedding] = useState(null);
  const [expandedPhoto, setExpandedPhoto] = useState(null);
  const [matchedPhotos, setMatchedPhotos] = useState([]);
  const [selectedPhotoIds, setSelectedPhotoIds] = useState(new Set());
  const [searchError, setSearchError] = useState(null);
  const [threshold, setThreshold] = useState(0.65);

  const { data: order } = useQuery({
    queryKey: ['orders', orderId],
    queryFn: () => ordersApi.get(orderId),
  });

  const finalizeMutation = useMutation({
    mutationFn: async ({ orderId, photoIds }) => {
      await ordersApi.addItems(orderId, photoIds);
      await ordersApi.updateStatus(orderId, 'PENDING');
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      setStep('confirm');
    },
  });

  const runSearch = async (embedding) => {
    setStep('searching');
    setSearchError(null);

    try {
      const eventId = order?.eventId;
      if (!eventId) {
        setSearchError('Order has no linked event.');
        setStep('camera');
        return;
      }

      const results = await matchingApi.search(eventId, embedding, threshold);
      const photos = results.matchedPhotos || [];
      setMatchedPhotos(photos);
      setSelectedPhotoIds(new Set(photos.map((p) => p.photoId)));
      setStep('results');
    } catch (err) {
      setSearchError(err.message || 'Search failed.');
      setStep('results');
    }
  };

  const handleCapture = async (dataUrl) => {
    setStep('searching');
    setSearchError(null);

    try {
      let embedding;

      if (window.electronAPI) {
        const filePath = await window.electronAPI.saveTempPhoto({ dataUrl });
        const result = await window.electronAPI.extractEmbedding({ imagePath: filePath });

        if (!result.success) {
          setSearchError(result.error || 'Failed to extract face embedding.');
          setStep('camera');
          return;
        }
        embedding = result.embedding;
      } else {
        setSearchError('Embedding extraction requires the Electron desktop app.');
        setStep('camera');
        return;
      }

      setCapturedEmbedding(embedding);
      await runSearch(embedding);
    } catch (err) {
      setSearchError(err.message || 'Search failed.');
      setStep('camera');
    }
  };

  const handleSearchAgain = () => {
    if (capturedEmbedding) {
      runSearch(capturedEmbedding);
    }
  };

  const handleRetake = () => {
    setCapturedEmbedding(null);
    setMatchedPhotos([]);
    setSelectedPhotoIds(new Set());
    setSearchError(null);
    setStep('camera');
  };

  const togglePhoto = (photoId) => {
    setSelectedPhotoIds((prev) => {
      const next = new Set(prev);
      if (next.has(photoId)) {
        next.delete(photoId);
      } else {
        next.add(photoId);
      }
      return next;
    });
  };

  const handleFinalize = () => {
    finalizeMutation.mutate({
      orderId: Number(orderId),
      photoIds: Array.from(selectedPhotoIds),
    });
  };

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <Link to="/orders" className="text-gray-400 hover:text-gray-600 text-sm">
          Orders
        </Link>
        <span className="text-gray-300">/</span>
        <h1 className="page-title">Walk-in Retrieval — Order #{orderId}</h1>
      </div>

      {searchError && (
        <div className="mb-4 p-3 bg-red-50 text-red-700 text-sm rounded-lg">{searchError}</div>
      )}

      {step === 'camera' && (
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Take Customer Photo</h2>
          <p className="text-sm text-gray-500 mb-4">
            Position the customer facing the camera and capture their photo. The system
            will search for their photos in the event.
          </p>

          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700">Match sensitivity</label>
              <span className="text-sm font-semibold text-primary-600">
                {threshold <= 0.45 ? 'Strict' : threshold <= 0.65 ? 'Balanced' : 'Broad'}
              </span>
            </div>
            <input
              type="range"
              min="0.30"
              max="0.90"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-primary-500"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>Fewer, closer matches</span>
              <span>More, looser matches</span>
            </div>
          </div>

          <CameraCapture
            onCapture={handleCapture}
            onCancel={() => navigate('/orders')}
          />
        </div>
      )}

      {step === 'searching' && (
        <div className="card text-center py-12">
          <Spinner size="lg" />
          <p className="mt-4 text-gray-500">Extracting embedding and searching for matches...</p>
        </div>
      )}

      {step === 'results' && (
        <div>
          <div className="card mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700">Match sensitivity</label>
              <span className="text-sm font-semibold text-primary-600">
                {threshold <= 0.45 ? 'Strict' : threshold <= 0.65 ? 'Balanced' : 'Broad'} ({threshold.toFixed(2)})
              </span>
            </div>
            <input
              type="range"
              min="0.30"
              max="0.90"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-primary-500"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>Fewer, closer matches</span>
              <span>More, looser matches</span>
            </div>
            <div className="flex gap-3 mt-4">
              <button onClick={handleRetake} className="btn-secondary">
                Retake Photo
              </button>
              <button onClick={handleSearchAgain} className="btn-primary">
                Search Again
              </button>
            </div>
          </div>

          <div className="card mb-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold">
                  {matchedPhotos.length} Matching Photos Found
                </h2>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setSelectedPhotoIds(new Set(matchedPhotos.map((p) => p.photoId)))}
                  className="text-sm text-primary-600 hover:underline"
                >
                  Select All
                </button>
                <button
                  onClick={() => setSelectedPhotoIds(new Set())}
                  className="text-sm text-gray-500 hover:underline"
                >
                  Deselect All
                </button>
              </div>
            </div>

            {matchedPhotos.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-500">No matching photos found.</p>
                <p className="text-xs text-gray-400 mt-1">Try increasing the sensitivity slider above and search again.</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {matchedPhotos.map((photo) => (
                  <div
                    key={photo.photoId}
                    onClick={() => togglePhoto(photo.photoId)}
                    className={`relative cursor-pointer rounded-lg overflow-hidden border-2 transition-colors ${
                      selectedPhotoIds.has(photo.photoId)
                        ? 'border-primary-500'
                        : 'border-transparent hover:border-gray-300'
                    }`}
                  >
                    <img
                      src={`http://localhost:4567/photo?path=${encodeURIComponent(photo.localPath)}&w=400`}
                      alt={photo.filename}
                      className="w-full h-32 object-cover"
                      loading="lazy"
                    />
                    {selectedPhotoIds.has(photo.photoId) && (
                      <div className="absolute top-2 right-2 w-6 h-6 bg-primary-500 rounded-full flex items-center justify-center">
                        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); setExpandedPhoto(photo); }}
                      className="absolute bottom-8 right-1 w-6 h-6 bg-black/50 hover:bg-black/70 rounded flex items-center justify-center transition-colors"
                      title="View full photo"
                    >
                      <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                      </svg>
                    </button>
                    <p className="text-xs text-gray-500 p-1 truncate">{photo.filename}</p>
                    {photo.similarityScore > 0 && (
                      <p className="text-xs text-gray-400 px-1 pb-1">
                        {(photo.similarityScore * 100).toFixed(0)}% match
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {matchedPhotos.length > 0 && (
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500">
                {selectedPhotoIds.size} of {matchedPhotos.length} photos selected
              </p>
              <button
                onClick={handleFinalize}
                disabled={selectedPhotoIds.size === 0 || finalizeMutation.isPending}
                className="btn-primary"
              >
                {finalizeMutation.isPending ? 'Saving...' : 'Finalize Order'}
              </button>
            </div>
          )}

          {finalizeMutation.isError && (
            <p className="mt-3 text-sm text-red-600">
              Failed to finalize: {finalizeMutation.error?.message}
            </p>
          )}
        </div>
      )}

      {expandedPhoto && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setExpandedPhoto(null)}
        >
          <div className="relative max-w-4xl max-h-full" onClick={(e) => e.stopPropagation()}>
            <img
              src={`http://localhost:4567/photo?path=${encodeURIComponent(expandedPhoto.localPath)}`}
              alt={expandedPhoto.filename}
              className="max-w-full max-h-[85vh] object-contain rounded-lg"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-black/50 rounded-b-lg px-4 py-2 flex items-center justify-between">
              <div>
                <p className="text-white text-sm font-medium">{expandedPhoto.filename}</p>
                {expandedPhoto.similarityScore > 0 && (
                  <p className="text-gray-300 text-xs">{(expandedPhoto.similarityScore * 100).toFixed(0)}% match</p>
                )}
              </div>
              <button
                onClick={() => { togglePhoto(expandedPhoto.photoId); setExpandedPhoto(null); }}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  selectedPhotoIds.has(expandedPhoto.photoId)
                    ? 'bg-primary-500 text-white hover:bg-primary-600'
                    : 'bg-white text-gray-900 hover:bg-gray-100'
                }`}
              >
                {selectedPhotoIds.has(expandedPhoto.photoId) ? 'Deselect' : 'Select'}
              </button>
            </div>
            <button
              onClick={() => setExpandedPhoto(null)}
              className="absolute top-2 right-2 w-8 h-8 bg-black/50 hover:bg-black/70 rounded-full flex items-center justify-center transition-colors"
            >
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {step === 'confirm' && (
        <div className="card text-center py-8">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">Order Finalized</h2>
          <p className="text-gray-500 mb-6">
            Order #{orderId} has been updated with {selectedPhotoIds.size} selected photos.
          </p>
          <div className="flex gap-3 justify-center">
            <Link to="/orders" className="btn-secondary">View Orders</Link>
            <Link to="/orders/fulfillment" className="btn-primary">Go to Fulfillment</Link>
          </div>
        </div>
      )}
    </div>
  );
}
