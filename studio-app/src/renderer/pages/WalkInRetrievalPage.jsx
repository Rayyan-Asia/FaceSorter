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
  const [matchedPhotos, setMatchedPhotos] = useState([]);
  const [selectedPhotoIds, setSelectedPhotoIds] = useState(new Set());
  const [searchError, setSearchError] = useState(null);

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

  const handleCapture = async (dataUrl) => {
    setStep('searching');
    setSearchError(null);

    try {
      let embedding;

      if (window.electronAPI) {
        // Save the captured photo to a temp file and extract embedding locally
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

      // Search for matching photos in the event
      const eventId = order?.eventId;
      if (!eventId) {
        setSearchError('Order has no linked event.');
        setStep('camera');
        return;
      }

      const results = await matchingApi.search(eventId, embedding);
      const photos = results.matchedPhotos || [];
      setMatchedPhotos(photos);
      setSelectedPhotoIds(new Set(photos.map((p) => p.photoId)));
      setStep('results');
    } catch (err) {
      setSearchError(err.message || 'Search failed.');
      setStep('camera');
    }
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
          <div className="card mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">
                {matchedPhotos.length} Matching Photos Found
              </h2>
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
                <button onClick={() => setStep('camera')} className="btn-secondary mt-4">
                  Try Again
                </button>
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
                      src={`http://localhost:4567/photo?path=${encodeURIComponent(photo.localPath)}`}
                      alt={photo.filename}
                      className="w-full h-32 object-cover"
                    />
                    {selectedPhotoIds.has(photo.photoId) && (
                      <div className="absolute top-2 right-2 w-6 h-6 bg-primary-500 rounded-full flex items-center justify-center">
                        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                    )}
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
              <div className="flex gap-3">
                <button onClick={() => setStep('camera')} className="btn-secondary">
                  Retake Photo
                </button>
                <button
                  onClick={handleFinalize}
                  disabled={selectedPhotoIds.size === 0 || finalizeMutation.isPending}
                  className="btn-primary"
                >
                  {finalizeMutation.isPending ? 'Saving...' : 'Finalize Order'}
                </button>
              </div>
            </div>
          )}

          {finalizeMutation.isError && (
            <p className="mt-3 text-sm text-red-600">
              Failed to finalize: {finalizeMutation.error?.message}
            </p>
          )}
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
