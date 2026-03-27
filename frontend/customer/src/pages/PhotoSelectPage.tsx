import { useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import { confirmOrder, type MatchedPhoto } from "../services/api";

export default function PhotoSelectPage() {
  const { orderId } = useParams<{ orderId: string }>();
  const location = useLocation();
  const navigate = useNavigate();

  const matches: MatchedPhoto[] = (location.state as { matches?: MatchedPhoto[] })?.matches ?? [];

  const [selected, setSelected] = useState<Set<number>>(() => new Set(matches.map((m) => m.photoId)));
  const [submitting, setSubmitting] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggle(photoId: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(photoId)) {
        next.delete(photoId);
      } else {
        next.add(photoId);
      }
      return next;
    });
  }

  function selectAll() {
    setSelected(new Set(matches.map((m) => m.photoId)));
  }

  function deselectAll() {
    setSelected(new Set());
  }

  async function handleConfirm() {
    if (!orderId || selected.size === 0) return;

    setSubmitting(true);
    setError(null);

    try {
      await confirmOrder(orderId, Array.from(selected));
      setConfirmed(true);
    } catch {
      setError("Failed to confirm order. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  if (confirmed) {
    return (
      <div className="flex flex-col items-center pt-16 text-center">
        <div className="h-16 w-16 rounded-full bg-green-100 flex items-center justify-center mb-4">
          <svg className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Order Confirmed</h2>
        <p className="text-gray-500 mb-1">
          {selected.size} photo{selected.size !== 1 ? "s" : ""} selected for order #{orderId}.
        </p>
        <p className="text-sm text-gray-400">
          The studio will prepare your photos for collection.
        </p>
      </div>
    );
  }

  if (matches.length === 0) {
    return (
      <div className="flex flex-col items-center pt-16 text-center">
        <h2 className="text-xl font-bold text-gray-900 mb-2">No Photos Found</h2>
        <p className="text-gray-500 mb-6">
          We could not find any matching photos. Try taking another selfie.
        </p>
        <button
          onClick={() => navigate(`/camera/${orderId}`)}
          className="rounded-lg bg-blue-600 px-6 py-3 text-white font-medium hover:bg-blue-700 transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Your Photos</h2>
          <p className="text-sm text-gray-500">
            {matches.length} match{matches.length !== 1 ? "es" : ""} found.
            Deselect any that are not you.
          </p>
        </div>
        <div className="flex gap-2 text-sm">
          <button onClick={selectAll} className="text-blue-600 hover:underline">
            All
          </button>
          <span className="text-gray-300">|</span>
          <button onClick={deselectAll} className="text-blue-600 hover:underline">
            None
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-6">
        {matches.map((photo) => {
          const isSelected = selected.has(photo.photoId);
          return (
            <button
              key={photo.photoId}
              onClick={() => toggle(photo.photoId)}
              className={`relative aspect-square rounded-lg overflow-hidden border-2 transition-all
                ${isSelected ? "border-blue-600 ring-2 ring-blue-200" : "border-transparent opacity-60"}`}
            >
              <img
                src={photo.url}
                alt={photo.filename}
                className="w-full h-full object-cover"
                loading="lazy"
              />
              {isSelected && (
                <div className="absolute top-2 right-2 h-6 w-6 rounded-full bg-blue-600 flex items-center justify-center">
                  <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                  </svg>
                </div>
              )}
            </button>
          );
        })}
      </div>

      {error && (
        <p className="text-sm text-red-600 text-center mb-4" role="alert">
          {error}
        </p>
      )}

      <div className="sticky bottom-4 flex gap-3">
        <button
          onClick={() => navigate(`/camera/${orderId}`)}
          className="flex-1 rounded-lg border border-gray-300 px-4 py-3 text-gray-700 font-medium
                     hover:bg-gray-50 transition-colors"
        >
          Retake Selfie
        </button>
        <button
          onClick={handleConfirm}
          disabled={submitting || selected.size === 0}
          className="flex-1 rounded-lg bg-blue-600 px-4 py-3 text-white font-medium
                     hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {submitting ? "Confirming..." : `Confirm (${selected.size})`}
        </button>
      </div>
    </div>
  );
}
