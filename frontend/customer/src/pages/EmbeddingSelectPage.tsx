import { useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import { getPhotosByEmbeddings, type EmbeddingCandidate } from "../services/api";

export default function EmbeddingSelectPage() {
  const { orderId } = useParams<{ orderId: string }>();
  const location = useLocation();
  const navigate = useNavigate();

  const candidates: EmbeddingCandidate[] =
    (location.state as { candidates?: EmbeddingCandidate[] })?.candidates ?? [];

  const [selected, setSelected] = useState<Set<number>>(
    () => new Set(candidates.map((c) => c.embeddingId)),
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggle(embeddingId: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(embeddingId)) {
        next.delete(embeddingId);
      } else {
        next.add(embeddingId);
      }
      return next;
    });
  }

  async function handleConfirm() {
    if (!orderId || selected.size === 0) return;

    setLoading(true);
    setError(null);

    try {
      const photos = await getPhotosByEmbeddings(orderId, Array.from(selected));
      navigate(`/photos/${orderId}`, { state: { matches: photos } });
    } catch {
      setError("Failed to load photos. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  if (candidates.length === 0) {
    return (
      <div className="flex flex-col items-center pt-16 text-center">
        <h2 className="text-xl font-bold text-gray-900 mb-2">No Faces Found</h2>
        <p className="text-gray-500 mb-6">
          We could not find any matching faces in this event. Try taking another selfie.
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
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-900">Which faces are you?</h2>
        <p className="text-sm text-gray-500 mt-1">
          We found {candidates.length} possible match{candidates.length !== 1 ? "es" : ""}.
          Select all that are you, then tap Continue.
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-6">
        {candidates.map((candidate) => {
          const isSelected = selected.has(candidate.embeddingId);
          return (
            <button
              key={candidate.embeddingId}
              onClick={() => toggle(candidate.embeddingId)}
              className={`relative aspect-square rounded-lg overflow-hidden border-2 transition-all
                ${isSelected ? "border-blue-600 ring-2 ring-blue-200" : "border-transparent opacity-50"}`}
            >
              {candidate.representativePhotoId != null ? (
                <img
                  src={`/api/photos/${candidate.representativePhotoId}`}
                  alt={candidate.representativeFilename ?? "Face"}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
              ) : (
                <div className="w-full h-full bg-gray-100 flex items-center justify-center">
                  <svg className="h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
                  </svg>
                </div>
              )}
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
          disabled={loading || selected.size === 0}
          className="flex-1 rounded-lg bg-blue-600 px-4 py-3 text-white font-medium
                     hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Loading..." : `Continue (${selected.size})`}
        </button>
      </div>
    </div>
  );
}
