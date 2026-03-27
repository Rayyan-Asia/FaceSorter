import { useRef, useState, useCallback, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { searchFaces, type MatchedPhoto } from "../services/api";

type Stage = "preview" | "capturing" | "reviewing" | "searching";

export default function CameraPage() {
  const { orderId } = useParams<{ orderId: string }>();
  const navigate = useNavigate();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [stage, setStage] = useState<Stage>("preview");
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setStage("preview");
    } catch {
      setError(
        "Unable to access camera. Please allow camera permissions and try again.",
      );
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => {
    startCamera();
    return stopCamera;
  }, [startCamera, stopCamera]);

  function capturePhoto() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    setCapturedImage(canvas.toDataURL("image/jpeg", 0.9));
    stopCamera();
    setStage("reviewing");
  }

  function retake() {
    setCapturedImage(null);
    setError(null);
    startCamera();
  }

  async function submitPhoto() {
    if (!capturedImage || !orderId) return;

    setStage("searching");
    setError(null);

    try {
      const res = await fetch(capturedImage);
      const blob = await res.blob();
      const matches: MatchedPhoto[] = await searchFaces(orderId, blob);

      navigate(`/photos/${orderId}`, { state: { matches } });
    } catch {
      setError("Face search failed. Please try again.");
      setStage("reviewing");
    }
  }

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-xl font-bold text-gray-900 mb-1">Take a Selfie</h2>
      <p className="text-sm text-gray-500 mb-6 text-center">
        Look directly at the camera so we can find your photos.
      </p>

      <div className="relative w-full max-w-sm aspect-[3/4] bg-black rounded-xl overflow-hidden">
        {stage === "preview" && (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover mirror"
            style={{ transform: "scaleX(-1)" }}
          />
        )}

        {(stage === "reviewing" || stage === "searching") && capturedImage && (
          <img
            src={capturedImage}
            alt="Captured selfie"
            className="w-full h-full object-cover"
            style={{ transform: "scaleX(-1)" }}
          />
        )}

        {stage === "searching" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <div className="text-white text-center">
              <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-white border-r-transparent mb-3" />
              <p className="text-sm">Searching for your photos...</p>
            </div>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />

      {error && (
        <p className="text-sm text-red-600 mt-4 text-center" role="alert">
          {error}
        </p>
      )}

      <div className="mt-6 flex gap-3 w-full max-w-sm">
        {stage === "preview" && (
          <button
            onClick={capturePhoto}
            className="flex-1 rounded-lg bg-blue-600 px-4 py-3 text-white font-medium
                       hover:bg-blue-700 transition-colors"
          >
            Capture
          </button>
        )}

        {stage === "reviewing" && (
          <>
            <button
              onClick={retake}
              className="flex-1 rounded-lg border border-gray-300 px-4 py-3 text-gray-700 font-medium
                         hover:bg-gray-50 transition-colors"
            >
              Retake
            </button>
            <button
              onClick={submitPhoto}
              className="flex-1 rounded-lg bg-blue-600 px-4 py-3 text-white font-medium
                         hover:bg-blue-700 transition-colors"
            >
              Search Photos
            </button>
          </>
        )}
      </div>
    </div>
  );
}
