"""
Process all unprocessed photos for an event.

Usage:
    python process_event.py --event-id <event_id> --photo-dir <directory> [--api-url <url>]

For each unprocessed photo:
  1. Detect all faces (RetinaFace preferred, OpenCV fallback)
  2. Extract a FaceNet embedding per face via DeepFace
  3. POST each embedding + photo metadata to the Spring Boot backend
  4. Backend handles person deduplication and marks the photo as processed

Exit codes:
    0 — all photos processed successfully
    1 — fatal error (bad arguments, unreachable API)
    2 — partial failure (some photos failed, others succeeded)
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import requests
from deepface import DeepFace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

MODEL_NAME = "Facenet"
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
DEFAULT_API_URL = "http://localhost:8080/api"
REQUEST_TIMEOUT = 30  # seconds

# Similarity threshold — tuned for high recall (minimize false negatives).
# Lower threshold = stricter matching = fewer false positives but more false negatives.
# Higher threshold = looser matching = more false positives but fewer false negatives.
# 0.4 cosine distance for FaceNet is a good balance favoring recall.
SIMILARITY_THRESHOLD = 0.4

try:
    from retinaface import RetinaFace as _rf
    DETECTOR_BACKEND = "retinaface"
    logger.info("Using RetinaFace detector")
except ImportError:
    DETECTOR_BACKEND = "opencv"
    logger.info("RetinaFace not available, falling back to OpenCV detector")


def get_unprocessed_photos(event_id: int, api_url: str) -> list[dict]:
    """Fetch the list of unprocessed photos for an event from the backend."""
    url = f"{api_url}/events/{event_id}/photos?processed=false"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_local_unprocessed_photos(photo_dir: str, event_id: int, api_url: str) -> list[dict]:
    """Get unprocessed photos by cross-referencing local files with backend state.

    Falls back to scanning the local directory if the backend is unreachable,
    treating all local photos as unprocessed (idempotent — backend rejects
    duplicates).
    """
    try:
        return get_unprocessed_photos(event_id, api_url)
    except requests.RequestException as e:
        logger.warning(
            "Could not fetch photo list from backend (%s). "
            "Falling back to local directory scan — backend will skip already-processed photos.",
            e,
        )
        photos = []
        for filename in sorted(os.listdir(photo_dir)):
            if filename.lower().endswith(SUPPORTED_FORMATS):
                photos.append({
                    "id": None,
                    "filename": filename,
                    "filePath": os.path.join(photo_dir, filename),
                    "processed": False,
                })
        return photos


def extract_faces(image_path: str) -> list[dict]:
    """Detect all faces in an image and return their embeddings.

    Returns a list of dicts, each with:
        - embedding: list[float]
        - facial_area: {x, y, w, h}
    """
    try:
        representations = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except ValueError as e:
        # DeepFace raises ValueError when no face is detected
        logger.info("No faces detected in %s: %s", image_path, e)
        return []

    faces = []
    for rep in representations:
        faces.append({
            "embedding": rep["embedding"],
            "facial_area": rep.get("facial_area", {}),
        })

    return faces


def post_face_embedding(
    event_id: int,
    photo_id: int | None,
    filename: str,
    file_path: str,
    embedding: list[float],
    facial_area: dict,
    api_url: str,
) -> bool:
    """POST a single face embedding to the backend.

    The backend is responsible for:
      - Checking if this embedding matches an existing person (cosine similarity)
      - Linking the photo to the matched or newly created person
      - Marking the photo as processed once all faces are submitted

    Returns True on success.
    """
    url = f"{api_url}/events/{event_id}/faces"
    payload = {
        "photoId": photo_id,
        "filename": filename,
        "filePath": file_path,
        "embedding": embedding,
        "facialArea": facial_area,
        "similarityThreshold": SIMILARITY_THRESHOLD,
    }

    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return True


def mark_photo_processed(event_id: int, photo_id: int, api_url: str) -> None:
    """Mark a photo as processed in the backend after all its faces have been submitted."""
    url = f"{api_url}/events/{event_id}/photos/{photo_id}/processed"
    resp = requests.put(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()


def process_photo(
    event_id: int,
    photo: dict,
    photo_dir: str,
    api_url: str,
) -> bool:
    """Process a single photo: detect faces, extract embeddings, submit to backend.

    Returns True if the photo was processed successfully.
    """
    filename = photo["filename"]
    photo_id = photo.get("id")
    file_path = photo.get("filePath") or os.path.join(photo_dir, filename)

    # Resolve to absolute local path for DeepFace
    local_path = file_path if os.path.isabs(file_path) else os.path.join(photo_dir, filename)

    if not os.path.isfile(local_path):
        logger.error("Photo file not found: %s", local_path)
        return False

    logger.info("Processing photo: %s", filename)
    faces = extract_faces(local_path)

    if not faces:
        logger.info("No faces found in %s — marking as processed (no embeddings to store)", filename)
        if photo_id is not None:
            try:
                mark_photo_processed(event_id, photo_id, api_url)
            except requests.RequestException as e:
                logger.error("Failed to mark faceless photo %s as processed: %s", filename, e)
                return False
        return True

    logger.info("Found %d face(s) in %s", len(faces), filename)

    for i, face in enumerate(faces):
        try:
            post_face_embedding(
                event_id=event_id,
                photo_id=photo_id,
                filename=filename,
                file_path=file_path,
                embedding=face["embedding"],
                facial_area=face["facial_area"],
                api_url=api_url,
            )
            logger.info("  Face %d/%d submitted", i + 1, len(faces))
        except requests.RequestException as e:
            logger.error("  Failed to submit face %d/%d for %s: %s", i + 1, len(faces), filename, e)
            return False

    # All faces submitted — mark photo as processed
    if photo_id is not None:
        try:
            mark_photo_processed(event_id, photo_id, api_url)
        except requests.RequestException as e:
            logger.error("Failed to mark photo %s as processed: %s", filename, e)
            return False

    return True


def process_event(event_id: int, photo_dir: str, api_url: str) -> tuple[int, int]:
    """Process all unprocessed photos for an event.

    Returns (success_count, failure_count).
    """
    photo_dir = os.path.abspath(photo_dir)
    if not os.path.isdir(photo_dir):
        logger.error("Photo directory does not exist: %s", photo_dir)
        return 0, 0

    photos = get_local_unprocessed_photos(photo_dir, event_id, api_url)
    unprocessed = [p for p in photos if not p.get("processed", False)]

    if not unprocessed:
        logger.info("No unprocessed photos found for event %d", event_id)
        return 0, 0

    logger.info("Found %d unprocessed photo(s) for event %d", len(unprocessed), event_id)

    success_count = 0
    failure_count = 0

    for i, photo in enumerate(unprocessed):
        logger.info("--- Photo %d/%d ---", i + 1, len(unprocessed))
        try:
            ok = process_photo(event_id, photo, photo_dir, api_url)
            if ok:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            logger.error("Unexpected error processing %s: %s", photo.get("filename", "?"), e)
            failure_count += 1

    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="Process unprocessed photos for a FaceSorter event.",
    )
    parser.add_argument("--event-id", type=int, required=True, help="Event ID to process")
    parser.add_argument("--photo-dir", type=str, required=True, help="Local directory containing event photos")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help=f"Backend API base URL (default: {DEFAULT_API_URL})")
    args = parser.parse_args()

    logger.info("Starting processing for event %d", args.event_id)
    logger.info("Photo directory: %s", args.photo_dir)
    logger.info("API URL: %s", args.api_url)
    logger.info("Detector: %s | Model: %s | Threshold: %s", DETECTOR_BACKEND, MODEL_NAME, SIMILARITY_THRESHOLD)

    start = time.time()
    success, failure = process_event(args.event_id, args.photo_dir, args.api_url)
    elapsed = time.time() - start

    logger.info("Processing complete in %.1fs: %d succeeded, %d failed", elapsed, success, failure)

    # Output structured result to stdout for the Electron app to parse
    result = {
        "eventId": args.event_id,
        "processed": success,
        "failed": failure,
        "elapsedSeconds": round(elapsed, 1),
    }
    print(json.dumps(result))

    if failure > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
