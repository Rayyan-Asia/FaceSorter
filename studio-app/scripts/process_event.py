"""
Process event photos: detect faces, extract embeddings, push results to backend API.

Usage:
    python process_event.py --event-id 1 --photos-dir /path/to/photos --api-url http://localhost:8080/api --token <jwt>

This script is spawned by the Electron app as a child process.
It fetches unprocessed photo records from the backend, runs face detection and
embedding extraction on each, and POSTs the results back.
"""

import argparse
import json
import os
import sys
import tempfile
import urllib.request
import urllib.error

import cv2
import numpy as np

SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)


def make_request(url, token, method='GET', payload=None):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
    }
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode().strip()
            return json.loads(body) if body else True
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} error on {method} {url}: {e.reason}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"  URL error on {method} {url}: {e}", file=sys.stderr)
        return None


def get_unprocessed_photos(api_url, event_id, token):
    """Fetch unprocessed photo records from the backend."""
    url = f"{api_url}/studio/events/{event_id}/photos/unprocessed"
    result = make_request(url, token)
    if result is None:
        print("Warning: Could not fetch photo list from API. Processing all local photos.", file=sys.stderr)
    return result


def detect_faces(image_path):
    """Detect faces in an image, return (faces, resized_img)."""
    img = cv2.imread(image_path)
    if img is None:
        return [], None

    height, width = img.shape[:2]
    max_dim = 800
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)),
                         interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )
    return (faces if len(faces) > 0 else []), img


def extract_embedding(img, x, y, w, h):
    """Extract a FaceNet embedding for a detected face region."""
    try:
        from deepface import DeepFace

        face_crop = img[y:y + h, x:x + w]
        tmp_dir = tempfile.gettempdir()
        temp_path = os.path.join(tmp_dir, f'_facesorter_crop_{os.getpid()}.jpg')
        cv2.imwrite(temp_path, face_crop)

        result = DeepFace.represent(
            img_path=temp_path,
            model_name='Facenet',
            enforce_detection=False,
            detector_backend='skip'
        )

        os.unlink(temp_path)

        if result and len(result) > 0:
            return result[0]['embedding']
    except Exception as e:
        print(f"  Embedding extraction error: {e}", file=sys.stderr)

    return None


def submit_embeddings(api_url, event_id, photo_id, embeddings, token):
    """POST face embeddings (or empty list) for a photo to mark it processed."""
    url = f"{api_url}/faces/embeddings"
    payload = {
        'photoId': photo_id,
        'eventId': int(event_id),
        'embeddings': embeddings,
    }
    return make_request(url, token, method='POST', payload=payload)


def list_local_photos(photos_dir):
    """List all image files in the local photos directory."""
    return sorted(
        f for f in os.listdir(photos_dir)
        if f.lower().endswith(SUPPORTED_FORMATS)
    )


def main():
    parser = argparse.ArgumentParser(description='Process event photos for FaceSorter')
    parser.add_argument('--event-id', required=True, help='Event ID')
    parser.add_argument('--photos-dir', required=True, help='Local directory containing photos')
    parser.add_argument('--api-url', required=True, help='Backend API base URL')
    parser.add_argument('--token', required=True, help='JWT auth token')
    args = parser.parse_args()

    if not os.path.isdir(args.photos_dir):
        print(f"Error: Photos directory does not exist: {args.photos_dir}", file=sys.stderr)
        sys.exit(1)

    # Fetch unprocessed photo records (contains id + filename)
    api_photos = get_unprocessed_photos(args.api_url, args.event_id, args.token)

    if api_photos is not None:
        # Use API list — keyed by filename for O(1) lookup
        photo_map = {p['filename']: p for p in api_photos}
        photo_files = [f for f in list_local_photos(args.photos_dir) if f in photo_map]
    else:
        # Fallback: process all local files, but we won't have photo IDs → skip API submission
        photo_map = {}
        photo_files = list_local_photos(args.photos_dir)

    total = len(photo_files)
    if total == 0:
        print("No unprocessed photos found.")
        sys.exit(0)

    print(f"Processing {total} photos for event {args.event_id}...")

    processed_count = 0
    face_count = 0

    for i, filename in enumerate(photo_files):
        filepath = os.path.join(args.photos_dir, filename)
        photo_record = photo_map.get(filename)
        photo_id = photo_record['id'] if photo_record else None

        print(f"[{i + 1}/{total}] {filename}")

        faces, img = detect_faces(filepath)

        if img is None:
            print(f"  Skipped: could not read image")
            continue

        embeddings = []

        if len(faces) == 0:
            print(f"  No faces detected")
        else:
            for j, (x, y, w, h) in enumerate(faces):
                embedding = extract_embedding(img, x, y, w, h)
                if embedding is not None:
                    embeddings.append(embedding)
                    face_count += 1
                    print(f"  Face {j + 1}: embedding extracted")
                else:
                    print(f"  Face {j + 1}: embedding extraction failed")

        # Submit embeddings (or empty list) to mark photo as processed
        if photo_id is not None:
            result = submit_embeddings(args.api_url, args.event_id, photo_id, embeddings, args.token)
            if result is not None or len(embeddings) == 0:
                processed_count += 1
        else:
            processed_count += 1

    print(f"\nDone. Processed {processed_count}/{total} photos, {face_count} faces embedded.")


if __name__ == '__main__':
    main()
