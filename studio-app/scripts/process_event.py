"""
Process event photos: detect faces, extract embeddings, push results to backend API.

Usage:
    python process_event.py --event-id 1 --photos-dir /path/to/photos --api-url http://localhost:8080/api

This script is spawned by the Electron app as a child process.
It reads unprocessed photos from the local filesystem, runs face detection
and embedding extraction, and POSTs the results to the backend.
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

import cv2
import numpy as np

SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)


def get_unprocessed_photos(api_url, event_id):
    """Fetch list of unprocessed photo records from the backend."""
    url = f"{api_url}/events/{event_id}/photos?processed=false"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"Warning: Could not fetch photo list from API: {e}", file=sys.stderr)
        return None


def list_local_photos(photos_dir):
    """List all image files in the local photos directory."""
    photos = []
    for fname in sorted(os.listdir(photos_dir)):
        if fname.lower().endswith(SUPPORTED_FORMATS):
            photos.append(fname)
    return photos


def detect_faces(image_path):
    """Detect faces in an image and return list of (x, y, w, h) bounding boxes."""
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
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return faces if len(faces) > 0 else [], img


def extract_embedding(img, x, y, w, h):
    """Extract a FaceNet embedding for a detected face region."""
    try:
        from deepface import DeepFace

        face_crop = img[y:y + h, x:x + w]

        temp_path = os.path.join(os.environ.get('TMPDIR', '/tmp'),
                                 f'_facesorter_crop_{os.getpid()}.jpg')
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


def post_embedding(api_url, event_id, photo_filename, embedding, bbox):
    """POST a face embedding to the backend API."""
    url = f"{api_url}/events/{event_id}/embeddings"
    payload = json.dumps({
        'photoFilename': photo_filename,
        'embedding': embedding,
        'boundingBox': {
            'x': int(bbox[0]),
            'y': int(bbox[1]),
            'width': int(bbox[2]),
            'height': int(bbox[3]),
        },
    }).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"  API error posting embedding: {e}", file=sys.stderr)
        return None


def mark_photo_processed(api_url, event_id, photo_filename):
    """Mark a photo as processed in the backend."""
    url = f"{api_url}/events/{event_id}/photos/{photo_filename}/processed"
    req = urllib.request.Request(url, method='PUT',
                                headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as resp:
            return True
    except urllib.error.URLError as e:
        print(f"  API error marking processed: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Process event photos for FaceSorter')
    parser.add_argument('--event-id', required=True, help='Event ID')
    parser.add_argument('--photos-dir', required=True, help='Local directory containing photos')
    parser.add_argument('--api-url', required=True, help='Backend API base URL')
    args = parser.parse_args()

    if not os.path.isdir(args.photos_dir):
        print(f"Error: Photos directory does not exist: {args.photos_dir}", file=sys.stderr)
        sys.exit(1)

    photo_files = list_local_photos(args.photos_dir)
    if not photo_files:
        print(f"No image files found in {args.photos_dir}")
        sys.exit(0)

    # Try to get unprocessed list from API; if unavailable, process all local photos
    api_photos = get_unprocessed_photos(args.api_url, args.event_id)
    if api_photos is not None:
        unprocessed_names = {p.get('filename', p.get('name', '')) for p in api_photos}
        photo_files = [f for f in photo_files if f in unprocessed_names]

    total = len(photo_files)
    print(f"Processing {total} photos for event {args.event_id}...")

    processed_count = 0
    face_count = 0

    for i, filename in enumerate(photo_files):
        filepath = os.path.join(args.photos_dir, filename)
        print(f"[{i + 1}/{total}] {filename}")

        faces, img = detect_faces(filepath)

        if img is None:
            print(f"  Skipped: could not read image")
            continue

        if len(faces) == 0:
            print(f"  No faces detected")
        else:
            for j, (x, y, w, h) in enumerate(faces):
                embedding = extract_embedding(img, x, y, w, h)
                if embedding is not None:
                    result = post_embedding(args.api_url, args.event_id,
                                            filename, embedding, (x, y, w, h))
                    if result:
                        face_count += 1
                        print(f"  Face {j + 1}: embedding posted")
                    else:
                        print(f"  Face {j + 1}: embedding extracted (API unavailable)")
                else:
                    print(f"  Face {j + 1}: embedding extraction failed")

        mark_photo_processed(args.api_url, args.event_id, filename)
        processed_count += 1

    print(f"\nDone. Processed {processed_count}/{total} photos, {face_count} faces embedded.")


if __name__ == '__main__':
    main()
