"""
Process event photos: detect faces, extract embeddings, push results to backend API.

Usage:
    python process_event.py --event-id 1 --photos-dir /path/to/photos --api-url http://localhost:8080/api --token <jwt>

This script is spawned by the Electron app as a child process.
It fetches unprocessed photo records from the backend, runs face detection and
embedding extraction on each, and POSTs the results back.

Uses InsightFace (ArcFace 512-dim) with ONNX Runtime DirectML for GPU acceleration.
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

import gpu_config  # configures ONNX Runtime device before insightface loads

SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


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
    url = f"{api_url}/studio/events/{event_id}/photos/unprocessed"
    result = make_request(url, token)
    if result is None:
        print("Warning: Could not fetch photo list from API.", file=sys.stderr)
    return result


def extract_embeddings_from_photo(image_path, app):
    """Detect all faces and extract ArcFace embeddings."""
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return []

    try:
        faces = app.get(img)
        return [f.embedding.tolist() for f in faces if f.embedding is not None]
    except Exception as e:
        print(f"  Embedding extraction error: {e}", file=sys.stderr)
        return []


def submit_embeddings(api_url, event_id, photo_id, embeddings, token):
    url = f"{api_url}/faces/embeddings"
    payload = {
        'photoId': photo_id,
        'eventId': int(event_id),
        'embeddings': embeddings,
    }
    return make_request(url, token, method='POST', payload=payload)


def list_local_photos(photos_dir):
    return sorted(
        f for f in os.listdir(photos_dir)
        if f.lower().endswith(SUPPORTED_FORMATS)
    )


def main():
    device_info = gpu_config.configure()
    providers = device_info["providers"]
    print(f"Compute device: {device_info['device']}"
          + (f" — {device_info['gpu_name']}" if device_info['gpu_name'] else "")
          + (f" ({device_info['vram_gb']} GB VRAM)" if device_info['vram_gb'] else ""))

    parser = argparse.ArgumentParser(description='Process event photos for FaceSorter')
    parser.add_argument('--event-id', required=True)
    parser.add_argument('--photos-dir', required=True)
    parser.add_argument('--api-url', required=True)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.photos_dir):
        print(f"Error: Photos directory does not exist: {args.photos_dir}", file=sys.stderr)
        sys.exit(1)

    # Load InsightFace model once, reuse across all photos
    print("Loading face recognition model...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model ready.")

    api_photos = get_unprocessed_photos(args.api_url, args.event_id, args.token)

    if api_photos is not None:
        photo_map = {p['filename']: p for p in api_photos}
        photo_files = [f for f in list_local_photos(args.photos_dir) if f in photo_map]
    else:
        photo_map = {}
        photo_files = list_local_photos(args.photos_dir)

    total = len(photo_files)
    if total == 0:
        print("No unprocessed photos found.")
        sys.exit(0)

    print(f"Processing {total} photos...")

    processed_count = 0
    face_count = 0

    for i, filename in enumerate(photo_files):
        filepath = os.path.join(args.photos_dir, filename)
        photo_record = photo_map.get(filename)
        photo_id = photo_record['id'] if photo_record else None

        print(f"[{i + 1}/{total}] {filename}")

        if not os.path.isfile(filepath):
            print(f"  Skipped: file not found")
            continue

        embeddings = extract_embeddings_from_photo(filepath, app)
        if not embeddings:
            print(f"  No faces detected")
        else:
            face_count += len(embeddings)
            print(f"  {len(embeddings)} face(s) embedded")

        if photo_id is not None:
            result = submit_embeddings(args.api_url, args.event_id, photo_id, embeddings, args.token)
            if result is not None or not embeddings:
                processed_count += 1
        else:
            processed_count += 1

    print(f"\nDone. Processed {processed_count}/{total} photos, {face_count} faces embedded.")


if __name__ == '__main__':
    main()
