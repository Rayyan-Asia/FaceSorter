"""
Extract a face embedding from a single image.

Usage:
    python extract_embedding.py --image /path/to/photo.jpg

Outputs the embedding as a JSON array to stdout.
Spawned by the Electron app for walk-in customer face matching.
"""

import argparse
import json
import sys

import cv2


HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)


def main():
    parser = argparse.ArgumentParser(description='Extract face embedding from an image')
    parser.add_argument('--image', required=True, help='Path to the image file')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(json.dumps({'error': 'Could not read image'}))
        sys.exit(1)

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

    if len(faces) == 0:
        print(json.dumps({'error': 'No face detected in image'}))
        sys.exit(1)

    # Use the largest detected face
    areas = [w * h for (x, y, w, h) in faces]
    best_idx = areas.index(max(areas))
    x, y, w, h = faces[best_idx]

    face_crop = img[y:y + h, x:x + w]

    import os
    import tempfile
    temp_path = os.path.join(tempfile.gettempdir(), f'_facesorter_extract_{os.getpid()}.jpg')
    cv2.imwrite(temp_path, face_crop)

    try:
        from deepface import DeepFace
        result = DeepFace.represent(
            img_path=temp_path,
            model_name='Facenet',
            enforce_detection=False,
            detector_backend='skip'
        )
        os.unlink(temp_path)

        if result and len(result) > 0:
            print(json.dumps(result[0]['embedding']))
        else:
            print(json.dumps({'error': 'Embedding extraction returned empty result'}))
            sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
