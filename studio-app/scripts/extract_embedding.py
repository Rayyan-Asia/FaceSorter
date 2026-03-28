"""
Extract the primary face embedding from a single image.

Usage:
    python extract_embedding.py --image <image_path>

Outputs a JSON object to stdout:
    {"embedding": [float, ...], "model": "Facenet"}

Exit codes:
    0 — success
    1 — no face detected or error
"""

import argparse
import sys
import json
import logging

import cv2
import numpy as np
from deepface import DeepFace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

MODEL_NAME = "Facenet"

# RetinaFace is preferred for production; fall back to opencv if unavailable.
try:
    from retinaface import RetinaFace as _rf
    DETECTOR_BACKEND = "retinaface"
    logger.info("Using RetinaFace detector")
except ImportError:
    DETECTOR_BACKEND = "opencv"
    logger.info("RetinaFace not available, falling back to OpenCV detector")


def extract_embedding(image_path: str) -> list[float]:
    """Detect the primary face in an image and return its FaceNet embedding.

    Uses enforce_detection=True so that images with no detectable face raise
    an error rather than silently returning a garbage embedding.
    """
    representations = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
    )

    if not representations:
        raise ValueError("DeepFace returned no representations")

    # Pick the face with the highest confidence (largest facial area as proxy).
    primary = max(representations, key=lambda r: r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0))
    return primary["embedding"]


def main():
    parser = argparse.ArgumentParser(
        description="Extract a FaceNet embedding from a single image.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    try:
        embedding = extract_embedding(args.image)
    except Exception as e:
        logger.error("Failed to extract embedding from %s: %s", args.image, e)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    result = {
        "embedding": embedding,
        "model": MODEL_NAME,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
