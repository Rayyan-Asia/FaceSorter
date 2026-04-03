"""
Extract the primary face embedding from a single image using InsightFace (ArcFace).

Usage:
    python extract_embedding.py --image <image_path>

Outputs a JSON object to stdout:
    {"embedding": [float, ...], "model": "ArcFace"}

Exit codes:
    0 — success
    1 — no face detected or error
"""

import argparse
import sys
import json
import logging
import os

import gpu_config  # configures ONNX Runtime device before insightface loads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

MODEL_NAME = "ArcFace"


def get_face_app(providers):
    import insightface
    from insightface.app import FaceAnalysis

    # buffalo_l: RetinaFace detector + ArcFace recognition (512-dim embeddings)
    app = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def extract_embedding(image_path: str, app) -> list[float]:
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)
    if not faces:
        raise ValueError("No face detected in image")

    # Pick the largest face (highest area) as the primary subject
    primary = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return primary.embedding.tolist()


def main():
    device_info = gpu_config.configure(logger)
    providers = device_info["providers"]

    parser = argparse.ArgumentParser(
        description="Extract an ArcFace embedding from a single image.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    try:
        app = get_face_app(providers)
        embedding = extract_embedding(args.image, app)
    except Exception as e:
        logger.error("Failed to extract embedding from %s: %s", args.image, e)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    print(json.dumps({"embedding": embedding, "model": MODEL_NAME}))


if __name__ == "__main__":
    main()
