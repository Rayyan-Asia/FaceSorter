import os
import sys
import json
from deepface import DeepFace
import cv2
from scipy.spatial.distance import cosine
import hashlib

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')


def prepare_database(directory):
    """
    Ensure the database directory contains all images.
    DeepFace will use this as the reference database for comparison.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The directory '{directory}' does not exist.")

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(supported_formats)]

    if not image_paths:
        raise ValueError(f"No supported image files found in directory '{directory}'.")

    return image_paths


def is_embedding_similar(existing_embeddings, new_embedding, threshold=0.4):
    """
    Check if a face embedding is similar to any existing embeddings.
    Args:
        existing_embeddings (list): List of known embeddings.
        new_embedding (list): The new embedding to compare.
        threshold (float): Distance threshold for similarity.
    Returns:
        bool: True if similar embedding exists, otherwise False.
    """
    for embedding in existing_embeddings:
        # Calculate cosine distance
        distance = cosine(embedding, new_embedding)
        if distance < threshold:
            return True  # Image is already processed
    return False

def save_cropped_faces(input_dir = '/home/datablock/PycharmProjects/test-face-sort/static/photos', output_dir="/home/datablock/PycharmProjects/test-face-sort/static/faces"):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    cropped_faces = []
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(supported_formats):  # Check if the file is an image
            img_path = os.path.join(input_dir, file_name)
            img = cv2.imread(img_path)

            # Resize the image if it's too large
            height = img.shape[0]
            width = img.shape[1]
            size = height * width

            if size > (500 ** 2):
                r = 500.0 / img.shape[1]
                dim = (500, int(img.shape[0] * r))
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )

            print(f"Faces detected: {faces}")

            for idx, (x, y, w, h) in enumerate(faces):
                eyes_count = 0
                img_crop = img[y:y + h, x:x + w]
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Detect eyes within the face region
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                for (ex, ey, ew, eh) in eyes:
                    # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    eyes_count += 1

                # Create a valid file name for the cropped face
                filename = f"crop_{idx + 1}_{os.path.splitext(os.path.basename(img_path))[0]}{os.path.splitext(os.path.basename(img_path))[1]}"
                path = os.path.join(output_dir, filename)

                # Save the cropped face to the output directory
                cropped_faces.append(path)
                cv2.imwrite(path, img_crop)

    return cropped_faces


def batch_images_with_find(image_paths, model_name="Facenet", distance_metric="cosine", threshold=None):
    """
    Use DeepFace `find` to batch similar images, avoiding reprocessing already matched faces.
    Args:
        image_paths: List of image paths to process.
        model_name: DeepFace model to use (default: "Facenet").
        distance_metric: Metric for similarity (default: "cosine").
        threshold: Custom threshold for similarity (default: use model defaults).
    Returns:
        A JSON-compatible map of face embeddings and their matched image batches.
    """
    processed_embeddings = []
    face_batches = []

    for img_path in image_paths:
        try:
            # Generate embedding for the current image
            representation = DeepFace.represent(img_path, model_name=model_name, enforce_detection=True)
            print('Representation: ' + str(representation))
            current_embedding = representation[0]['embedding']

            # Check if this embedding already exists in the map
            if is_embedding_similar(processed_embeddings, current_embedding, threshold=0.4):
                continue

            # Use `DeepFace.find` to find matching images in the database
            matches = DeepFace.find(
                img_path=img_path,
                db_path=os.path.dirname(img_path),
                model_name=model_name,
                distance_metric=distance_metric,
                threshold=threshold,
                silent=True,
                refresh_database=True
            )

            # Extract matched images from the result
            matched_images = matches[0]["identity"].tolist() if len(matches) > 0 else []
            matched_images_orig = []
            matched_images_cropped = []

            for img in matched_images:
                parts = img.split("_", 2)
                path = parts[-1] if len(parts) > 2 else parts[-1]
                orig_path = './photos/' + path
                crop_path = f'./faces/{os.path.splitext(os.path.basename(img))[0]}{os.path.splitext(os.path.basename(img))[1]}'
                matched_images_orig.append(orig_path)
                matched_images_cropped.append(crop_path)


            # Save the embedding and its matched batch
            processed_embeddings.append(current_embedding)

            embedding_str = ','.join(map(str, current_embedding))

            hash_object = hashlib.md5(embedding_str.encode())
            unique_number = int(hash_object.hexdigest(), 16)

            face_batches.append({
                "face_embedding_hash": unique_number,
                "batch": matched_images_orig,
                "crop_batch": matched_images_cropped
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return face_batches


def main():
    image_directory = "/home/datablock/PycharmProjects/test-face-sort/static/photos"

    try:
        image_paths = save_cropped_faces(image_directory)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print("Batching similar images using DeepFace `find`...")
    results = batch_images_with_find(image_paths)

    if not results:
        print("No similar face batches were found.")
        sys.exit(0)

    output_file = "face_batches_find.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
