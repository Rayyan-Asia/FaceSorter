import os
import sys
import json
import base64
from deepface import DeepFace
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

# Input image path
img_path = 'static/photos/WhatsApp Image 2024-11-22 at 4.26.20 PM(3).jpeg'
output_dir = 'static/faces'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the image
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
cropped_faces = []

for idx, (x, y, w, h) in enumerate(faces):
    eyes_count = 0
    img_crop = img[y:y + h, x:x + w]
    #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect eyes within the face region
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eyes_count += 1

    # Save the cropped face if it has two or more detected eyes
    if eyes_count >= 2:
        # Create a valid file name for the cropped face
        filename = f"crop_{idx+1}.jpg"
        path = os.path.join(output_dir, filename)

        # Save the cropped face to the output directory
        cropped_faces.append(path)
        cv2.imwrite(path, img_crop)
        print(f"Saved cropped face to {path}")

        # Optional: Display the cropped face
        cv2.imshow('Cropped Face', img_crop)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cv2.destroyAllWindows()
