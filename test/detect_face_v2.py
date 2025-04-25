import cv2
import os

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get folder path from user
folder_path = input("Enter the folder path containing images: ")

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# Loop through all image files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"{filename} - ERROR loading image")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            print(f"{filename} - YES")
        else:
            print(f"{filename} - NO")
