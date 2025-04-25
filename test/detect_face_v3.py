import dlib
import cv2
import os

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'model/shape_predictor_68_face_landmarks.dat')  # Download link below

# Ask for folder
folder_path = input("Enter the folder path: ")
image_extensions = ('.jpg', '.jpeg', '.png')

# Loop through images
for file in os.listdir(folder_path):
    if file.lower().endswith(image_extensions):
        path = os.path.join(folder_path, file)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        if not faces:
            print(f"{file} - Face: NO")
            continue

        shape = predictor(gray, faces[0])  # first face
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        print(f"{file} - Face: YES | Landmarks Detected: {len(coords)}")
