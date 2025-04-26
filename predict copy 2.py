import dlib
import cv2
import os
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

def normalize_landmarks(landmarks, face_width, face_height):
    return [(x / face_width, y / face_height) for x, y in landmarks]

def calculate_landmark_distance(landmarks1, landmarks2):
    return sum(euclidean((x1, y1), (x2, y2)) for (x1, y1), (x2, y2) in zip(landmarks1, landmarks2))

def run_deepfake_detection(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png')
    landmarks_list = []
    image_files = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(image_extensions):
            path = os.path.join(folder_path, file)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if not faces:
                continue
            shape = predictor(gray, faces[0])
            coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            normalized_coords = normalize_landmarks(coords, faces[0].width(), faces[0].height())
            landmarks_list.append(normalized_coords)
            image_files.append(file)

    if len(landmarks_list) < 2:
        return {
            "total_images": len(image_files),
            "genuine_count": 0,
            "deepfake_count": 0,
            "genuine_percentage": 0,
            "deepfake_percentage": 0,
            "video_classification": "Insufficient Data"
        }

    consistency_scores = []
    for i in range(len(landmarks_list)):
        for j in range(i + 1, len(landmarks_list)):
            distance = calculate_landmark_distance(landmarks_list[i], landmarks_list[j])
            consistency_scores.append(distance)

    consistency_scores = np.array(consistency_scores)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(consistency_scores.reshape(-1, 1))
    labels = kmeans.labels_
    deepfake_label = 1 if np.mean(consistency_scores[labels == 1]) > np.mean(consistency_scores[labels == 0]) else 0

    result_labels = []
    for landmarks in landmarks_list:
        distances = [calculate_landmark_distance(landmarks, other) for other in landmarks_list if landmarks != other]
        avg_distance = np.mean(distances)
        label = kmeans.predict([[avg_distance]])[0]
        result_labels.append(label)

    genuine_count = result_labels.count(1 - deepfake_label)
    deepfake_count = result_labels.count(deepfake_label)
    total_images = len(result_labels)
    genuine_percentage = (genuine_count / total_images) * 100
    deepfake_percentage = (deepfake_count / total_images) * 100

    video_classification = "genuine" if genuine_percentage > 60 else "deepfake"

    return {
        "total_images": total_images,
        "genuine_count": genuine_count,
        "deepfake_count": deepfake_count,
        "genuine_percentage": genuine_percentage,
        "deepfake_percentage": deepfake_percentage,
        "video_classification": video_classification
    }
