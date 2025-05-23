import dlib
import cv2
import os
import numpy as np
import json
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from db import insert_face, insert_analysis_result

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'model/shape_predictor_68_face_landmarks.dat')

# Normalize landmarks
def normalize_landmarks(landmarks, face_width, face_height):
    return [(x / face_width, y / face_height) for x, y in landmarks]

# Calculate Euclidean distance
def calculate_landmark_distance(landmarks1, landmarks2):
    return sum(euclidean((x1, y1), (x2, y2)) for (x1, y1), (x2, y2) in zip(landmarks1, landmarks2))

# Main function to call
def run_deepfake_detection(folder_path, frame_ids):
    image_extensions = ('.jpg', '.jpeg', '.png')
    landmarks_list = []
    image_files = []
    frame_id_mapping = {}

    all_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)])
    
    # Map each image file to its frame_id
    for i, file in enumerate(all_files):
        frame_id_mapping[file] = frame_ids[i]

    total_faces_detected = 0

    for file in all_files:
        path = os.path.join(folder_path, file)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if not faces:
            print(f"{file} - Face: NO")
            continue

        shape = predictor(gray, faces[0])
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        face_width = faces[0].width()
        face_height = faces[0].height()
        normalized_coords = normalize_landmarks(coords, face_width, face_height)

        landmarks_list.append(normalized_coords)
        image_files.append(file)

        # Insert face into database
        frame_id = frame_id_mapping[file]
        face_rect = faces[0]
        x, y, width, height = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        confidence = 1.0  # dlib doesn't provide confidence score, assuming 1.0
        face_id = insert_face(frame_id, x, y, width, height, confidence)
        # (We will insert analysis result later after clustering)

        print(f"{file} - Face: YES | Landmarks Detected: {len(coords)}")
        total_faces_detected += 1

    if len(landmarks_list) < 2:
        return {
            "total_images": len(all_files),
            "genuine_count": 0,
            "deepfake_count": 0,
            "genuine_percentage": 0,
            "deepfake_percentage": 0,
            "video_classification": "Insufficient Data"
        }, total_faces_detected, {}

    def compare_landmark_consistency(landmarks_list):
        consistency_scores = []
        for i in range(len(landmarks_list)):
            for j in range(i + 1, len(landmarks_list)):
                distance = calculate_landmark_distance(landmarks_list[i], landmarks_list[j])
                consistency_scores.append(distance)
                print(f"Comparing {image_files[i]} and {image_files[j]} - Distance: {distance:.2f}")
        return np.array(consistency_scores)

    consistency_scores = compare_landmark_consistency(landmarks_list)

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(consistency_scores.reshape(-1, 1))
    labels = kmeans.labels_

    deepfake_label = 1 if np.mean(consistency_scores[labels == 1]) > np.mean(consistency_scores[labels == 0]) else 0

    result = []
    genuine_count = 0
    deepfake_count = 0

    # Insert analysis results now
    for i, file in enumerate(image_files):
        label = labels[i]
        status = "deepfake" if label == deepfake_label else "genuine"
        if status == "genuine":
            probability = 0.1  # low probability for genuine
            genuine_count += 1
        else:
            probability = 0.9  # high probability for deepfake
            deepfake_count += 1

        frame_id = frame_id_mapping[file]
        face_id = insert_face(frame_id, 0, 0, 0, 0, 1.0)  # dummy face to insert analysis, since face already added above
        insert_analysis_result(face_id, probability)

        result.append(f"{file} - {status} (Cluster: {label})")

    print("\n--- Classification Results ---")
    for res in result:
        print(res)

    total_images = len(all_files)
    genuine_percentage = (genuine_count / total_images) * 100
    deepfake_percentage = (deepfake_count / total_images) * 100

    print("\n--- Overall Classification ---")
    print(f"Total Images Processed: {total_images}")
    print(f"Genuine Images: {genuine_count} ({genuine_percentage:.2f}%)")
    print(f"Deepfake Images: {deepfake_count} ({deepfake_percentage:.2f}%)")

    video_classification = "genuine" if genuine_percentage > 50 else "deepfake"
    print(f"\nOverall Video Classification: {video_classification}")

    summary = {
        "total_images": total_images,
        "genuine_count": genuine_count,
        "deepfake_count": deepfake_count,
        "genuine_percentage": genuine_percentage,
        "deepfake_percentage": deepfake_percentage,
        "video_classification": video_classification
    }

    frame_statuses = dict(zip(image_files, ["genuine" if labels[i] != deepfake_label else "deepfake" for i in range(len(image_files))]))

    # Save frame statuses to results folder
    video_name = os.path.basename(folder_path)
    os.makedirs('results', exist_ok=True)
    result_path = os.path.join('results', f'{video_name}_frame_status.json')
    with open(result_path, 'w') as f:
        json.dump(frame_statuses, f)

    return summary, total_faces_detected, frame_statuses
