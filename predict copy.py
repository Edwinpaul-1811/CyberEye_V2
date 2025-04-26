import dlib
import cv2
import os
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'model/shape_predictor_68_face_landmarks.dat')

# Helper function to normalize the landmarks (scale the coordinates)
def normalize_landmarks(landmarks, face_width, face_height):
    normalized = [(x / face_width, y / face_height) for x, y in landmarks]
    return normalized

# Helper function to calculate distance between two sets of landmarks
def calculate_landmark_distance(landmarks1, landmarks2):
    # Calculate Euclidean distance between corresponding points
    total_distance = 0
    for (x1, y1), (x2, y2) in zip(landmarks1, landmarks2):
        total_distance += euclidean((x1, y1), (x2, y2))
    return total_distance

# Get folder path from user
folder_path = input("Enter the folder path: ")
image_extensions = ('.jpg', '.jpeg', '.png')

# Store landmarks for comparison
landmarks_list = []
image_files = []

# Process each image
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

        # Normalize landmarks based on the size of the face
        face_width = faces[0].width()
        face_height = faces[0].height()
        normalized_coords = normalize_landmarks(coords, face_width, face_height)

        # Store normalized landmarks and corresponding image filename
        landmarks_list.append(normalized_coords)
        image_files.append(file)
        print(f"{file} - Face: YES | Landmarks Detected: {len(coords)}")

# Compare the landmarks between pairs of images
def compare_landmark_consistency(landmarks_list):
    consistency_scores = []

    # Calculate distances between each pair of images
    for i in range(len(landmarks_list)):
        for j in range(i + 1, len(landmarks_list)):
            distance = calculate_landmark_distance(landmarks_list[i], landmarks_list[j])
            consistency_scores.append(distance)
            print(f"Comparing {image_files[i]} and {image_files[j]} - Distance: {distance:.2f}")

    return np.array(consistency_scores)

# Run comparison on collected landmarks
consistency_scores = compare_landmark_consistency(landmarks_list)

# Apply K-Means clustering on the distances
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(consistency_scores.reshape(-1, 1))

# Get cluster labels: 0 for "real", 1 for "fake" or vice versa
labels = kmeans.labels_

# Determine the label for deepfake (higher distances likely indicate deepfake)
deepfake_label = 1 if np.mean(consistency_scores[labels == 1]) > np.mean(consistency_scores[labels == 0]) else 0

# Store the final classification results
result = []

# Output the results and collect stats
genuine_count = 0
deepfake_count = 0

for i, file in enumerate(image_files):
    label = labels[i]
    status = "deepfake" if label == deepfake_label else "genuine"
    result.append(f"{file} - {status} (Cluster: {label})")
    
    if status == "genuine":
        genuine_count += 1
    else:
        deepfake_count += 1

# Display the results
print("\n--- Classification Results ---")
for res in result:
    print(res)

# Display the overall result
total_images = len(image_files)
genuine_percentage = (genuine_count / total_images) * 100
deepfake_percentage = (deepfake_count / total_images) * 100

print("\n--- Overall Classification ---")
print(f"Total Images Processed: {total_images}")
print(f"Genuine Images: {genuine_count} ({genuine_percentage:.2f}%)")
print(f"Deepfake Images: {deepfake_count} ({deepfake_percentage:.2f}%)")

# Classify the overall video based on the percentage of genuine vs deepfake images
if genuine_percentage > 50:
    video_classification = "genuine"
else:
    video_classification = "deepfake"

print(f"\nOverall Video Classification: {video_classification}")

def classify_video_frames(folder_path):
    # ... [your existing code above, excluding input() line] ...
    
    return {
        "total_images": total_images,
        "genuine_count": genuine_count,
        "genuine_percentage": genuine_percentage,
        "deepfake_count": deepfake_count,
        "deepfake_percentage": deepfake_percentage,
        "video_classification": video_classification
    }
