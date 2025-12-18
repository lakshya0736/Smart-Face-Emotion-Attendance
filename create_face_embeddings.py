import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.svm import SVC
import pickle

# Paths
cropped_faces_folder = r"data/cropped_faces"
embeddings_path = r"models/embeddings.pkl"
classifier_path = r"models/face_classifier.pkl"

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load FaceNet model
embedder = FaceNet()

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype("float32")
    face_img = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(face_img)
    return embedding[0]

X = []  # embeddings
y = []  # labels (names)

# Loop through student folders
for student_name in os.listdir(cropped_faces_folder):
    student_path = os.path.join(cropped_faces_folder, student_name)

    for img_file in os.listdir(student_path):
        img_path = os.path.join(student_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        embedding = get_embedding(img)
        X.append(embedding)
        y.append(student_name)

X = np.array(X)
y = np.array(y)

print("Total embeddings:", len(X))

# Train SVM classifier
classifier = SVC(kernel="linear", probability=True)
classifier.fit(X, y)

# Save embeddings + classifier
with open(embeddings_path, "wb") as f:
    pickle.dump((X, y), f)

with open(classifier_path, "wb") as f:
    pickle.dump(classifier, f)

print("Embeddings + Classifier saved successfully!")
