import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import pandas as pd
from datetime import datetime
from keras_facenet import FaceNet
from emotion_predict import predict_emotion

# Load FaceNet embedder
embedder = FaceNet()

# Load trained classifier
with open("models/face_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

# Attendance CSV
attendance_file = "attendance.csv"
students = ["Lakshya", "Karan"]


# Create file if not exists
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time", "Emotion", "Status"])
    df.to_csv(attendance_file, index=False)

# Face detection
mp_face = mp.solutions.face_detection
detect = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Generate FaceNet embedding
def get_embedding(img):
    img = cv2.resize(img, (160, 160))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return embedder.embeddings(img)[0]

# Mark attendance once only
# def mark_attendance(name):
#     df = pd.read_csv(attendance_file)

#     if name in df["Name"].values:
#         return  # already marked

#     now = datetime.now().strftime("%H:%M:%S")
#     df.loc[len(df)] = [name, now]
#     df.to_csv(attendance_file, index=False)
#     print(f"Attendance marked for {name}")

def mark_attendance(name, emotion):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Emotion", "Status"])
    else:
        df = pd.read_csv(attendance_file)

    # If already marked present today → skip
    if ((df["Name"] == name) & (df["Date"] == today)).any():
        return

    # Mark everyone absent first (only once per day)
    for student in students:
        if not ((df["Name"] == student) & (df["Date"] == today)).any():
            df.loc[len(df)] = [student, today, "", "", "Absent"]

    # Update detected student to Present
    df.loc[(df["Name"] == name) & (df["Date"] == today),
           ["Time", "Emotion", "Status"]] = [time_now, emotion, "Present"]

    df.to_csv(attendance_file, index=False)

    print(f"✔ {name} marked PRESENT at {time_now} ({emotion})")

def show_popup(name, emotion):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Attendance Marked",
        f"✅ Attendance Marked Successfully!\n\n"
        f"Name: {name}\n"
        f"Emotion: {emotion}\n"
        f"Status: Present"
    )
    root.destroy()


# Start webcam
# def run_attendance():
#     cap = cv2.VideoCapture(0)
    
#     attendance_done = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         h, w, _ = frame.shape
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         detections = detect.process(rgb).detections

#         if detections:
#             for det in detections:
#             # extract bounding box
#                 box = det.location_data.relative_bounding_box
#                 x = int(box.xmin * w)
#                 y = int(box.ymin * h)
#                 w_box = int(box.width * w)
#                 h_box = int(box.height * h)

#                 x, y = max(0, x), max(0, y)

#                 face = frame[y:y+h_box, x:x+w_box]
#                 emotion = predict_emotion(face)


#                 if face.size == 0:
#                     continue

#                 embedding = get_embedding(face)
#                 embedding = embedding.reshape(1, -1)

#             # Predict person
#                 pred = classifier.predict(embedding)[0]
#                 prob = classifier.predict_proba(embedding).max()

#                 if prob > 0.70 and not attendance_done:  # threshold
#                     name = pred
#                     emotion = predict_emotion(face)
#                     attendance_done = True
#                     mark_attendance(name, emotion)
#                     show_popup(name, emotion)
                    
#                     cap.release()
#                     cv2.destroyAllWindows()
#                     return

#                 else:
#                     name = "Unknown"
#                     label = "Unknown"

#             # Draw on camera
#                 cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x, y-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         cv2.imshow("Attendance Camera", frame)

#         if cv2.waitKey(1) == ord('q'):
#             break

def run_attendance():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    attendance_done = False
    detected_name = "Unknown"
    detected_emotion = ""
    detected_prob = 0.0
    close_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detect.process(rgb).detections

        if detections:
            for det in detections:
                box = det.location_data.relative_bounding_box
                x = int(box.xmin * w)
                y = int(box.ymin * h)
                w_box = int(box.width * w)
                h_box = int(box.height * h)

                x, y = max(0, x), max(0, y)
                face = frame[y:y+h_box, x:x+w_box]

                if face.size == 0:
                    continue

                embedding = get_embedding(face).reshape(1, -1)

                pred = classifier.predict(embedding)[0]
                prob = classifier.predict_proba(embedding).max()

                if prob > 0.70:
                    detected_name = pred
                    detected_prob = prob
                    detected_emotion = predict_emotion(face)

                    if not attendance_done:
                        attendance_done = True
                        mark_attendance(detected_name, detected_emotion)
                        show_popup(detected_name, detected_emotion)

                else:
                    detected_name = "Unknown"
                    detected_emotion = ""
                    detected_prob = 0.0

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)

        # Show info on camera
        if detected_name != "Unknown":
            text = f"{detected_name} | {detected_emotion} | {detected_prob*100:.1f}%"
        else:
            text = "Scanning..."

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Attendance Camera", frame)

        # Close camera 60 frames (~2 seconds) after marking
        if attendance_done:
            close_counter += 1
            if close_counter > 60:
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    run_attendance()



# cap = cv2.VideoCapture(0)

# print("Starting camera...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     detections = detect.process(rgb).detections

#     if detections:
#         for det in detections:
#             # extract bounding box
#             box = det.location_data.relative_bounding_box
#             x = int(box.xmin * w)
#             y = int(box.ymin * h)
#             w_box = int(box.width * w)
#             h_box = int(box.height * h)

#             x, y = max(0, x), max(0, y)

#             face = frame[y:y+h_box, x:x+w_box]

#             if face.size == 0:
#                 continue

#             embedding = get_embedding(face)
#             embedding = embedding.reshape(1, -1)

#             # Predict person
#             pred = classifier.predict(embedding)[0]
#             prob = classifier.predict_proba(embedding).max()

#             if prob > 0.70:  # threshold
#                 name = pred
#                 mark_attendance(name)
#                 label = f"{name} ({prob*100:.1f}%)"
#             else:
#                 label = "Unknown"

#             # Draw on camera
#             cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     cv2.imshow("Attendance Camera", frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()