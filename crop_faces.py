import cv2
import mediapipe as mp
import os

input_folder = r"data/raw_images"
output_folder = r"data/cropped_faces"
os.makedirs(output_folder, exist_ok=True)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def crop_face(image, detections):
    h, w, _ = image.shape
    for det in detections:
        box = det.location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        w_box = int(box.width * w)
        h_box = int(box.height * h)
        return image[y:y+h_box, x:x+w_box]  # first face only
    return None

for student in os.listdir(input_folder):
    student_in = os.path.join(input_folder, student)
    student_out = os.path.join(output_folder, student)
    os.makedirs(student_out, exist_ok=True)

    for img in os.listdir(student_in):
        img_path = os.path.join(student_in, img)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = face_detector.process(rgb).detections

        if detections:
            face = crop_face(image, detections)
            if face is not None:
                save_path = os.path.join(student_out, img)
                cv2.imwrite(save_path, face)

print("All faces cropped successfully!")
