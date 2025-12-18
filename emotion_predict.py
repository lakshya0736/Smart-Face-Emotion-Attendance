import cv2
import numpy as np
# from tensorflow.keras.models import load_model
from tensorflow import keras
load_model = keras.models.load_model


emotion_labels = [
    "Angry","Disgust","Fear",
    "Happy","Sad","Surprise","Neutral"
]

emotion_model = load_model("models/emotion_model.h5")

def predict_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray/255.0
    gray = gray.reshape(1, 48, 48, 1)

    preds = emotion_model.predict(gray, verbose=0)
    return emotion_labels[np.argmax(preds)]