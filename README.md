🎓 Smart Face Emotion-Based Attendance System

A real-time AI-powered attendance system that uses Face Recognition + Facial Emotion Detection to automatically mark attendance through a university-style web portal.

This project integrates Deep Learning, Computer Vision, and Web APIs to deliver a fully automated, contactless attendance solution.



🚀 Features

✔ Real-time face detection using MediaPipe
✔ Face recognition using FaceNet embeddings + SVM classifier
✔ Facial emotion recognition using CNN trained on FER-2013 dataset
✔ Automatic attendance marking with Date, Time, Emotion & Status
✔ Clean University Portal UI (HTML, CSS, JavaScript)
✔ Backend powered by FastAPI
✔ Camera opens only on user action
✔ Attendance stored in CSV (can be extended to DB)



🧠 System Architecture
User → Web Portal → FastAPI Backend
                 ↓
            Webcam Capture
                 ↓
           Face Detection
                 ↓
        Face Recognition (FaceNet)
                 ↓
       Emotion Detection (CNN)
                 ↓
       Attendance Marked (CSV)



🛠 Tech Stack
🔹 Backend

Python 3.11

FastAPI

OpenCV

MediaPipe

TensorFlow / Keras

FaceNet (keras-facenet)

Scikit-learn

Pandas

🔹 Frontend

HTML5

CSS3

JavaScript (Fetch API)

🔹 Deep Learning

FaceNet (128-D embeddings)

CNN (FER-2013 emotion dataset)



📸 How Attendance Works

1️⃣ User opens the web portal
2️⃣ Clicks “Mark Attendance”
3️⃣ Camera opens
4️⃣ System detects:

Face

Identity

Emotion
5️⃣ Attendance is marked as Present
6️⃣ Camera auto-closes
7️⃣ Popup confirms attendance

📊 Attendance Format (CSV)
Name	Date	Time	Emotion	Status
Lakshya	2025-12-15	15:30:12	Happy	Present
Karan	2025-12-15			Absent

🎯 Emotion Classes Supported

Happy

Sad

Angry

Neutral

Surprise

Fear

Disgust

🧪 Model Training
Emotion Model

Dataset: FER-2013

Model: CNN

Framework: TensorFlow / Keras

Face Recognition

Embeddings: FaceNet

Classifier: SVM


🌟 Future Enhancements

🔐 Database integration (MySQL / Firebase)

📊 Attendance analytics dashboard

🎥 Live emotion overlay

📱 Mobile-friendly UI

🏫 Multi-class support

🔒 Admin authentication


🧑‍💻 Author

Lakshya Srivastav
Computer Vision & Deep Learning Project


⭐ Acknowledgements

FER-2013 Dataset

MediaPipe by Google

FaceNet Research Paper

TensorFlow & OpenCV Communities

📌 License

This project is licensed for academic and educational use.
