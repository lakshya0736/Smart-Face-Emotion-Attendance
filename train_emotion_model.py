import tensorflow as tf
from tensorflow import keras
Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Flatten = keras.layers.Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
train_dir = "data/emotion_datasets/train"
test_dir = "data/emotion_datasets/test"

# Parameters
img_size = 48
batch_size = 64
num_classes = 7

# Data generators (auto preprocessing)
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=20
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/emotion_model.h5")

print("✅ Emotion model trained and saved!")
