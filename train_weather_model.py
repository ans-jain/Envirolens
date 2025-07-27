import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set dataset directory (CHANGE THIS to your dataset path)
DATASET_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\ALL SUBJECTS\web programming\Project_web\dataset"

# Define categories (change these based on your dataset)
CATEGORIES = ["sunny", "night", "cloudy"]  # Adjust as per your dataset

# Image size for resizing
IMG_SIZE = 224

# Load images and labels
def load_data():
    images, labels = [], []
    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        label = CATEGORIES.index(category)  # Convert category name to index
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)  # Read image
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 224x224
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

# Load dataset
X, y = load_data()

# Normalize images (scale pixel values to 0-1)
X = X / 255.0

# Split dataset into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN Model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(CATEGORIES), activation='softmax')  # Output layer (number of categories)
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save trained model
model.save("weather_model.h5")
print("Model training complete. Model saved as weather_model.h5")

