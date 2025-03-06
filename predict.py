import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Load the trained model

model_path = r"D:\Srinidhi\VIT\Subjects\CAPSTONE PROJECT\Trys\Handwriting\Try2\handwriting_cnn_model.h5"
cnn_model = load_model(model_path)
# Path to the image you want to classify
image_path = r'D:\Srinidhi\VIT\Subjects\CAPSTONE PROJECT\Trys\Handwriting\Dataset1\Gambo\Test\Reversal\1_5.png'  # Replace with your image path

# Load the image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to match the model input size (32x32)
img_resized = cv2.resize(img, (32, 32))

# Normalize the image (same preprocessing done during training)
img_normalized = img_resized / 255.0

# Reshape the image to match the expected input format (batch_size, height, width, channels)
img_batch = np.expand_dims(img_normalized, axis=-1)  # Shape becomes (32, 32, 1)
img_batch = np.expand_dims(img_batch, axis=0)  # Shape becomes (1, 32, 32, 1)

# Make a prediction
predictions = cnn_model.predict(img_batch)

# Print the prediction (probabilities for each class)
print(f"Predictions: {predictions}")

# Get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)

# Print the predicted class
categories = ["Corrected", "Normal", "Reversal"]  # List of categories
predicted_label = categories[predicted_class[0]]
print(f"Predicted class: {predicted_label}")