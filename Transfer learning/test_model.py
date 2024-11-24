import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

# Paths
MODEL_PATH = 'face_recognition_model.h5'
INPUT_IMAGE_PATH = 'dl.jpg'
RESULT_DIR = 'result'

# Parameters
IMG_SIZE = 160

# Load Model
model = load_model(MODEL_PATH)


# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    processed_img = preprocess_input(np.expand_dims(resized_img, axis=0))
    return img, processed_img

# Recognize Faces
original_image, processed_image = preprocess_image(INPUT_IMAGE_PATH)
predictions = model.predict(processed_image)
predicted_class = np.argmax(predictions, axis=1)

# Annotate and Save Results
class_labels = {v: k for k, v in train_generator.class_indices.items()}  # Update this based on your class mapping
predicted_label = class_labels[predicted_class[0]]
cv2.putText(original_image, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite(f"{RESULT_DIR}/recognized.jpg", original_image)
print(f"Result saved to {RESULT_DIR}/recognized.jpg")
