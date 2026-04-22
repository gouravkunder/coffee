import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tkinter import Tk, filedialog
import json

# Load model
model = tf.keras.models.load_model("coffee_model.keras")

# Load classes
with open("class_names.json", "r") as f:
    class_names = json.load(f)

Tk().withdraw()
file_path = filedialog.askopenfilename()

img = image.load_img(file_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

pred = model.predict(img_array)

class_idx = np.argmax(pred)
confidence = pred[0][class_idx] * 100

print("Prediction:", class_names[class_idx])
print("Confidence:", f"{confidence:.2f}%")