import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import os
import math

print("🚀 Training Coffee Bean Model...")

dataset_path = "dataset"
img_size = (224, 224)
batch_size = 32

# Data generator
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_data.class_indices.keys())

# Save class names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

steps_per_epoch = math.ceil(train_data.samples / batch_size)
val_steps = math.ceil(val_data.samples / batch_size)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=[early_stop]
)

# ✅ Save model (so no retraining needed)
model.save("coffee_model.keras")

# ✅ Save history
with open("history.json", "w") as f:
    json.dump(history.history, f)

print("🎉 Training Done!")

# ---------------------------
# 📊 PLOTS
# ---------------------------
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train", "Validation"])
plt.savefig("accuracy.png")

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train", "Validation"])
plt.savefig("loss.png")

# ---------------------------
# 📉 CONFUSION MATRIX
# ---------------------------
val_data.reset()
preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# ---------------------------
# 📄 CLASSIFICATION REPORT
# ---------------------------
report = classification_report(y_true, y_pred, target_names=class_names)

with open("report.txt", "w") as f:
    f.write(report)

print(report)