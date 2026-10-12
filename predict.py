import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("skin_cancer_model.h5")

# Load test image
img = image.load_img("test2.jpg", target_size=(64, 64))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Malignant (Cancer Detected)")
else:
    print("Benign (No Cancer)")
