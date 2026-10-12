from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model("skin_cancer_model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        if file:
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Preprocess image
            img = image.load_img(file_path, target_size=(64, 64))
            img = image.img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)
            prob = prediction[0][0]

            if prob > 0.5:
                confidence = prob * 100
                result = "Malignant"
                description = "The lesion appears suspicious and may indicate skin cancer."
                advice = "Immediate consultation with a certified dermatologist is strongly recommended for further clinical examination and possible biopsy."
            else:
                confidence = (1 - prob) * 100
                result = "Benign"
                description = "The lesion appears non-cancerous based on AI analysis."
                advice = "No immediate medical emergency detected. However, regular monitoring is advised. Consult a healthcare professional if changes occur."

            return render_template(
                "result.html",
                result=result,
                confidence=f"{confidence:.2f}",
                description=description,
                advice=advice,
                image_path=file_path
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)