from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')


# Define the function to preprocess the uploaded image
def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("Error preprocessing image:", e)
        return None


# Define the function to classify the image
def predict_image(img):
    try:
        result = model.predict(img)
        if result[0][0] == 1:
            return "Dog"
        else:
            return "Cat"
    except Exception as e:
        print("Error predicting image:", e)
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # If the file exists and is allowed
        if file:
            try:
                # Generate a unique filename to prevent conflicts
                filename = str(uuid.uuid4()) + '.jpg'
                img_path = os.path.join('static', 'uploads', filename)
                file.save(img_path)

                # Preprocess the uploaded image
                img = preprocess_image(img_path)
                if img is None:
                    return render_template('index.html', message='Error preprocessing image')

                # Predict the image
                prediction = predict_image(img)
                if prediction is None:
                    return render_template('index.html', message='Error predicting image')

                # Pass the relative image path to the template
                return render_template('result.html', prediction=prediction, img_filename=filename)
            except Exception as e:
                print("Error handling image upload:", e)
                return render_template('index.html', message='Error handling image upload')


if __name__ == '__main__':
    app.run(debug=True)
