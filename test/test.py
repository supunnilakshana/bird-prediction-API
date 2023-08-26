import json
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.densenet import preprocess_input
from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


CATEGORIES = []
for i in range(48):
    CATEGORIES.append(str(i))


def getPrediction(filename):
    result = 100
    image = load_img('H:/chamara_aiya/aswanna/upload/' +
                     filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    prediction = model.predict(image)
    for i in range(len(CATEGORIES)):
        if np.round(prediction[0][i]) == 1.0:
            result = i
    return result


UPLOAD_FOLDER = r'H:\chamara_aiya\aswanna\upload'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/prediction', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            results = "No file part"
            return results
        file = request.files['file']
        if file.filename == '':
            results = "No file selected for uploading"
            return results
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label = getPrediction(filename)
            # result = pest_details(label)
            if label != None:
                print(label)
                return str(label)
            else:
                return "can't identified"


if __name__ == "__main__":
    # app.run(host='192.168.43.81', port=5000, debug=True, threaded=False)
    app.run(port=8000)
