import json
import os
import uuid
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)


BRID_SPEC_PATH = 'data/data_species.json'
UPLOAD_FOLDER = 'predict_images'
PORT = 5000
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


with open(BRID_SPEC_PATH, 'r') as json_file:
    json_data = json.load(json_file)

bird_data = {}


for key, value in json_data.items():
    bird_data[int(key)] = value


# load model
model = load_model('model/BC1.h5', compile=False)


############# APIs #####################
# root
@app.route('/')
def home():
    return "Bird prediction API"

# check server health status


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return "OK"


# predictiion API

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'invalid arguements'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'invalid arguements'}), 400

    if file:
        unique_filename = str(uuid.uuid4()) + '.' + \
            file.filename.rsplit('.', 1)[1]
        filename = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filename)
        file_url = filename
        result: int = predictbird(file_url)
        if result != -1:
            return jsonify({'status': 'IDENTIFILED', 'prediction': {
                'id': result,
                'name': bird_data[result]
            }}), 200
        else:
            return jsonify({'status': 'NOTIDENTIFILED', }), 200


################### services ######################

# prediction funtion

def predictbird(img_url: str) -> int:
    img = load_img(img_url, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    print(answer)
    # Get the predicted class index and the associated confidence level
    y_class = answer.argmax(axis=-1)
    confidence = answer.max()
    print(confidence)
    # threshold for confidence level
    threshold = 0.7

    if confidence >= threshold:
        return int(y_class)
    else:
        return -1


################### main ######################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

    # app.run(host='127.0.0.1', port=PORT, debug=True)
