import json
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)


json_file_path = 'data/data_species.json'


with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

bird_data = {}


for key, value in json_data.items():
    bird_data[int(key)] = value


# load model
model = load_model('model/BC1.h5', compile=False)


@app.route('/')
def home():
    return "Bird prediction API"


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return "OK"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
