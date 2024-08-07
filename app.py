from flask import Flask, request, jsonify
from src.model import Model
import pandas as pd

from dotenv import load_dotenv

load_dotenv()
model = Model()
app = Flask(__name__)
@app.route('/')

# def hello_world():
#     return "Coucou !"


@app.route('/', methods=['GET'])
def home():
    return "OK !", 200


@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    df = pd.read_json(body)
    results = [int(x) for x in model.predict(df).flatten()]
    return jsonify(results), 200


if __name__ == "__main__":
    # Please do not set debug=True in production
    app.run(host="0.0.0.0", port=5000, debug=True)