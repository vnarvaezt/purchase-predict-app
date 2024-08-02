from flask import Flask, request, jsonify
from model import Model

from dotenv import load_dotenv

load_dotenv()
model = Model()

app = Flask(__name__)
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
    app.run(port=5000)