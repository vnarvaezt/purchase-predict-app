import pandas as pd
import requests

dataset = pd.read_csv("~/purchase_predict_api/tests/data/primary.csv")
result = requests.post(
    "http://127.0.0.1:9000/predict",
    json=dataset.sample(n=10).to_json()
).json()

print(result)