import pandas as pd
import requests

dataset = pd.read_csv("~/purchase_predict_api/data/primary.csv")
#dataset = dataset.drop(["user_session", "user_id", "purchased"], axis=1)

requests.post(
    "http://127.0.0.1:5000/predict",
    json=dataset.sample(n=10).to_json()
).json()