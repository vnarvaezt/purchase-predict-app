import mlflow
from mlflow.tracking import MlflowClient
import joblib
import os
model_name = "sk-learn-random-forest-reg-model"
model_version = "latest"

class Model():
    def __init__(self):
        self.registry_name = os.getenv("MLFLOW_REGISTRY_NAME")
        self.load_model()


    def load_model(self):
        client = MlflowClient()
        # load model
        model_version = client.get_latest_versions(self.registry_name, stages=["None"])[0]
        model_uri = f"models:/{self.registry_name}/{model_version.version}"
        self.model = mlflow.sklearn.load_model(model_uri)
        # transformed pipeline
        path_pipeline = client.download_artifacts(model_version.run_id, "transform_pipeline.pkl")
        self.transform_pipeline = joblib.load(path_pipeline)

    def predict(self, X):
        X_ready = self.transform_pipeline.transform(X)
        return self.model.predict(X_ready)


        