import mlflow
from mlflow.tracking import MlflowClient
import joblib
import os
model_name = "sk-learn-random-forest-reg-model"
model_version = "latest"

class Model():
    def __init__(self):
        self.registry_name = os.getenv("MLFLOW_REGISTRY_NAME")
        self.mlflow_server = os.getenv("MLFLOW_SERVER")
        self.load_model()


    def load_model(self):
        client = MlflowClient(self.mlflow_server)
        mlflow.set_tracking_uri(self.mlflow_server)
        # load model
        model_version = client.get_latest_versions(name=self.registry_name, stages=["None"])[0]
        model_uri = f"models:/{self.registry_name}/{model_version.version}"
        self.model = mlflow.sklearn.load_model(model_uri)
        # transformed pipeline
        path_pipeline = client.download_artifacts(model_version.run_id, "transform_features.pkl")
        self.transform_pipeline = joblib.load(path_pipeline)

    def predict(self, X):
        X = X.drop(["user_id", "user_session", "purchased"], axis=1).copy()
        if self.model:
            if self.transform_pipeline:
                for name, encoder in self.transform_pipeline:
                    X[name] = X[name].fillna("unknown")
                    X[name] = encoder.transform(X[name])
        return self.model.predict(X)


        