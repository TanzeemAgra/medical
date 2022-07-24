from src.get_data import read_params
import pandas as pd
from sklearn.metrics import mean_absolute_error
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os
from mlflow import pyfunc


def log_production_model(config_path):
    config = read_params(config_path)  
    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_source=mlflow_config["artifacts_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs(experiment_ids="Experiment ID")
    #lowest = runs["metrics.mae"].sort_values(ascending=True)[0]
    #lowest_run_id = runs[runs["metrics.mae"] == lowest]["run_id"][0]

    client = MlflowClient()
    result = client.create_model_version(
    name=model_name,
    source=model_source,
    run_id="1499fcc1e37044b9be2a6cfd30623e00")

    #model_name = "sk-learn-random-forest-reg-model"
    stage = 'Staging'

    #model = mlflow.pyfunc.load_model( model_uri=f"models:/{model_name}/{stage}")

    #model_version = 1
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    model_path = config["webapp_model_dir"] #"prediction_service/model"
    joblib.dump(loaded_model, model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data=log_production_model(config_path=parsed_args.config)