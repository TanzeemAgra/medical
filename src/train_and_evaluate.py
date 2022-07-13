
import os, sys
import warnings
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse, joblib, json

def eval_metrics(actual, pred):
    rmse=np.sqrt(mean_squared_error(actual, pred))
    mae=mean_absolute_error(actual, pred)
    r2=r2_score
    return rmse,mae,r2

def train_and_evaluate(config_path):
    






if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)