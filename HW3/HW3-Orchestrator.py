

import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import mlflow
from pathlib import Path
import numpy as np


from prefect import task, flow, get_run_logger
import os


@task
def load_dataframe(filename):
    df = pd.read_parquet(filename)
    return df



# In[8]:

@task
def read_dataframe1(df):
    #df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds()/60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(df.shape)

    return df


# In[9]:

@task
def transform_dataframe(train_df, val_df, categorical):
    train_dicts = train_df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    print(f"Dimensionality after OHE: {X_train.shape[-1]}")

    target = 'duration'
    y_train = train_df[target].values

    val_dicts = val_df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_val = val_df[target].values

    return X_train, y_train, X_val, y_val, dv








# In[21]:
@task
def train(X_train, X_val, y_train, y_val, dv):
    with mlflow.start_run(run_name="NYC_FarePredictor"):
        #train = xgb.DMatrix(X_train, label=y_train)
        #valid = xgb.DMatrix(X_val, label=valid_df[target])


        #best_params = {
            #'learning_rate': 0.09585,
           # 'max_depth': 30,
           # 'min_child_weight': 1.6059,
           # 'objective': 'reg:linear',
           # 'reg_alpha': 0.018,
           # 'reg_lambda': 0.01165,
            #'seed': 42,
        #}

        #mlflow.log_params(best_params)
       # booster = xgb.train(
         #   params=best_params,
         #   dtrain=train,
          #  num_boost_round=30,
          #  evals=[(valid, "validation")],
           # early_stopping_rounds=50,
        #)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)

        #y_pred = booster.predict(valid)

        mse = mean_squared_error(y_train, y_pred)
        rmse = np.sqrt(mse)
        print("Training RMSE:", rmse)
        print("Intercept", lr.intercept_)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        #mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")

@flow
def main(train_path: str = "/Users/muhammadwisalabdullah/Downloads/yellow_tripdata_2023-03.parquet",
         valid_path: str = "/Users/muhammadwisalabdullah/Downloads/yellow_tripdata_2023-02.parquet",categorical = ['PULocationID', 'DOLocationID']):

    train_df = load_dataframe(train_path)
    valid_df = load_dataframe(valid_path)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("NYC_FarePredictor")

    train_df1 = read_dataframe1(train_df)
    valid_df1 = read_dataframe1(valid_df)

    X_train, y_train, X_val, y_val, dv = transform_dataframe(train_df1, valid_df1, categorical)

    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)

    train(X_train, X_val, y_train, y_val, dv)




# In[ ]:
if __name__ == "__main__":
    main()




