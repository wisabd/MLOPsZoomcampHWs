import sys

#get_ipython().system('pip list | grep scikit-learn')
#get_ipython().system('python -V')

import pickle
import pandas as pd
import numpy as np

year = int(sys.argv[1])  #2021
month = int(sys.argv[2]) #2

#/Users/muhammadwisalabdullah/PycharmProjects/MLOPsZoomcampHWs/HW1/dict_vectorizer.bin
#/Users/muhammadwisalabdullah/PycharmProjects/MLOPsZoomcampHWs/HW1/mlruns/1/b47201c45dbb4454b69ed704253f06c0/artifacts/dict_vectorizer.bin
with open('/Users/muhammadwisalabdullah/PycharmProjects/MLOPsZoomcampHWs/HW1/mlruns/1/1e6ca94c7943433f9540263ca2dc4f34/artifacts/dict_vectorizer.bin', "rb") as f_in:
    dv = pickle.load(f_in)

with open('/Users/muhammadwisalabdullah/PycharmProjects/MLOPsZoomcampHWs/HW1/mlruns/1/1e6ca94c7943433f9540263ca2dc4f34/artifacts/model/model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

train_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'/Users/muhammadwisalabdullah/PycharmProjects/MLOPsZoomcampHWs/HW4/yellow_tripdata_{year:04d}-{month:02d}.parquet'

train_df = read_data(train_path)

train_df['PU_DO'] = train_df['PULocationID'] + '_' + train_df['DOLocationID']
categorical = ['PU_DO']
train_dicts = train_df[categorical].to_dict(orient='records')

X_train = dv.transform(train_dicts)
y_pred = model.predict(X_train)


print(y_pred)

y_pred = model.predict(X_train)

print(f"Mean of predictions: {np.mean(y_pred)}")

std_dev = np.std(y_pred)

print(f"Standard deviation of predictions: {std_dev:.4f}")

year = 2023
month = 3

#output_file = "/Users/muhammadwisalabdullah/PycharmProjects/MLOPsZoomcampHWs/HW4/results_df.parquet"

train_df['ride_id'] = f'{year:04d}/{month:02d}_' + train_df.index.astype('str')

df_result = pd.DataFrame()
df_result['ride_id'] = train_df['ride_id']
df_result['predicted_duration'] = y_pred

import os
os.makedirs("/Users/muhammadwisalabdullah/PycharmProjects/MLOPsZoomcampHWs/HW4/", exist_ok=True)

train_df.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

