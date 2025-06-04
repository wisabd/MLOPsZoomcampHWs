#!/usr/bin/env python
#

# In[2]:


import pandas as pd
import pickle
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import mlflow
from pathlib import Path




df = pd.read_parquet("/Users/muhammadwisalabdullah/Downloads/yellow_tripdata_2023-01.parquet")


# In[5]:


df.head()


# In[6]:


print(df.tpep_dropoff_datetime - df.tpep_pickup_datetime)
df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds()/60)

print(df.duration)


# In[7]:


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("NYC_FarePredictor")


# In[8]:


def read_dataframe1(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds()/60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


# In[9]:


train_df = read_dataframe1("/Users/muhammadwisalabdullah/Downloads/yellow_tripdata_2023-01.parquet")
valid_df = read_dataframe1("/Users/muhammadwisalabdullah/Downloads/yellow_tripdata_2023-02.parquet")
print(train_df.shape)
print(valid_df.shape)


# In[10]:


train_df['PU_DO'] = train_df['PULocationID'] + '_' + train_df['DOLocationID']
valid_df['PU_DO'] = valid_df['PULocationID'] + '_' + valid_df['DOLocationID']


# In[11]:


valid_df.head()


# In[12]:


categorical = ['PU_DO']


# In[ ]:





# In[13]:


train_dicts = train_df[categorical].to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
print(f"Dimensionality after OHE: {X_train.shape[-1]}")


# In[14]:


target = 'duration'
y_train = train_df[target].values


# In[15]:


val_dicts = valid_df[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)


# In[17]:


y_val = valid_df[target].values


# In[20]:



models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


# In[21]:


with mlflow.start_run(run_name="NYC_FarePredictor"):
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=valid_df[target])


    best_params = {
        'learning_rate': 0.09585,
        'max_depth': 30,
        'min_child_weight': 1.6059,
        'objective': 'reg:linear',
        'reg_alpha': 0.018,
        'reg_lambda': 0.01165,
        'seed': 42,
    }

    mlflow.log_params(best_params)
    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=30,
        evals=[(valid, "validation")],
        early_stopping_rounds=50,
    )
    y_pred = booster.predict(valid)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


# In[ ]:




