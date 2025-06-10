#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().system('python -V')


# In[22]:


import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow


# In[23]:


df = pd.read_parquet("/Users/muhammadwisalabdullah/Downloads/yellow_tripdata_2023-01.parquet")


# In[24]:


df.head()


# In[25]:


print(df.tpep_dropoff_datetime - df.tpep_pickup_datetime)
df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds()/60)

print(df.duration)


# In[26]:


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("NYC_FarePredictor")


# In[7]:


def read_dataframe1(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds()/60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


# In[8]:


train_df = read_dataframe1("/Users/muhammadwisalabdullah/Downloads/yellow_tripdata_2023-01.parquet")
valid_df = read_dataframe1("/Users/muhammadwisalabdullah/Downloads/yellow_tripdata_2023-02.parquet")
print(train_df.shape)
print(valid_df.shape)


# In[9]:


train_df['PU_DO'] = train_df['PULocationID'] + '_' + train_df['DOLocationID']
valid_df['PU_DO'] = valid_df['PULocationID'] + '_' + valid_df['DOLocationID']


# In[10]:


valid_df.head()


# In[11]:


categorical = ['PU_DO']


# In[12]:


train_dicts = train_df[categorical].to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
print(f"Dimensionality after OHE: {X_train.shape[-1]}")


# In[13]:


target = 'duration'
y_train = train_df[target].values


# In[14]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[15]:


y_pred = lr.predict(X_train)
import numpy as np

mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
print(rmse)


# In[16]:


val_dicts = valid_df[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
print(f"Dimensionality after OHE: {X_val.shape[-1]}")


# In[17]:


target = 'duration'
y_val = valid_df[target].values
y_pred = lr.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(rmse)


# In[20]:


with mlflow.start_run(run_name="NYC_FarePredictor-LinearRegression"):
    mlflow.log_param("rmse", rmse)
    with open("dict_vectorizer.bin", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("dict_vectorizer.bin")

    mlflow.log_model(lr, artifact_path="model")

