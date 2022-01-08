#%%
from numpy import NaN
from pandas.core.frame import DataFrame
from sklearn.preprocessing import OneHotEncoder
from home import homedb
import configuration
from time import gmtime
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier as ClassificationModel
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import copy
from tqdm import tqdm
"""
Our data base currently stores by events.
To create a valid ML classification case, we will parse all last
sensor states for each actuator event and append it to the dataframe.
"""

df_states = homedb().get_data("states").copy()
df_states = df_states[df_states["entity_id"].isin(configuration.devices)].copy()

df_sen_states = df_states[df_states["entity_id"].isin(configuration.sensors)].copy()
df_act_states = df_states[df_states["entity_id"].isin(configuration.actuators)].copy()

number_act_states = len(df_act_states)

df_output = copy.deepcopy(df_act_states)
df_output = df_output[df_output['state']!='unavailable']

for sensor in configuration.sensors:
    df_output[sensor] = ""

for index, row in tqdm(df_output.iterrows(),total=df_output.shape[0]):
    target = configuration.devices
    created_time = row["created"]
    for device in target:
        last_device_state = df_states[
            (df_states["entity_id"] == device)
            & (df_states["created"] < created_time)
            & (df_states["state"] != 'unavailable')
        ]
        if not last_device_state.empty:
            df_output.loc[index, device] = last_device_state['state'].iloc[0]
        else:
            df_output.loc[index, device] = np.NaN

"""
Code to add one hot encoding for date time.
This will help give features for time of day and day of the week.
"""
df_output["hour"] = df_output["created"].dt.hour
df_output["weekday"] = df_output["created"].dt.date.apply(lambda x: x.weekday())
df_output = df_output.drop(columns=["created"])


# Hot encoding for all features
def one_hot_encoder(df: DataFrame, column: str) -> DataFrame:
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = df.drop(column, axis=1)
    df = df.join(one_hot)
    return df


output_list = ["entity_id", "state"]
feature_list = list(set(df_output.columns) - set(output_list))

for feature in feature_list:
    # For lux sensors, these are already in Int format so no encoding.
    if feature not in configuration.sensors_lux:
        df_output = one_hot_encoder(df_output, feature)

# Remove some empty entity_id rows
df_output = df_output[df_output["entity_id"] != ""]

# Lux sensors has 'unknown values' which need to be removed
for lux in configuration.sensors_lux:
    df_output[lux] = df_output[lux].replace([np.NaN, "unknown", ""], 0)

df_output.to_csv(configuration.states_csv, index=False)

# %%
