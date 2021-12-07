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

"""
Our data base currently stores by events.
To create a valid ML classification case, we will parse all last
sensor states for each actuator event and append it to the dataframe.
"""

df_states = homedb().get_data("states").copy()
df_sen_states = df_states[df_states["entity_id"].isin(configuration.sensors)].copy()
df_act_states = df_states[df_states["entity_id"].isin(configuration.actuators)].copy()

for sensor in configuration.sensors:
    df_act_states[sensor] = ""


for index, row in df_act_states.iterrows():
    target = configuration.sensors
    created_time = row["created"]
    for sensor in target:
        if not df_sen_states[
            (df_sen_states["entity_id"] == sensor)
            & (df_sen_states["created"] < created_time)
        ].empty:
            df_act_states.loc[index, sensor] = (
                df_sen_states[
                    (df_sen_states["entity_id"] == sensor)
                    & (df_sen_states["created"] < created_time)
                ]
                .head(1)["state"]
                .values
            )
        else:
            df_act_states.loc[index, sensor] = np.NaN

"""
Code to add one hot encoding for date time.
This will help give features for time of day and day of the week.
"""

# df_act_states["second"] = df_act_states["created"].dt.minute
# df_act_states["minute"] = df_act_states["created"].dt.minute
df_act_states["hour"] = df_act_states["created"].dt.hour
df_act_states["weekday"] = df_act_states["created"].dt.date.apply(lambda x: x.weekday())
df_act_states = df_act_states.drop(columns=["created"])


# Hot encoding for all features
def one_hot_encoder(df: DataFrame, column: str) -> DataFrame:
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = df.drop(column, axis=1)
    df = df.join(one_hot)
    return df


output_list = ["entity_id", "state"]
feature_list = list(df_act_states.columns)
for output in output_list:
    feature_list.remove(output)
for feature in feature_list:
    # For lux sensors, these are already in Int format so no encoding.
    if feature not in configuration.sensors_lux:
        df_act_states = one_hot_encoder(df_act_states, feature)

# Remove some empty entity_id rows
df_act_states = df_act_states[df_act_states["entity_id"] != ""]

# Lux sensors has 'unknown values' which need to be removed
for lux in configuration.sensors_lux:
    df_act_states[lux] = df_act_states[lux].replace([np.NaN, "unknown", ""], 0)

df_act_states.to_csv(configuration.states_csv, index=False)
