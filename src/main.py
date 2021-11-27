#%%
from numpy import NaN
from pandas.core.frame import DataFrame
import sklearn
from sklearn.preprocessing import OneHotEncoder
from home import homedb
import configuration
import logging
from time import gmtime

import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier as ClassificationModel
from sklearn.metrics import accuracy_score
import datetime

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
            df_act_states.loc[index, sensor] = NaN

"""
Code to add one hot encoding for date time.
This will help give features for time of day and day of the week.
"""

df_act_states["second"] = df_act_states["created"].dt.minute
df_act_states["minute"] = df_act_states["created"].dt.minute
df_act_states["hour"] = df_act_states["created"].dt.hour
df_act_states["weekday"] = df_act_states["created"].dt.date.apply(lambda x: x.weekday())


def one_hot_encoder(df: DataFrame, column: str) -> DataFrame:
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = df.drop(column, axis=1)
    df = df.join(one_hot)
    return df


df_act_states = one_hot_encoder(df_act_states, "weekday")
df_act_states = one_hot_encoder(df_act_states, "minute")
df_act_states = one_hot_encoder(df_act_states, "hour")
df_act_states = one_hot_encoder(df_act_states, "second")


df_act_states.to_csv("act_states.csv")
df_sen_states.to_csv("sen_states.csv")
#%%
# Generate dataset from home states
X = []
y = []
for home in home_states:
    X.append(home.feature_vector())
    y.append(home.output_vector())


#%%
# Split into random training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=gmtime().tm_sec
)  # 2)

# Fit to model
model = ClassificationModel()

model.fit(X_train, y_train)

y_predictions = model.predict(X_test)

# Extract predictions for each output variable and calculate accuracy and f1 score
for ov in range(len(y_test[0])):
    variable_y_predictions = [prediction[ov] for prediction in y_predictions]
    variable_y_test = [test[ov] for test in y_test]
    print(
        "Accuracy Score for output variable {}: {} %".format(
            ov,
            round(
                accuracy_score(variable_y_test, variable_y_predictions, True) * 100, 2
            ),
        )
    )
    # print("F1 Score for output variable {}: {}".format(ov, f1_score(variable_y_test, variable_y_predictions)))

# Store the preprocessor and model

joblib.dump(encoder, "models/feature_vector_encoder.pkl")
joblib.dump(model, "models/random_forest_model.pkl")
