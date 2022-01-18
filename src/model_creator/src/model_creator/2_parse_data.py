#%%
from audioop import add
from msilib import sequence
from turtle import position
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
import utils
from joblib import Parallel, delayed
import tqdm
from tqdm import tqdm
import numpy as np
from multiprocessing import current_process
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map

# from pandarallel import pandarallel
def parallelize_dataframe(df, func):
    num_processes = cpu_count()
    df_split = np.array_split(df, num_processes)

    with tqdm(total=df.shape[0]) as pbar:
        df_output = pd.concat(
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(func)(split, pbar) for split in df_split
            )
        )
    return df_output


def add_device_states(df_output: pd.DataFrame, pbar):
    for index, row in df_output.iterrows():
        for device in configuration.devices:
            previous_device_state = df_states[
                (df_states["entity_id"] == device)
                & (df_states["created"] < row["created"])
                & (df_states["state"] != "unavailable")
            ]
            if not previous_device_state.empty:
                df_output.loc[index, device] = previous_device_state["state"].iloc[0]
            else:
                df_output.loc[index, device] = np.NaN
        pbar.update(1)
    return df_output


#%%

df_all = homedb().get_data("states")
df_states = df_all[df_all["entity_id"].isin(configuration.devices)]
df_act_states = df_all[df_all["entity_id"].isin(configuration.actuators)]


if __name__ == "__main__":
    """
    Our data base currently stores by events.
    To create a valid ML classification case, we will parse all last
    sensor states for each actuator event and append it to the dataframe.
    """

    number_act_states = len(df_act_states)

    df_output = copy.deepcopy(df_act_states)
    df_output = df_output[df_output["state"] != "unavailable"]

    print("Start parallelization processing...")

    df_output = parallelize_dataframe(df_output, add_device_states)

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

    output_list = ["entity_id", "state", "created"]
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