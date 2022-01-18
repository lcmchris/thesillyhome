import appdaemon.plugins.hass.hassapi as hass
import configuration
import pickle
import pandas as pd
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import copy


class ModelExecutor(hass.Hass):
    def initialize(self):
        self.handle = self.listen_state(self.state_handler)
        self.act_model_set = self.load_models()

    def load_models(self):
        act_model_set = {}
        for act in configuration.actuators:
            with open(
                f"{configuration.home}/src/appdaemon/apps/model/{act}.pickle", "rb"
            ) as pickle_file:
                content = pickle.load(pickle_file)
                act_model_set[act] = content
        return act_model_set

    def state_handler(self, entity, attribute, old, new, kwargs):
        if entity in configuration.sensors:
            print(f"{entity} is {new}")

            # Get feature list from parsed data header, set all columns to 0
            feature_list = pd.read_csv(configuration.states_csv).columns
            feature_list = feature_list.drop(["entity_id", "state"])
            feature_list = pd.DataFrame(columns=feature_list)
            feature_list = feature_list.append(pd.Series(), ignore_index=True)
            feature_list.iloc[0] = 0

            # Get state of all sensors for model input
            df_sen_states = copy.deepcopy(feature_list)
            for sensor in configuration.sensors:
                true_state = self.get_state(entity_id=sensor)
                if sensor not in configuration.sensors_lux:
                    if (sensor + "_" + true_state) in df_sen_states.columns:
                        df_sen_states[sensor + "_" + true_state] = 1
                elif sensor in configuration.sensors_lux:
                    if (true_state) in df_sen_states.columns:
                        df_sen_states[sensor] = true_state

            # Execute all models for sensor and set states
            for act, model in self.act_model_set.items():
                prediction = model.predict(df_sen_states)[0].split("::")[1]
                if prediction == "on":
                    print(f"Turn on {act}")
                    self.turn_on(act)
                elif prediction == "off":
                    print(f"Turn off {act}")
                    self.turn_off(act)
