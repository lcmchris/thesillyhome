import appdaemon.plugins.hass.hassapi as hass
import configuration
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class ModelExecutor(hass.Hass):
    def initialize(self):
        self.handle = self.listen_state(self.state_handler)
        self.act_model_set = self.load_models()
        print(self.act_model_set)

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

        # Get state of all sensors for model input
        sensor_state = {}
        for sensor in configuration.sensors:
            sensor_state[sensor] = self.get_state(entity_id=sensor)
        sensor_state = pd.DataFrame.from_dict([sensor_state])
        print(sensor_state)

        # Execute all models for sensor
        for act, model in self.act_model_set.items():
            print(model)
            print(model.predict(sensor_state))
