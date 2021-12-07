import appdaemon.plugins.hass.hassapi as hass
from src.model_creator import configuration


class ModelExecutor(hass.Hass):
    def initialize(self):
        self.handle = self.listen_state(
            self.state_handler, entity=configuration.sensors
        )

    def state_handler(self, entity, attribute, old, new, kwargs):
        print(self.handle)
        print(entity)
