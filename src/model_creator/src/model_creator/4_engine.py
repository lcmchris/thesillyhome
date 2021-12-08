#%%
from hassapi import Hass
import homeassistant.helpers.event as ev
import configuration

home = Hass(hassurl="https://novabb.duckdns.org/", token=configuration.auth_token)

states = home.get_states()
print(states)
