states_csv = "E:/thesillyhomedb/states.csv"

"""
full list of domains
"sun",
"light",
"sensor",
"device_tracker",
"binary_sensor",
"weather",
"person",
"switch",
"automation",
"cover",
"media_player",
"persistent_notification",
"script",
"zone",
"scene",
"""

# domains in scope
domains = [
    "light",
    "switch",
    "binary_sensor",
    "sensor",
    "device_tracker",
    "weather",
]

"""
Entities in scope
Actuators - light & switches
All lights & switches inscope

"""
act_light = [
    "light.bedroom_tommy",
    "light.bedroom_bobby",
    "light.hallway_lights",
    # "light.hallway_5",
    # "light.hallway_4",
    # "light.hallway_3",
    # "light.hallway_2",
    # "light.hallway_1",
]

act_switch = [
    "switch.0x04cf8cdf3c7ceb38_right",
    "switch.0x04cf8cdf3c7ceb38_center",
    "switch.0x04cf8cdf3c7ceb38_left",
    "switch.livingroom_center_socket_right",
    "switch.livingroom_center_socket_left",
]

actuators = act_light + act_switch

# Sensors entities
sensors = [
    "device_tracker.chris_phone",
    "sensor.0x00158d0001e4ad46_illuminance_lux",
    "sensor.0x00158d00066941c9_illuminance_lux",
    "sensor.bedroom_entrance_illuminance_lux",
    "sensor.livingroom_desk_sensor_illuminance_lux",
    "binary_sensor.0x00158d0001e4ad46_occupancy",
    "binary_sensor.0x00158d00066941c9_occupancy",
    "binary_sensor.0x00158d00045d21c2_occupancy",
    "binary_sensor.livingroom_desk_sensor_occupancy",
    "weather.home",
]

devices = actuators + sensors
