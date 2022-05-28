import json

"""
This is the config yaml:
    options:
    actuactors_id:
        - "Test_id_1"
        - "Test_id_2"
    sensors_id:
        - "Test_id_1"
        - "Test_id_2"
    database:
        - "Test_id_1"
        - "Test_id_2"

>>>

{'options': {'actuactors_id': {'Test_id_1': 'ACCENT_2',
   'Test_id_2': 'Times New Roman',
 'sensors_id': {'font': {'Test_id_1': 'ACCENT_2',
   'Test_id_2': 'Times New Roman',
...
"""
env = "PROD"
# Opening default options JSON file
if env == "DEV":
    data_dir = "data_test"
else:
    data_dir = "/data"

f = open(f"{data_dir}/options.json")

options = json.load(f)

actuators = options["actuactors_id"]
sensors = options["sensors_id"]
db_options = options["db_options"][0]
db_password = db_options["db_password"]
db_database = db_options["db_database"]
db_username = db_options["db_username"]
db_host = db_options["db_host"]
db_port = db_options["db_port"]

model_name = "Base"
model_version = "0.0.0"
model_name_version = f"{model_name}_{model_version}"


def extract_float_sensors(sensors: list):
    float_sensors_types = ["lux"]
    float_sensors = []
    for sensor in sensors:
        if sensor.split("_")[-1] in float_sensors_types:
            float_sensors.append(sensor)
    return float_sensors


float_sensors = extract_float_sensors(sensors)

output_list = ["entity_id", "state", "last_changed", "duplicate"]
