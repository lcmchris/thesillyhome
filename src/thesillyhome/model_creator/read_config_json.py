import json
import requests
import os

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

# Opening default options JSON file
f = open("/data/options.json")

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

# # Requests - maybe not so great to use this and instead allow for direct config
# '''curl -sSL -H "Authorization: Bearer $SUPERVISOR_TOKEN" http://supervisor/network/info
# docker inspect homeassistant | grep SUPERVISOR_TOKEN

#         {
#     "addon": "awesome_mysql",
#     "host": "172.0.0.17",
#     "port": "8883",
#     "username": "awesome_user",
#     "password": "strong_password"
#     }
# '''

# supervisor_token = os.environ['SUPERVISOR_TOKEN']
# headers = {"Authorization": f"Bearer {supervisor_token}"}
# r = requests.get('http://supervisor/cli/stats', headers=headers )
# print(r)
# db_output_request = r.json()
# print(db_output_request)
# # host = db_output_request['host']
# # port = db_output_request['port']
# # username = db_output_request['username']
# # password = db_output_request['password']
