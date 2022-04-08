import mysql.connector
import pandas as pd
from datetime import datetime
import configuration

"""
  Get data from DB and store locally
"""


class homedb:
    def __init__(self):
        self.host = "192.168.1.100"
        self.port = "3306"
        self.user = "homeassistant"
        self.password = "urY8BTmQJD7eZuDM"

    def get_data(self, table: str):
        assert table in ["states"], "Table not valid."
        mydb = mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
        )
        mycursor = mydb.cursor(dictionary=True)
        if table == "states":
            query = "SELECT entity_id, state,\
                created from \
                homeassistant.states ORDER BY created DESC;"

        mycursor.execute(query)
        myresult = mycursor.fetchall()
        df = pd.DataFrame.from_dict(myresult)
        df.to_csv(configuration.states_csv)
        return df

    def store_data(self, table: str):
        today = datetime.today().strftime("%Y_%m_%d")
        self.get_data(table).to_csv(f"{today}_{table}.csv")
