import mysql.connector
import pandas as pd
from datetime import datetime
import configuration

"""
  Get data from DB and store locally
"""


class homedb:
    def __init__(self):
        self.host = "34.255.195.220"
        self.port = "3306"
        self.user = "lcmchris"
        self.password = "1Bittermelon."

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
                created, state_id, old_state_id from \
                homeassistant.states ORDER BY created ASC;"

        mycursor.execute(query)
        myresult = mycursor.fetchall()
        df = pd.DataFrame.from_dict(myresult)
        df.to_csv(configuration.states_csv)
        return df

    def store_data(self, table: str):
        today = datetime.today().strftime("%Y_%m_%d")
        self.get_data(table).to_csv(f"{today}_{table}.csv")
