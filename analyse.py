import pandas as pd
import sqlite3 as sql


conn  = sql.connect("immobilier.db")
data = pd.read_sql_query("SELECT * FROM Data",conn)
print(data.corr())