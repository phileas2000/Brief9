import pandas as pd
import sqlite3 as sql


data = pd.read_csv("df_clean.csv",low_memory=False)
data.columns   = data.columns.str.replace(' ','_')
conn = sql.connect('immobilier.db')
cursor = conn.cursor()

data.to_sql("Data",conn, if_exists='replace')