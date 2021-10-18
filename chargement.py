import pandas as pd
import sqlite3 as sql

data = pd.read_csv("valeursfoncieres-2020.txt",sep="|",low_memory=False)
print(len(data))
data =  data.drop_duplicates()

data.columns   = data.columns.str.replace(' ','_')
print(data.columns)
data["Valeur_fonciere"] = data["Valeur_fonciere"].str.replace(',','.')
data = data.astype({"Valeur_fonciere": float})
data = data[['Valeur_fonciere','Code_voie', 'Type_de_voie', 'Code_postal', 'Commune',
'Surface_reelle_bati', 'Nombre_pieces_principales', 'Surface_terrain']]
print("DROPNA")
print(len(data))
print(data.describe())
data =  data.dropna()
print(len(data))
print(data.describe())

conn = sql.connect('immobilier.db')
cursor = conn.cursor()

data.to_sql("Data",conn, if_exists='replace')
print(pd.read_sql_query("SELECT * FROM Data limit 10000",conn))