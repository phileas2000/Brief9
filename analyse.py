import pandas as pd
import sqlite3 as sql

def decategoriser(colonne):
    dataMap = pd.read_sql_query("SELECT "+str(colonne)+" ,AVG(Valeur_fonciere) as Valeur_fonciere FROM Data GROUP BY "+str(colonne)+" ORDER BY AVG(Valeur_fonciere)",conn)
    #print(dataMap["Valeur_fonciere"].drop_duplicates())
    serie = dataMap["Valeur_fonciere"].squeeze()
    print(dataMap["Valeur_fonciere"].values)
    dictMap =  pd.Series(dataMap["Valeur_fonciere"].values ,index=dataMap[colonne]).to_dict()
    print(dictMap)
    
    
    return pd.to_numeric(data[colonne].replace(dictMap))

conn  = sql.connect("immobilier.db")
data = pd.read_sql_query("SELECT * FROM Data",conn)
data = data.sample(n = 10000,random_state=1)
colonnes_categorielle = ["Commune","Code_postal"]
for e in colonnes_categorielle:
        data[e] = decategoriser(e)
#print(data[colonnes_categorielle])
#colonnes_categorielle = colonnes_categorielle.append("Valeur_fonciere")
print(colonnes_categorielle)
print(data.corr())

