import pandas as pd
import sqlite3 as sql

from sklearn.model_selection import train_test_split 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from  sklearn.ensemble import RandomForestRegressor



def sample(conn):
    data = pd.read_sql_query("SELECT * FROM Data",conn)
    data = data.sample(n = 10000,random_state=1)
    return data

def decategoriser(colonne):
    dataMap = pd.read_sql_query("SELECT "+str(colonne)+" ,AVG(Valeur_fonciere) as Valeur_fonciere FROM Data GROUP BY "+str(colonne)+" ORDER BY AVG(Valeur_fonciere)",conn)
    dictMap =  pd.Series(dataMap["Valeur_fonciere"].values ,index=dataMap[colonne])
    return pd.to_numeric(data[colonne].replace(dictMap))

def split_data(data):

    y =  data["Valeur_fonciere"].values
    X = data.drop(columns= ( ["Valeur_fonciere","index"]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=15)
    return X_train, X_test, y_train, y_test

def random_forest(X_train,y_train,X_test):
    model = RandomForestRegressor(n_estimators=300,max_depth=50)
    model.fit(X_train, y_train)
    return model.predict(X_test)





modele = random_forest
conn  = sql.connect("immobilier.db")
data = sample(conn)


for e in ["Code_postal","Commune","Code_voie",'Type_de_voie']:
    data[e] = decategoriser(e) 


data = pd.DataFrame(data)
data = split_data(data)

X_train = data[0] # scaler.fit_transform()
X_test = data[1] #scaler.fit_transform()
y_train = data[2]
y_test = data[3]
prediction = modele(X_train,y_train,X_test)

prediction = modele(X_train,y_train,X_train)
print("Prédiction: "+ str(prediction))
