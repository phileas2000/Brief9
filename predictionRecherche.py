import pickle
import pandas as pd
import sqlite3 as sql

from sklearn.model_selection import train_test_split 
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from  sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold


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
    #X =  data["Valeur_fonciere"].values.reshape(-1,1) #.drop(columns= ( ["Valeur_fonciere"]))
    X = data.drop(columns= ( ["Valeur_fonciere","index"]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=15)
    return X_train, X_test, y_train, y_test

def hyperparametre_randomforest(X_train,y_train,X_test):   
    model = RandomForestRegressor(n_estimators=1000,max_depth=150,min_impurity_decrease=0.1)
    model.fit(X_train, y_train)
    distributions = dict(n_estimators=sp_randInt(100,2000),
    max_depth=sp_randInt(100,2000),min_impurity_decrease=sp_randFloat(0.1,0))
    clf = RandomizedSearchCV( RandomForestRegressor(),distributions,random_state=110,n_iter=1)
    search = clf.fit(X_train, y_train)
    print("Meilleurs paramètres :"+ str(search.best_params))
    return model.predict(X_test)


def random_forest(X_train,y_train,X_test):   
    model = RandomForestRegressor(n_estimators=1000,max_depth=150,min_impurity_decrease=0.1)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def linear_regression(X_train,y_train,X_test):   
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def decision_tree(X_train,y_train,X_test):   
    model = RandomForestRegressor(n_estimators=1,max_depth=100)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def elastic_net(X_train,y_train,X_test):   
    regr = ElasticNet(random_state=5)
    regr.fit(X_train, y_train)
    return regr.predict(X_test)

def ridge(X_train,y_train,X_test):   
    regr = Ridge(random_state=17)
    regr.fit(X_train, y_train)
    return regr.predict(X_test)
    

def sgd(X_train,y_train,X_test):   
    reg = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3))
    clf = SGDRegressor()
    clf.fit(X_train, y_train) 
    #print("CLF_SCORE:"+str(clf.score(X_train,y_train)))
    return (abs(clf.predict(X_test))/0.60)


def graph(predictions,y_test):

    fig1, ax1 = plt.subplots()
    ax1.scatter(range(len(predictions)),predictions,marker='x',color="black")
    fig2, ax2 = plt.subplots()
    ax2.scatter(range(len(y_test)),y_test,marker='x',color="grey")
    plt.show()
    
    
    plt.xlabel('Valeur prédite')
    plt.ylabel('Valeur réelle')
    plt.show()





def error(predictions,y_test):
    print('Mean Absolute Error:', metrics.mean_absolute_error( y_test, predictions))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print('R2 score:', r2_score(y_test, predictions))

def errorKfold(predictions,y_test):
    return  r2_score(y_test, predictions)


modele = random_forest
conn  = sql.connect("immobilier.db")
data = sample(conn)

print(data.columns)
for e in ["Code_postal","Commune","Code_voie",'Type_de_voie']:
    data[e] = decategoriser(e) 
#scaler = StandardScaler() 
#scaler.fit(data)
print(data.columns)
print(data[["Code_postal","Commune","Code_voie"]])
dataUnif = pd.DataFrame(data)
data = split_data(dataUnif)
#scaler.transform(data)
X_train = data[0] # scaler.fit_transform()
X_test = data[1] #scaler.fit_transform()
y_train = data[2]
y_test = data[3]
hyperparametre_randomforest(X_train,y_train,X_test)
#graph(prediction,y_test)
n_split = 3
kf = KFold(n_splits=n_split)
moy = 0
'''for train_index,test_index in kf.split(dataUnif.values):
      
      X_train, X_test = dataUnif.values[train_index], dataUnif.values[test_index]
      y_train, y_test = dataUnif["Valeur_fonciere"].values[train_index], dataUnif["Valeur_fonciere"].values[test_index]
      prediction = modele(X_train,y_train,X_test)
      print("Prediction: "+str(errorKfold(prediction,y_test)))
      moy += errorKfold(prediction,y_test)
print("Précision moyenne: " + str(moy/n_split))
prediction = modele(X_train,y_train,X_test)
print("Sans cross_validation: ")
error(prediction,y_test)
assert errorKfold(prediction,y_test)'''










