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
from  sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


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

def random_forest(X_train,y_train,X_test):   
    model = RandomForestRegressor(n_estimators=300,max_depth=50)
    model.fit(X_train, y_train)
    print("X_train")
    print(X_train)
    print(model.feature_importances_)

    parameters ={'learning_rate': sp_randFloat(),
             'subsample': sp_randFloat(),
             'n_estimators': sp_randInt(100,1000),
             'max_depth': sp_randInt(4,10),


    }
    randm_src = RandomizedSearchCV(estimator=model,param_distributions = parameters,cv =2,n_iter = 10,n_jobs =-1)
    randm_src.fit(X_train,y_train)
    print("Results from Random Search")
    print("The best estiamtor acroos ALL searched params:"+str(randm_src.best_estimator_))
    print("The best score acorss ALL searched params"+str(randm_src.best_score_))
    print("RTHe best parameters"+str(randm_src.best_params_))
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




modele = random_forest
conn  = sql.connect("immobilier.db")
data = sample(conn)

print(data.columns)
for e in ["Code_postal","Commune","Code_voie",'Type_de_voie']:
    data[e] = decategoriser(e) 
scaler = StandardScaler() 
#scaler.fit(data)
print(data.columns)
print(data[["Code_postal","Commune","Code_voie"]])
data = pd.DataFrame(data)
data = split_data(data)
#scaler.transform(data)
X_train = data[0] # scaler.fit_transform()
X_test = data[1] #scaler.fit_transform()
y_train = data[2]
y_test = data[3]
prediction = modele(X_train,y_train,X_test)
graph(prediction,y_test)
error(prediction,y_test)
prediction = modele(X_train,y_train,X_train)
print("Sans cross_validation: ")
error(prediction,y_train)

fh = open("modele.ser","wb")
#p = pickle.Pickler(fh,pickle.HIGHEST_PROTOCOL)
#p.dump(modele)
fh.close()










