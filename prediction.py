import pandas as pd
import sqlite3 as sql
from  sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def sample(conn):
    data = pd.read_sql_query("SELECT * FROM Data",conn)
    data = data.sample(n = 10000,random_state=1)
    return data
  
def split_data(data):
    #data = data.sort_values(by = "Valeur_fonciere")
    y =  data["Valeur_fonciere"].values
    X =  data["Valeur_fonciere"].values.reshape(-1,1) #.drop(columns= ( ["Valeur_fonciere"]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=15)
    return X_train, X_test, y_train, y_test

def random_forest(X_train,y_train,X_test):   
    model = RandomForestRegressor(n_estimators=300,max_depth=50)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def decision_tree(X_train,y_train,X_test):   
    model = RandomForestRegressor(n_estimators=1,max_depth=100)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def elastic_net(X_train,y_train,X_test):   
    regr = ElasticNet(random_state=0)
    regr.fit(X_train, y_train)
    return regr.predict(X_test)
    

def sgd(X_train,y_train,X_test):   
    reg = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3))
    clf = SGDRegressor()
    clf.fit(X_train, y_train) 
    #print("CLF_SCORE:"+str(clf.score(X_train,y_train)))
    return clf.predict(X_test)


def graph(predictions,y_test):
   # fig, ax = plt.subplots()
   # ax.scatter(predictions,y_test,marker='o',color="red")

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





conn  = sql.connect("immobilier.db")

data = sample(conn)
encoder = OrdinalEncoder()
data[["Code_postal","Commune","Type_de_voie","Code_voie"]] = encoder.fit_transform(data[["Code_postal","Commune","Type_de_voie","Code_voie"]],data["Valeur_fonciere"] ) 
data = split_data(data)
X_train = data[0]
X_test = data[1]
y_train = data[2]
y_test = data[3]
#prediction = random_forest(X_train,y_train,X_test)
prediction = sgd(X_train,y_train,X_test)
print("RATIO: " + str(prediction[1]/y_test[1] ))
graph(prediction,y_test)
error(prediction,y_test)