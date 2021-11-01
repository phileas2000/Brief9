import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv(r'../data/CURATED/df_final.csv', encoding="UTF8")



scalX = MaxAbsScaler()
scaly= MaxAbsScaler()

zz = pd.get_dummies(df['local'])
yy = pd.DataFrame(df['valeur'])
XX = df.drop(['Id','local','valeur', 'tot_lot', 'typ_lot', 'prix_m2'], axis=1)

scaly.fit(yy.values)
scalX.fit(XX)

y = pd.DataFrame(scaly.transform(yy.values))
X = pd.concat([pd.DataFrame(scalX.fit_transform(XX), columns= XX.columns),zz], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


regresseur = RandomForestRegressor(min_samples_leaf=2, min_samples_split=5, n_estimators=1000)
regresseur.fit(X_train,y_train)

pickle.dump(regresseur,open('model.pkl','wb'))
pickle.dump(scalX,open('scalerX.pkl','wb'))
pickle.dump(scaly,open('scalery.pkl','wb'))
