from flask import Flask, render_template, request, Response
from flask import jsonify
import pickle
import pandas as pd
import numpy as npy
import sklearn
from sklearn.preprocessing import StandardScaler
import os
class Apps():
    def predicted(tpbs,srb,np,st,nd,app,loc,mai,ter):
        model = pickle.load(open(os.path.join(os.path.dirname(__file__),'Pricing_prediction.pkl'), 'rb'))
        scalX = pickle.load(open(os.path.join(os.path.dirname(__file__),'ScalX.pkl'), 'rb'))
        scaly = pickle.load(open(os.path.join(os.path.dirname(__file__),'Scaly.pkl'), 'rb'))
        df = pd.read_csv(os.path.join(os.path.dirname(__file__),"data/df_final.csv"))
        df.drop(df[df.local == 'mix'].index, inplace=True)
        
        dfl= df[['local']]
        result=[]
        for col,i in zip(dfl, dfl.columns):
            m = pd.Series([i])
            mu = pd.DataFrame(npy.sort(dfl[col].unique()),columns=m)
            result.append(mu)
        result
      
            
        rs = result[0] == tpbs
        rsv = rs.astype(int)
        print(rsv)
        ft = rsv['local'].tolist()
        ft
        print(ft)
        cats = [elem for elem in ft]
        print(cats)
        cat = cats 
        num = [app,loc,mai,ter,nd,np,srb,st]  
        print(num)
        df_num = pd.Series(num)
        df_cat = pd.DataFrame(cat)
        numscal = pd.DataFrame(scalX.transform(df_num.values.reshape(1, -1)))
        #numscal = pd.DataFrame(scalX.transform(df_num.values))
        entree = pd.concat([df_cat.T, numscal],axis=1).values
        print(entree)
        prop = model.predict(entree)
        Our_estimation = scaly.inverse_transform(prop[0]).round(2)
        print(Our_estimation)
        resp = pd.DataFrame(Our_estimation)
        if Our_estimation < 0:
            return render_template('index.html', prediction_text="Sorry !!!")
        else:
            return Response(resp.to_json(orient="records"), mimetype='application/json')