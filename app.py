
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)  
model  = pickle.load(open('model_XGBR.pkl', 'rb')) 
scalerX = pickle.load(open('scalerX.pkl', 'rb'))
scalery = pickle.load(open('scalery.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
    

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        appartment = int(request.form['appartment'])
        House = int(request.form['House'])
        Commercial = int(request.form['Commercial'])
        Land = int(request.form['Land'])
        dependency = int(request.form['dependency'])
        builtsurface = float(request.form['builtsurface'])
        landsurface = float(request.form['landsurface'])
        Numberofpieces = int(request.form['Numberofpieces'])

        Property = request.form['Property']
        local_seul = 0
        local_batiment= 0
        local_lotissement= 0
        local_centre_commercial= 0
        local_terrain= 0
        local_mix= 0


        if Property== 'alone': 
            local_seul = 1
        elif Property == 'Building' :
            local_batiment= 1           
        elif Property == 'subdivision' :
            local_lotissement= 1
        elif Property == 'mall' :
            local_centre_commercial= 1 
        elif Property == 'Land' :
            local_terrain= 1
        else :
            local_mix= 1

        
        categ = [local_seul, local_batiment, local_lotissement, local_centre_commercial, local_terrain, local_mix]        
        num = [ appartment, House, Commercial, Land, dependency, builtsurface, landsurface, Numberofpieces]
        
        
        
        df_num=pd.DataFrame(num)        
        df_categ=pd.DataFrame(categ)
        #scaler.fit(df_num.values.reshape(-1, 1))
        numscal=pd.DataFrame(scalerX.transform(df_num.values.reshape(1, -1)))
        input=pd.concat([df_categ.T,numscal],axis=1).values
       
        
    
        prediction = model.predict(input)       
        output = abs(scalery.inverse_transform(prediction.reshape(-1, 1)))[0]

        return render_template('index.html', prediction_text='We predict price at  {} '.format(output[0])) 
 

if __name__ == '__main__':
    app.run(debug=True)

