from flask import Flask, render_template, jsonify, Response
import pandas as pd
from flask import request
import os
from .apps import Apps
app = Flask(__name__, template_folder='templates')
    # To get one variable, tape app.config['MY_VARIABLE']
class View(Apps):
    @app.route('/')
    def index():
        return render_template('index.html')
    @app.route('/typeVoie')
    def typeVoie():
        return Apps.data()
    @app.route('/zipCode')
    def zipCode():
        return Apps.getZipCode()
    @app.route('/commune/<zipCode>')
    def commune(zipCode):
        return Apps.getCommune(zipCode)
    @app.route('/particulier')
    def particulier():
        return render_template('particular.html')
    @app.route('/professionel')
    def professionel():
        return render_template('professionel.html')
    @app.route('/predict/<tpbs>/<srb>/<np>/<st>/<nd>/<app>/<loc>/<mai>/<ter>')
    def predict(tpbs,srb,np,st,nd,app,loc,mai,ter):
        print(tpbs,srb,nd,st,np,app,loc,mai,ter)
        return Apps.predicted(tpbs,srb,np,st,nd,app,loc,mai,ter)