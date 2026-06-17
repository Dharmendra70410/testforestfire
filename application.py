import pickle

from flask import Flask, request, jsonify, render_template, session, redirect, url_for

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application =Flask(__name__)
app=application
app.secret_key = 'section-29-dev-secret-key'

##import ridge regressor adn standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pk1','rb'))
standard_scaler=pickle.load(open('models/scaler.pk1', 'rb'))

# Prevent caching of pages
@app.after_request
def set_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response



@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Store form data to return to template
        form_data = {
            'Temperature': Temperature,
            'RH': RH,
            'Ws': Ws,
            'Rain': Rain,
            'FFMC': FFMC,
            'DMC': DMC,
            'ISI': ISI,
            'Classes': Classes,
            'Region': Region
        }

        #don't do fit.transform(data leakage) do only transform
        new_data_scaled= standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        results = ridge_model.predict(new_data_scaled)[0]

        session['form_data'] = form_data
        session['results'] = results

        return redirect(url_for('predict_datapoint'))

    form_data = session.get('form_data', {})
    results = session.get('results')

    return render_template('home.html', results=results, form_data=form_data)


@app.route('/resetdata')
def reset_data():
    session.pop('form_data', None)
    session.pop('results', None)
    return redirect(url_for('predict_datapoint'))

if __name__=="__main__":
    app.run(host="0.0.0.0")