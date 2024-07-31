from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
#loading the model and preprocessor files from disk using joblib library which is used to load python
rfr=pickle.load(open('rfr.pkl','rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))

app=Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])  #form action="/predict" method="
def predict():
    if request.method=='POST':
        State_Name	=request.form['State_Name']
        Crop_Type =request.form['Crop_Type']
        Crop =request.form['Crop']
        N = request.form['N']
        P = request.form['P']
        K = request.form['K']
        pH = request.form['pH']
        rainfall	= request.form['rainfall']
        temperature = request.form['temperature']
        Area_in_hectares=request.form['Area_in_hectares']
        features =np.array([[ State_Name,Crop_Type,Crop,N,P,K,pH,rainfall,temperature,Area_in_hectares]])
        transformed_features =preprocessor.transform(features)
        predicted_value =rfr.predict(transformed_features).reshape(1,-1)
        
        return render_template('index.html',predicted_value=predicted_value)

if __name__=='__main__':
    app.run(debug = True)
