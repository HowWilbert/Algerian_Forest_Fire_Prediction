import pickle
from flask import Flask,jsonify,render_template,request
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler



application = Flask(__name__)
app = application

# web app should able to interact with Ridge.pkl and standard scaller.pkl
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':   # when we click predict on the web page, we are redirected here...
        Temperature = float(request.form.get('Temperature'))   
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))     # Fetching the input data from our frontend
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])   #standerdize the input data
        result = ridge_model.predict(new_data_scaled)   # predict the value 
        
        return render_template('home.html',results=result[0])  # return the answer results 
    else:
        return render_template('home.html')  #when we do 5000/predict -> GET method is called

if __name__ == "__main__":
    app.run(host='0.0.0.0')