from flask import Flask,request, url_for, redirect, render_template, jsonify
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder,PowerTransformer
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

regressor = joblib.load('model.pkl')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = regressor.predict(data_unseen)
    prediction = int(prediction[0])
    return render_template('home.html', pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = regressor.predict(data_unseen)
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run()