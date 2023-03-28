'''
Author: Sandeep Pvn
Created Date: 28 March 2021
Description: This file contains the code a flask app
'''

import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

import sys
from src.exception import CustomException
from src.logger import logging

appl = Flask(__name__)

app = appl

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'GET':
            logging.info('GET request received')
            return render_template('predict.html')
        else:
            logging.info('POST request received')
            data = CustomData(
                gender = request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=request.form.get('reading_score'),
                writing_score=request.form.get('writing_score')
            )

            data_df = data.get_data_frame()
            logging.info(f"Data frame: {data_df}")

            pipeline = PredictPipeline()
            predicton = pipeline.predict(data_df)
            logging.info(f"Prediction: {predicton}")

            return render_template('predict.html', prediction=predicton[0])

    except Exception as e:
        logging.info(f"Error while creating data frame: {e}")
        raise CustomException(e, sys)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
