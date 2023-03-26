'''
Author: Sandeep Pvn
Created Date: 25 March 2021
'''

import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

# Model Evaluation Libraries
from sklearn.metrics import r2_score

import pickle

def save_object(filepath: str, obj: object):
    '''
    Description: This method is used to save the object to a file

    Input: filepath, obj
    Output: None
    '''
    logging.info('Saving the object to {}'.format(filepath))
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        logging.info('Error occured while saving the object to {}'.format(filepath))
        raise CustomException(e,sys)
    
def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models: dict) -> dict:
    '''
    Description: This method is used to evaluate the models

    Input: X_train, y_train, X_test, y_test, models
    Output: None
    '''
    logging.info('Evaluating the models')
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            report[model_name] = model.score(X_test, y_test)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            report[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2
            }
        return report
    except Exception as e:
        logging.info('Error occured while evaluating the models')
        raise CustomException(e,sys)