'''
Author: Sandeep Pvn
Created Date: 26 March 2021
'''

import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

# Model Evaluation Libraries
from sklearn.metrics import r2_score

# RandomSearchCV and GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


import pickle
import json

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
    
def load_object(filepath: str):
    '''
    Description: This method is used to load the object from a file
    '''
    logging.info('Loading the object from {}'.format(filepath))
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        logging.info('Error occured while loading the object from {}'.format(filepath))
        raise CustomException(e,sys)
    
def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models: dict, hyperparameter_tuning_path: str) -> dict:
    '''
    Description: This method is used to evaluate the models

    Input: X_train, y_train, X_test, y_test, models
    Output: report of type dict[model_name: {train_r2: float, test_r2: float}]
    '''
    logging.info('Evaluating the models')
    try:
        report = {}
        # Get the hyperparameters
        hyperparameters = json.load(open(hyperparameter_tuning_path))

        for model_name, model in models.items():
            # Get the hyperparameters for the model
            model_hyperparameters = hyperparameters[model_name]
            # Create the model
            search = RandomizedSearchCV(model, model_hyperparameters, n_iter=100, cv=5, verbose=0, random_state=42, n_jobs=-1)
            # search = GridSearchCV(model, model_hyperparameters, cv=5, n_jobs=-1)
            # Fit the model
            # model.fit(X_train, y_train)
            search.fit(X_train, y_train)

            # Get the best model
            model.set_params(**search.best_params_)
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