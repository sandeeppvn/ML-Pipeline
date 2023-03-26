'''
Author: Sandeep Pvn
Created Date: 25 March 2021
'''

import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from dataclasses import dataclass


# Model Libraries
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Model Evaluation Libraries
from sklearn.metrics import r2_score




@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')
    model_report_path = os.path.join('artifacts', 'model_report.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
   
    def train(self, train_arr: np.ndarray, test_arr: np.ndarray, preprocessor_obj: object = None): 
        '''
        Description: This method is used to train the model, it performs the following steps:
            1. Train the model
            2. Evaluate the model
            3. Find the best model
            4. Save the best model
            5. Predict the values using the best model
            6. Return the predicted r2 score

        Input: X_train, X_test, y_train, y_test
        Output: predicted_r2_score, model_report
        '''
        try:
            logging.info('Training the model')
            # Last column is the target feature
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
            logging.info('X_test shape: {}, y_test shape: {}'.format(X_test.shape, y_test.shape))

            # Train the model
            models = {
                'LinearRegression': LinearRegression(),
                'SVR': SVR(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'CatBoostRegressor': CatBoostRegressor(),
                'XGBRegressor': XGBRegressor()
            }

            report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)
            # Save the model report
            save_object(self.model_trainer_config.model_report_path, report)


            # Get the best model score and name from the report based on r2, the report structure is
            # report[model_name] = {
            #     'train_r2': train_r2,
            #     'test_r2': test_r2
            # }
            best_model_name = max(report, key=lambda key: report[key]['test_r2'])
            best_model_score = report[best_model_name]['test_r2']

            if best_model_score < 0.6:
                raise CustomException('The model is not performing well, so not saving the model',sys)

            logging.info('Best model is {} with score {}'.format(best_model_name, best_model_score))
            best_model = models[best_model_name]
            
            # Save the model
            save_object(self.model_trainer_config.trained_model_path, best_model)
            

            predicted_values = best_model.predict(X_test)
            predicted_r2_score = r2_score(y_test, predicted_values)
            logging.info('Predicted r2 score: {}'.format(predicted_r2_score))

            return (predicted_r2_score, report)


        except Exception as e:
            logging.info('Error while training the model')
            raise CustomException(e, sys)


