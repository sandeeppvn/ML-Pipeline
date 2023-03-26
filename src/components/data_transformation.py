'''
Author: Sandeep Pvn
Created Date: 25 March 2021
Description: This file contains the code for data transformation: feature enigeering, data cleaning, data imputation, data scaling, data encoding, etc.
'''

import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')
    raw_data_path = os.path.join('artifacts', 'raw_data.csv')
    TARGET_FEATURE = 'math_score'

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()        
        df = pd.read_csv(self.data_transformation_config.raw_data_path, nrows=1, header=0)
        self.categorical_features = df.select_dtypes(include=['object']).columns
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        self.numerical_features = self.numerical_features.drop(self.data_transformation_config.TARGET_FEATURE)

    def get_data_transformer_object(self):
        '''
        Description: This mthod is used to get the data transformer object, it performs the following steps:
            1. Create numerical and categorical pipelines
            2. Bundle preprocessing for numerical and categorical data using ColumnTransformer
            3. Return the preprocessor object

        Input: None
        Output: preprocessor object
        '''
        logging.info('Getting data transformer object')
        try:
            logging.info('Numerical features: {}'.format(self.numerical_features))
            logging.info('Categorical features: {}'.format(self.categorical_features))
            logging.info('Target feature: {}'.format(self.data_transformation_config.TARGET_FEATURE))

            # Create numerical and categorical pipelines
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info('Created numerical and categorical pipelines, categorical encoding and scaling')            

            # Bundle preprocessing for numerical and categorical data using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers = [
                    ('numerical_pipeline', numerical_pipeline, self.numerical_features),
                    ('categorical_pipeline', categorical_pipeline, self.categorical_features)
                ]
            )
            logging.info('Created preprocessor object')
            return preprocessor

        except Exception as e:
            logging.info('Error while getting data transformer object')
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_data_path, test_data_path):
        '''
        Description: This method is used to initiate the data transformation process. It performs the following steps:
            1. Get the preprocessor object
            2. Fit the preprocessor object on train data
            3. Transform the train and test data using the preprocessor object
            4. Save the preprocessor object to artifacts as a pickle file
            6. Return the transformed train data, test data and preprocessor object

        Input: train_data_path, test_data_path
        Output: transformed_train_data_path, transformed_test_data_path
        '''
        logging.info('Initiating data transformation')
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Completed reading train and test data')

            # Remove the target feature from data and save it separately
            y_train_df = train_df[self.data_transformation_config.TARGET_FEATURE]
            y_test_df = test_df[self.data_transformation_config.TARGET_FEATURE]

            X_train_df = train_df.drop(self.data_transformation_config.TARGET_FEATURE, axis=1)
            X_test_df = test_df.drop(self.data_transformation_config.TARGET_FEATURE, axis=1)
            logging.info('Created train and test data without target feature')

            # Get the preprocessor object
            preprocessor = self.get_data_transformer_object()

            # Fit the preprocessor object on train data
            preprocessor.fit(X_train_df)
            logging.info('Fitted preprocessor object on train data')

            # Transform the train and test data using the preprocessor object
            transformed_train_df = preprocessor.transform(X_train_df)
            transformed_test_df = preprocessor.transform(X_test_df)
            logging.info('Transformed train and test data using the preprocessor object')

            # Save the preprocessor object to artifacts, use utils save_object method
            save_object(
                self.data_transformation_config.preprocessor_obj_path,
                preprocessor
            )

            # Combine the target feature with the transformed train and test data using np.c_
            transformed_train_df = np.c_[transformed_train_df, y_train_df]
            transformed_test_df = np.c_[transformed_test_df, y_test_df]
            logging.info('Combined the target feature with the transformed train and test data using np.c_')

            # Return the transformed train data, test data and preprocessor object
            return (
                transformed_train_df,
                transformed_test_df,
                preprocessor
            )
            


        except Exception as e:
            logging.info('Error while initiating data transformation')
            raise CustomException(e,sys)


