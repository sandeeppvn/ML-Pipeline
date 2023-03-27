'''
Author: Sandeep Pvn
Created Date: 25 March 2021
Description: This file contains the code for data ingestion
Source: local, mongodb, s3, etc
'''

import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def ingest(self):
        '''
        Description: This method is used to initiate the data ingestion process. It performs the following steps:
            1. Read the data from source
            2. Split the data into train and test
            3. Save the raw data to artifacts
            4. Save the train and test data to artifacts
            5. Return the train and test data paths

        Input: None
        Output: train_data_path, test_data_path
        '''
        logging.info('Initiating data ingestion')
        try:
            df = pd.read_csv('data/stud.csv')
            logging.info('Data read as a dataframe')

            # Creating directory if not exists
            os.makedirs(os.path.dirname((self.ingestion_config.train_data_path)), exist_ok=True)
            
            # Saving the raw data to artifacts
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved to artifacts')

            # Splitting the data into train and test
            logging.info('Splitting the data into train and test')
            train, test = train_test_split(df, test_size=0.2, random_state=42)

            # Saving the train data to artifacts
            pd.DataFrame(train).to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            

            # Saving the test data to artifacts
            pd.DataFrame(test).to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed successfully')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info('Failed to ingest data')
            raise CustomException(e, sys)


if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.ingest()

    data_transformation = DataTransformation()
    train_data, test_data, preprocessor_obj = data_transformation.transform(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    predcited_r2_score, report = model_trainer.train(train_data, test_data, preprocessor_obj)


    
    # Print the model evaluation report in a readable format
    for model_name, model_report in report.items():
        print('\nModel: {}'.format(model_name))
        for metric_name, metric_value in model_report.items():
            print('{}: {}'.format(metric_name, metric_value))

    print('\nPredicted R2 score: {}'.format(predcited_r2_score))


