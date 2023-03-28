'''
Author: Sandeep Pvn
Created Date: 28 March 2021
'''

import sys
import pandas as pd
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        model_path = os.path.join('artifacts', "model.pkl")
        self.model = load_object(model_path)

        preprocessor_path = os.path.join('artifacts', "preprocessor.pkl")
        self.preprocessor = load_object(preprocessor_path)

    def predict(self, features):
        try:
            logging.info("Predicting the results")
            data_scaled = self.preprocessor.transform(features)
            predictions = self.model.predict(data_scaled)
            # Round the predictions to 2 decimal places
            predictions = [round(x, 2) for x in predictions]
            return predictions
        
        except Exception as e:
            logging.info(f"Error while predicting the results: {e}")
            raise CustomException(e, sys)

    
class CustomData:
    '''
    This class is used to load the data from the request and map it to the model
    '''
    def __init__(self, gender: str, race_ethnicity: int, parental_level_of_education: str, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            custom_data_input_df = pd.DataFrame(custom_data_input_dict)
            return custom_data_input_df
        except Exception as e:
            logging.info(f"Error while creating data frame: {e}")
            raise CustomException(e, sys)


        
    