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