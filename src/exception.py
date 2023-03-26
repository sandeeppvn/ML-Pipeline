'''
Author: Sandeep Pvn
Created Date: 25 March 2021
'''

import sys
import logging
import logger

def error_message_details(error, error_detail):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error at script: [{0}], line: [{1}], error: [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message