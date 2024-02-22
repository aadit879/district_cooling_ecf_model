import os
#os.chdir(os.path.join(os.getcwd(), r'cm\app\api_v1\my_calculation_module_directory'))
os.chdir(os.path.join(os.getcwd(), r'cm\app\api_v1'))



import sys
sys.path.append('.')

current_directory = os.getcwd()

# Get the path to the 'api_v1' directory by going one level up
api_v1_directory = os.path.dirname(current_directory)

# Add the 'api_v1' directory to sys.path
sys.path.append(api_v1_directory)

## change the helper and constant import statements in constant.py

import calculation_module
from cm.app import helper
from flask import request

data = request.get_json()

inputs_raster_selection = helper.validateJSON(data["inputs_raster_selection"])


inputs_parameter_selection = helper.validateJSON(data["inputs_parameter_selection"])



inputs_vector_selection = helper.validateJSON(data["inputs_vector_selection"])

root_directory = os.getcwd()
output_directory =  root_directory + '\\Output\\'

result = calculation_module.calculation(output_directory, inputs_raster_selection,inputs_vector_selection,inputs_parameter_selection)