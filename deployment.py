# Import required libraries
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil

# Load configuration from config.json and set path variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']  # Path where ingestedfiles.txt is stored
output_model_path = config['output_model_path']    # Path where trainedmodel.pkl and latestscore.txt are stored
prod_deployment_path = config['prod_deployment_path']  # Target deployment directory

# Function to deploy files to the production directory
def store_model_into_pickle(model):
    """
    Copy the latest pickle file (trainedmodel.pkl), the latestscore.txt value,
    and the ingestedfiles.txt file into the deployment directory.
    
    Args:
        model: Parameter included in the template (not used in this implementation).
    """
    # Ensure the deployment directory exists
    os.makedirs(prod_deployment_path, exist_ok=True)
    
    # Copy trainedmodel.pkl from output_model_path to prod_deployment_path
    shutil.copy(os.path.join(output_model_path, 'trainedmodel.pkl'), prod_deployment_path)
    
    # Copy latestscore.txt from output_model_path to prod_deployment_path
    shutil.copy(os.path.join(output_model_path, 'latestscore.txt'), prod_deployment_path)
    
    # Copy ingestedfiles.txt from output_folder_path to prod_deployment_path
    shutil.copy(os.path.join(output_folder_path, 'ingestedfiles.txt'), prod_deployment_path)

# Optional: Add a main block to test the function (not required by the task)
if __name__ == '__main__':
    # This is just a placeholder; the function is typically called from another script
    store_model_into_pickle(None)  # Model parameter is not used