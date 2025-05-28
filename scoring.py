from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path'])

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Load test data
    test_file_path = os.path.join(test_data_path, 'testdata.csv')
    test_df = pd.read_csv(test_file_path).iloc[:, 1:]
    
    # Assume the last column is the target
    target_column = test_df.columns[-1]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]
    
    # Load the trained model
    model_file_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions and calculate F1 score
    y_pred = model.predict(X_test)
    f1 = metrics.f1_score(y_test, y_pred)
    
    # Write the F1 score to latestscore.txt
    os.makedirs(output_model_path, exist_ok=True)
    score_file_path = os.path.join(output_model_path, 'latestscore.txt')
    with open(score_file_path, 'w') as f:
        f.write(str(f1))

if __name__ == '__main__':
    score_model()
