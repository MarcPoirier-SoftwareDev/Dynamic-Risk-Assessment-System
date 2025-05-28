import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions  # Assuming diagnostics.py is in the same directory

############### Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

############## Function for reporting
def score_model():
    """
    Calculate a confusion matrix using the test data and the deployed model,
    then save the confusion matrix plot to the workspace.
    """
    # Load the test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv')).iloc[:, 1:]
    
    # Assume the last column is the target variable (actual values)
    y_true = test_data.iloc[:, -1]
    
    # Get predictions using the model_predictions function from diagnostics.py
    y_pred = model_predictions()
    
    # Calculate the confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save the plot to the output_model_path directory
    plot_path = os.path.join(output_model_path, 'confusionmatrix.png')
    plt.savefig(plot_path)
    plt.close()

if __name__ == '__main__':
    score_model()