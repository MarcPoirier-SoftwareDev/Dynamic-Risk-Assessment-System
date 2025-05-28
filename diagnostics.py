import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
deployed_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')

################## Function to get model predictions
def model_predictions():
    """
    Read the deployed model and a test dataset, calculate predictions.
    Returns:
        list: A list containing all predictions.
    """
    # Load the test dataset
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv')).iloc[:, 1:]
    
    # Assume the last column is the target, drop it for predictions
    X_test = test_data.iloc[:, :-1]
    
    # Load the deployed model
    with open(deployed_model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return predictions.tolist()

################## Function to get summary statistics
def dataframe_summary():
    """
    Calculate summary statistics (mean, median, standard deviation) and NA percentages for each column.
    Returns:
        list: A list containing summary statistics and NA percentages for each column.
    """
    # Load the dataset
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv')).iloc[:, 1:]
    
    # Select numeric columns for summary statistics
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculate summary statistics
    summary_stats = []
    for column in numeric_data.columns:
        mean = numeric_data[column].mean()
        median = numeric_data[column].median()
        std = numeric_data[column].std()
        na_percentage = (numeric_data[column].isna().sum() / len(numeric_data[column])) * 100
        summary_stats.append({
            'column': column,
            'mean': mean,
            'median': median,
            'std': std,
            'na_percentage': na_percentage
        })
    
    # Calculate NA percentages for non-numeric columns as well
    for column in data.columns:
        if column not in numeric_data.columns:
            na_percentage = (data[column].isna().sum() / len(data[column])) * 100
            summary_stats.append({
                'column': column,
                'na_percentage': na_percentage
            })
    
    return summary_stats

################## Function to get timings
def execution_time():
    """
    Calculate the execution time of training.py and ingestion.py.
    Returns:
        list: A list containing two timing values in seconds for ingestion and training.
    """
    # Measure time for ingestion.py
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start_time
    
    # Measure time for training.py
    start_time = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start_time
    
    return [ingestion_time, training_time]

################## Function to check dependencies
def outdated_packages_list():
    """
    Check for outdated packages and return a list of dependencies with their current and latest versions.
    Returns:
        list: A list of dictionaries with package names, current versions, and latest versions.
    """
    # Run pip list --outdated to get outdated packages
    result = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf-8')
    
    # Parse the output
    lines = result.splitlines()[2:]  # Skip the header lines
    outdated_packages = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            package = parts[0]
            current_version = parts[1]
            latest_version = parts[2]
            outdated_packages.append({
                'package': package,
                'current_version': current_version,
                'latest_version': latest_version
            })
    
    return outdated_packages

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()