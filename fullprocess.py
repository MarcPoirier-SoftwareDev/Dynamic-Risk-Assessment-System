import os
import json
import subprocess
import pandas as pd
import pickle
from sklearn import metrics

# Load configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']  # Should be set to '/sourcedata/'
prod_deployment_path = config['prod_deployment_path']
output_model_path = config['output_model_path']  # Should be set to '/models/'

# Function to check and read new data
def check_new_data():
    """
    Check for new data in the input folder and ingest it if found.
    Returns True if new data is ingested, False otherwise.
    """
    # Read ingestedfiles.txt from the deployment directory
    ingested_files_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    with open(ingested_files_path, 'r') as f:
        ingested_files = set(f.read().splitlines())
    
    # List all files in the source data folder
    source_files = set(os.listdir(input_folder_path))
    
    # Determine if there are new files not listed in ingestedfiles.txt
    new_files = source_files - ingested_files
    
    if new_files:
        print(f"New files found: {new_files}. Running ingestion.py...")
        subprocess.run(['python', 'ingestion.py'])
        return True
    else:
        print("No new data found.")
        return False

# Function to check for model drift
def check_model_drift():
    """
    Check if model drift has occurred by comparing the latest score with a new score.
    Returns True if drift is detected (new score is lower), False otherwise.
    """
    # Read the latest score from latestscore.txt
    latest_score_path = os.path.join(prod_deployment_path, 'latestscore.txt')
    with open(latest_score_path, 'r') as f:
        latest_score = float(f.read())
    
    # Load the newest ingested data
    newest_data_path = os.path.join(output_model_path, 'finaldata.csv')  # Assumes ingestion.py outputs this
    data = pd.read_csv(newest_data_path)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    
    # Load the deployed model
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions on the new data
    y_pred = model.predict(X)
    
    # Calculate the new score (using F1 score as an example metric)
    new_score = metrics.f1_score(y, y_pred)
    
    # Check for model drift (new score lower than latest score)
    if new_score < latest_score:
        print(f"Model drift detected: New score ({new_score}) < Latest score ({latest_score})")
        return True
    else:
        print(f"No model drift detected: New score ({new_score}) >= Latest score ({latest_score})")
        return False

# Main process function
def main():
    """
    Automate the full ML model monitoring and re-deployment process.
    """
    ################## Check and Read New Data
    print("Checking for new data...")
    if not check_new_data():
        print("No new data found. Ending process.")
        return
    
    ################## Deciding Whether to Proceed (Part 1)
    # If we reach here, new data was found and ingested, so proceed
    
    ################## Checking for Model Drift
    print("Checking for model drift...")
    if not check_model_drift():
        print("No model drift found. Ending process.")
        return
    
    ################## Deciding Whether to Proceed (Part 2)
    # If we reach here, model drift was detected, so proceed
    
    ################## Re-training
    print("Model drift detected. Re-training the model...")
    subprocess.run(['python', 'training.py'])
    
    ################## Re-deployment
    print("Re-deploying the model...")
    subprocess.run(['python', 'deployment.py'])
    
    ################## Diagnostics and Reporting
    print("Running diagnostics and reporting on the re-deployed model...")
    subprocess.run(['python', 'apicalls.py'])
    subprocess.run(['python', 'reporting.py'])
    print("Process completed successfully.")

if __name__ == '__main__':
    main()