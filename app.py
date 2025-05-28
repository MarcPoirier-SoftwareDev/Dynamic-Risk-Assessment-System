from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics  # Assuming this contains diagnostics-related functions
# import predict_exited_from_saved_model  # Assuming this contains the prediction function
# import create_prediction_model  # Assuming this contains model creation utilities (if needed)
import scoring

###################### Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

####################### Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():        
    """
    Prediction endpoint that takes a dataset file location as input and returns model predictions.
    Expects a JSON body with 'file_path' key pointing to the dataset.
    """
    if request.method == 'POST':
        # Get the file path from the request
        file_path = request.json.get('file_path')
        
        # Load the dataset from the provided file path
        data = pd.read_csv(file_path).iloc[:, 1:]
        
        # Assuming the last column is the target, drop it for predictions
        X = data.iloc[:, :-1]
        
        # Call the prediction function from predict_exited_from_saved_model.py
        #predictions = predict_exited_from_saved_model.predict(X).tolist()  # Assuming predict() function exists
        predictions = diagnostics.model_predictions()
        
        return jsonify(predictions), 200

####################### Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def stats():        
    """
    Scoring endpoint that checks the score of the deployed model and returns the F1 score.
    """
    # Assuming diagnosis.py has a scoring function or similar (adjust based on actual implementation)
    f1_score = scoring.score_model()  # Replace with actual scoring function if different
    
    return jsonify({"f1_score": f1_score}), 200

####################### Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary_stats():        
    """
    Summary statistics endpoint that returns mean, median, and standard deviation for each numeric column.
    """
    # Call the summary statistics function from diagnosis.py
    summary_statistics = diagnosis.dataframe_summary()  # Assuming this returns a list of stats
    
    return jsonify(summary_statistics), 200

####################### Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():        
    """
    Diagnostics endpoint that returns execution timing, percent NA values, and dependency checks.
    """
    # Call diagnostic functions from diagnosis.py
    timings = diagnosis.execution_time()  # Timing of execution
    na_percentages = diagnosis.dataframe_summary()  # Assuming this includes NA percentages
    outdated_packages = diagnosis.outdated_packages_list()  # Dependency checks
    
    diagnostics_data = {
        "timings": timings,
        "na_percentages": na_percentages,
        "outdated_packages": outdated_packages
    }
    
    return jsonify(diagnostics_data), 200

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)