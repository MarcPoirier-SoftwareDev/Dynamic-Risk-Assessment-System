import requests
import json
import os

# Specify a URL that resolves to your workspace (Flask app running on port 8000)
URL = "http://127.0.0.1:8000/"

# Load config.json to get the output_model_path
with open('config.json', 'r') as f:
    config = json.load(f)
output_model_path = config['output_model_path']

# Call each API endpoint and store the responses
response1 = requests.post(URL + "prediction", json={"file_path": "/testdata/testdata.csv"})
response2 = requests.get(URL + "scoring")
response3 = requests.get(URL + "summarystats")
response4 = requests.get(URL + "diagnostics")

# Combine all API responses into a dictionary
responses = {
    "prediction": response1.json(),
    "scoring": response2.json(),
    "summarystats": response3.json(),
    "diagnostics": response4.json()
}

# Write the responses to your workspace in output_model_path as apireturns.txt
with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as f:
    json.dump(responses, f, indent=4)



