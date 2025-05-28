import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    """
    Reads all files from the input folder, compiles them into a single DataFrame,
    removes duplicates, and saves the result to a CSV file. Also records the list
    of ingested files in a text file.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Get list of files in the input folder (only files, not directories)
    file_list = [f for f in os.listdir(input_folder_path) 
                 if os.path.isfile(os.path.join(input_folder_path, f))]
    
    # Read each file into a pandas DataFrame
    dfs = [pd.read_csv(os.path.join(input_folder_path, file)) for file in file_list]
    
    # Compile all DataFrames into a single DataFrame
    if dfs:  # Check if there are any DataFrames to concatenate
        compiled_df = pd.concat(dfs, ignore_index=True)
    else:
        compiled_df = pd.DataFrame()  # Create an empty DataFrame if no files are found
    
    # Remove duplicate rows
    deduped_df = compiled_df.drop_duplicates()
    
    # Save the de-duplicated DataFrame to a CSV file
    deduped_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
    
    # Save the list of ingested files to a text file
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.write('\n'.join(file_list))


if __name__ == '__main__':
    merge_multiple_dataframe()
