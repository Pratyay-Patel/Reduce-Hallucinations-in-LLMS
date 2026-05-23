import pandas as pd
import json

def process_nemo_output():
    input_file = '/Users/sagar/Desktop/projects/Ecoprompt_eval/Classification_model/Data/Llama3.1acc_dataset.csv'
    output_file = '/Users/sagar/Desktop/projects/Ecoprompt_eval/Classification_model/Data/Llama3.1acc_dataset_split.csv'
    
    print(f"Reading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Clean up any purely empty columns caused by trailing commas in CSV
    df.dropna(how='all', axis=1, inplace=True)
    
    # Check if 'nemo_raw_output' is in columns
    if 'nemo_raw_output' not in df.columns:
        print("Column 'nemo_raw_output' not found. Exiting.")
        return
    
    print("Parsing JSON strings in 'nemo_raw_output'...")
    def extract_json(val):
        try:
            parsed = json.loads(val)
            # Extracted values are mostly single-item lists (e.g. ["Open QA"]), 
            # we extract the first item for cleaner columns
            return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError):
            return {}
            
    parsed_columns = df['nemo_raw_output'].apply(extract_json)
    parsed_df = pd.json_normalize(parsed_columns)
    
    print(f"Extracted {len(parsed_df.columns)} new columns.")
    
    # Concatenate original dataframe (excluding the raw output) with the new parsed columns
    df_final = pd.concat([df.drop('nemo_raw_output', axis=1), parsed_df], axis=1)
    
    print(f"Saving processed data to {output_file}...")
    df_final.to_csv(output_file, index=False)
    print("Processing complete!")

if __name__ == "__main__":
    process_nemo_output()
