import pandas as pd
import json

print("Reading Llama3.1acc_dataset.csv...")
df = pd.read_csv('Llama3.1acc_dataset.csv')

def parse_nemo(row):
    try:
        if pd.isna(row): return {}
        if isinstance(row, str):
            # Sometimes json format might need strict double quotes, assuming it's valid JSON
            data = json.loads(row)
        else:
            data = row
        return {k: v[0] if isinstance(v, list) and len(v) > 0 else v for k, v in data.items()}
    except Exception as e:
        return {}

print("Parsing nemo_raw_output...")
# We use apply to get dictionaries, then convert directly to DataFrame for performance
parsed_dicts = df['nemo_raw_output'].apply(parse_nemo).tolist()
parsed_df = pd.DataFrame(parsed_dicts)

print("Concatenating parsed columns with original data...")
df = pd.concat([df.drop(columns=['nemo_raw_output', 'nemo_raw_output.1'], errors='ignore'), parsed_df], axis=1)

if 'accuracy_score' in df.columns:
    df = df.rename(columns={'accuracy_score': 'label'})
if 'sample_index' in df.columns:
    df = df.rename(columns={'sample_index': 'id'})

print("Saving to finalllama.csv...")
df.to_csv('finalllama.csv', index=False)
print("Done! Here are the new columns:")
print(df.columns.tolist())
