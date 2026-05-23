import pandas as pd
import argparse

def remove_redundant_rows(file_path):
    print(f"Reading file: {file_path}")
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Store original row count
    original_count = len(df)
    
    # Drop completely duplicate rows
    df_deduped = df.drop_duplicates()
    
    # Calculate how many rows were removed
    removed_count = original_count - len(df_deduped)
    
    # Save the deduplicated dataframe back to the CSV
    df_deduped.to_csv(file_path, index=False)
    
    print(f"Original row count: {original_count}")
    print(f"Removed {removed_count} redundant (duplicate) rows.")
    print(f"New row count: {len(df_deduped)}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove redundant rows from a CSV file.")
    parser.add_argument("--file_path", type=str, default="./scenarios_evaluation/ecoprompt_results_final.csv", help="Path to the CSV file")
    args = parser.parse_args()
    
    remove_redundant_rows(args.file_path)
