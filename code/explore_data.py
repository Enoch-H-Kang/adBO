import pandas as pd
import os

def load_and_inspect_parquet(file_path):
    """
    Loads a Parquet file (expected to be a chunk of persona data)
    and prints some basic information using pandas.
    """
    try:
        # engine='pyarrow' is common, but pandas might pick it up automatically
        # if installed. You can also try engine='fastparquet' if you installed that.
        df = pd.read_parquet(file_path) 

        print(f"Successfully loaded: {file_path}")

        print("\n--- DataFrame Info ---")
        df.info() # Provides data types and memory usage

        print(f"\n\n--- First 5 Rows (df.head()) ---")
        # To display all columns if there are many, you might adjust pandas display options
        # pd.set_option('display.max_columns', None) # Uncomment if needed
        print(df.head())

        #print the entire content of the first row, column by column
        #print(f"\n\n--- First Row Details ---")
        #first_row = df.iloc[0]
        #print(df.iloc[0]['persona_json'])  
  
        print(f"\n\n--- Column Names ---")
        print(df.columns.tolist())


        print("\n--- Further Exploration ---")
        print("To understand the specific content of each column, you may need to:")
        print("1. Refer to the original paper, especially Table 1 (list of questions) and Appendices.")
        print("2. Examine the data dictionary or metadata if provided with the dataset on Hugging Face.")
        print("Each row in this DataFrame likely represents a single data point (e.g., one answer to one question by one participant) or a consolidated record per participant per question.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please ensure the path is correct. It's based on your VS Code screenshot.")
    except ImportError:
        print("Error: pandas or pyarrow (or fastparquet) might not be installed correctly.")
        print("Please ensure they are installed: pip3 install pandas pyarrow --user")
    except Exception as e:
        print(f"An unexpected error occurred while reading the Parquet file: {e}")

if __name__ == "__main__":
    print("Twin-2K-500 Dataset Explorer (Parquet Edition for VS Code)\n")

    # This path is based on your VS Code screenshot.
    # We'll try to load the first chunk.
    # The script is in 'code/', so the path to data is '../data/'.
    file_to_explore = "../data/Twin-2K-500/full_persona/chunks/persona_chunk_001.parquet"

    print(f"Attempting to load: {file_to_explore}")
    print("This script assumes it's located in the 'code' directory, and the 'data' directory is parallel to 'code'.\n")

    load_and_inspect_parquet(file_to_explore)

