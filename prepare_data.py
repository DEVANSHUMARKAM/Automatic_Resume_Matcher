import pandas as pd
import os

# --- Configuration ---
# IMPORTANT: Make sure this filename exactly matches the CSV file you downloaded.
CSV_FILE = 'UpdatedResumeDataSet.csv' 
OUTPUT_DIR = 'resumes'

def create_text_files():
    """
    Reads a CSV file containing resumes and creates a separate .txt file
    for each resume in the specified output directory.
    """
    # --- Pre-requisite: Install pandas ---
    # Before running, open your terminal with venv activated and run:
    # pip install pandas
    
    # Create the resumes directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        # Load the dataset
        print(f"Reading data from '{CSV_FILE}'...")
        df = pd.read_csv(CSV_FILE)

        # Check for the required columns
        if 'Category' not in df.columns or 'Resume' not in df.columns:
            print("Error: The CSV file must contain 'Category' and 'Resume' columns.")
            return

        # Loop through each row in the dataframe
        for index, row in df.iterrows():
            # Sanitize the category name to be a valid filename component
            category = str(row['Category']).replace(' ', '_').replace('/', '_')
            content = row['Resume']
            
            # Create a unique filename like "Data_Science_0.txt"
            filename = os.path.join(OUTPUT_DIR, f"{category}_{index}.txt")
            
            # Write the resume content to a .txt file
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(str(content))

        print(f"Successfully created {len(df)} .txt files in the '{OUTPUT_DIR}' folder.")

    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE}' was not found in this directory.")
        print("Please make sure you have downloaded the dataset and placed it in the same folder as this script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- This block runs only when you execute this script directly ---
if __name__ == '__main__':
    create_text_files()