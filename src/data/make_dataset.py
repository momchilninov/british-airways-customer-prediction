import os
import pandas as pd


PATH = "data/raw/BA_reviews.csv"

def read_csv(relative_path):
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the data directory
    data_directory = os.path.join(script_directory, "../..", relative_path)

    df = pd.read_csv(data_directory)
    
    columns_to_drop = [col for col in df.columns if 'Unnamed' in col]
    df.drop(columns=columns_to_drop, inplace=True)

    df = df.copy()
    return df

# Call the function to read the CSV files
BA_reviews_df = read_csv(PATH)

BA_reviews_df