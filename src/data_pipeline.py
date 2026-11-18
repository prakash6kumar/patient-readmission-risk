import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReadmissionDataPipeline:
    def __init__(self, raw_path: str, processed_path: str):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads raw data from the specified path into a Dataframe.
        """
        logging.info(f"Loading raw data from {self.raw_path}...")
        return pd.read_csv(
            self.raw_path,
            na_values='?',
            dtype={'payer_code': str}
        )

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops features with >40% missing values and drops invalid patient rows.
        """
        logging.info("Handling missing values...")

        # Drop columns with no value
        cols_to_drop = ['weight', 'medical_specialty', 'payer_code']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Drop rows with critical info missing for demographics
        df = df.dropna(subset=['race', 'gender'])
        df = df[df['gender'] != 'Unknown/Invalid']

        # Clean up rows with many missing values but are critical
        df['max_glu_serum'] = df['max_glu_serum'].fillna('Not Tested')
        df['A1Cresult'] = df['A1Cresult'].fillna('Not Tested')

        return df

    def binarize_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the multi-class target into a binary target.
        """
        logging.info("Binarizing target variable...")

        # Map '<30' to 1, everything else to 0
        df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

        # Drop the original target column to prevent data leakage
        df = df.drop(columns=['readmitted'])
        return df

    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles basic ID dropping and categorical alignment.
        """
        logging.info("Applying basic pre-processing...")
        
        # Dropping Encounter ID as it has no predictive value.
        df = df.drop(columns=['encounter_id'])
        
        return df

    def run_pipeline(self):
        """
        Runs the pipelines and saves the processed data.
        """
        logging.info("Starting data pipeline...")

        df = self.load_data()
        df = self.handle_missing_values(df)
        df = self.binarize_target(df)
        df = self.basic_preprocessing(df)

        logging.info(f"Saving processed data to {self.processed_path}...")
        df.to_csv(self.processed_path, index=False)
        logging.info(f"Pipeline complete. Final shape: {df.shape}")  

if __name__ == "__main__":
    RAW_DATA_FILE = "../data/raw/diabetic_data.csv"
    PROCESSED_DATA_FILE = "../data/processed/clean_diabetic_data.csv"

    pipeline = ReadmissionDataPipeline(RAW_DATA_FILE, PROCESSED_DATA_FILE)
    pipeline.run_pipeline()
