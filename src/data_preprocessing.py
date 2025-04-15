import os
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml, load_data

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        try:
            logger.info("Starting our Data Processing Step")
            
            # Drop unnecessary columns
            logger.info("Dropping the columns")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True, errors='ignore')
            
            # Get column lists from config
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]
            
            # Label Encoding
            logger.info("Applying label encoding")
            label_mappings = {}
            
            for col in cat_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            
            logger.info("Label mappings are:")
            for col, col_mapping in label_mappings.items():
                logger.info(f"{col} : {col_mapping}")
            
            # Skewness handling
            logger.info("Doing skewness handling")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x: x.skew())
            
            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])
            
            return df, label_mappings
            
        except Exception as e:
            logger.error(f"Error during preprocess step: {str(e)}")
            raise CustomException("Error while preprocessing data", sys)

    def balance_data(self, df):
        try:
            logger.info("Handling Imbalanced data")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled
            
            logger.info("Data Balanced Successfully")
            return balanced_df
            
        except Exception as e:
            logger.error(f"Error during balancing data step: {str(e)}")
            raise CustomException("Error while balancing data", sys)

    def feature_selection(self, df):
        try:
            logger.info("Starting our Feature selection process")
            
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance
            })
            
            # Sort features by importance
            sorted_features = feature_importance_df.sort_values(by='Importance', ascending=False)
            num_feature_to_select = self.config["data_processing"]["no_of_features"]
            
            # Select top features
            top_features = sorted_features['Feature'].head(num_feature_to_select).tolist()
            logger.info(f"Features selected: {top_features}")
            
            # Return dataframe with selected features
            selected_df = df[top_features + ['booking_status']]
            logger.info("Feature selection completed successfully")
            
            return selected_df
            
        except Exception as e:
            logger.error(f"Error during features selection step: {str(e)}")
            raise CustomException("Error while selecting features", sys)

    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving our data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}")
            
        except Exception as e:
            logger.error(f"Error during data saving: {str(e)}")
            raise CustomException("Error while saving data", sys)

    def process(self):
        try:
            logger.info("Loading Data from raw directory")
            
            # Load data
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            
            # Preprocess data
            train_df, label_mappings = self.preprocess_data(train_df)
            test_df, _ = self.preprocess_data(test_df)
            
            # Balance data
            train_df = self.balance_data(train_df)
            
            # Feature selection
            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]
            
            # Save processed data
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)
            
            logger.info("Data Processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline: {str(e)}")
            raise CustomException("Error while data preprocessing pipeline", sys)

if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()