import os
import sys
import pandas as pd
import numpy as np
import joblib
import traceback
import gc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from scipy.stats import randint

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_out_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_out_path = model_out_path
        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def _validate_data(self, df, dataset_name):
        """Validate data before processing"""
        if "booking_status" not in df.columns:
            raise ValueError(f"'booking_status' column missing in {dataset_name} data")
            
        if df.isnull().any().any():
            logger.warning(f"NaN values found in {dataset_name}, filling with median")
            df = df.fillna(df.median())
            
        if np.isinf(df.select_dtypes(include=np.number)).any().any():
            logger.warning(f"Infinite values found in {dataset_name}, replacing")
            df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median())
            
        return df

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = self._validate_data(load_data(self.train_path), "training")
            
            logger.info(f"Loading data from {self.test_path}")
            test_df = self._validate_data(load_data(self.test_path), "testing")

            X_train = train_df.drop(columns=["booking_status"]).astype(np.float32)
            y_train = train_df["booking_status"].astype(int)
            
            X_test = test_df.drop(columns=["booking_status"]).astype(np.float32)
            y_test = test_df["booking_status"].astype(int)

            logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
            logger.info(f"Class distribution - Train: {y_train.value_counts()}, Test: {y_test.value_counts()}")

            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Data loading error: {str(e)}\n{traceback.format_exc()}")
            raise CustomException("Data loading failed", sys)

    def train_lgbm(self, X_train, y_train):
        try:
            gc.collect()  # Clean memory before training
            
            logger.info("Initializing LightGBM model")
            lgbm_model = lgb.LGBMClassifier(
                random_state=self.random_search_params["random_state"],
                verbose=-1,
                is_unbalance=True,
                n_jobs=2  # Reduce parallelism if memory issues
            )

            logger.info("Starting hyperparameter tuning")
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=min(3, self.random_search_params["cv"]),  # Fewer folds if small dataset
                n_jobs=1,  # Safer for memory-intensive tasks
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"],
                error_score='raise'  # Show full errors
            )

            random_search.fit(X_train, y_train)
            logger.info("Hyperparameter tuning completed")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f"Best parameters: {best_params}")

            return best_lgbm_model
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}\n{traceback.format_exc()}")
            raise CustomException("Model training failed", sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model performance")
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            
            logger.info(f"Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}\n{traceback.format_exc()}")
            raise CustomException("Model evaluation failed", sys)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_out_path), exist_ok=True)
            joblib.dump(model, self.model_out_path)
            logger.info(f"Model saved to {self.model_out_path}")
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}\n{traceback.format_exc()}")
            raise CustomException("Model saving failed", sys)

    def run(self):
        try:
            with mlflow.start_run():

                logger.info("Starting model training pipeline")

                logger.info("Starting our MLFLOW experimentation")

                logger.info("Logging the training and testing dataset to MLFLOW")

                mlflow.log_artifact(self.train_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_path, artifact_path='datasets')

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(model, X_test, y_test)
                self.save_model(model)

                logger.info("Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_out_path)

                logger.info("Logging the params and metrics to MLFLOW")
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                
                logger.info("Model training completed successfully")
                return metrics
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}\n{traceback.format_exc()}")
            raise CustomException("Training pipeline failed", sys)

if __name__ == "__main__":
    try:
        trainer = ModelTraining(
            PROCESSED_TRAIN_DATA_PATH,
            PROCESSED_TEST_DATA_PATH,
            MODEL_OUTPUT_PATH
        )
        trainer.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)