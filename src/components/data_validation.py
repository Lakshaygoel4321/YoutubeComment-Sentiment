import sys
import pandas as pd
from pandas import DataFrame

from src.exception import USvisaException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            expected_columns = self._schema_config["columns"]
            status = len(dataframe.columns) == len(expected_columns)
            logging.info(f"Required column count correct: {status}")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            required_columns = [col for col in self._schema_config["columns"]]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logging.info(f"Missing columns: {missing_columns}")
            return not missing_columns
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            if not self.validate_number_of_columns(train_df):
                validation_error_msg += "Training dataframe column mismatch. "
            if not self.validate_number_of_columns(test_df):
                validation_error_msg += "Testing dataframe column mismatch. "

            if not self.is_column_exist(train_df):
                validation_error_msg += "Training dataframe missing columns. "
            if not self.is_column_exist(test_df):
                validation_error_msg += "Testing dataframe missing columns. "

            validation_status = len(validation_error_msg.strip()) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg.strip(),
                drift_report_file_path=""  # Empty since we removed drift detection
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys)