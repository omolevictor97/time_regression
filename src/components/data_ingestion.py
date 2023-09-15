import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split

## Initialize data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "raw.csv")

## create class for data ingestion


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestio has started")
            df = pd.read_csv(os.path.join("notebooks/data", "Final_Train2.csv"))
            logging.info("DataFrame has been read")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train Test Split")
            

        except Exception as e:
            logging.info("Error has occured in DataIngestion and in the initiate_data_ingestion method")
            raise CustomException(e, sys)
