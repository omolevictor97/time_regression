import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass

class DataTransformationConfig:
    preprocessor_file_obj_path:str = os.path.join("artifacts", "preprocessor.pkl")



class DataTransformation:
    
    def __init__(self):
        self.preprocessor = DataTransformationConfig()
    def data_transformation_obj(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e, sys)