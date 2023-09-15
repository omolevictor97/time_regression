import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
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
            #Define the columns to be ordinally encoded and numerical columns to be scaled
            logging.info("Data Transformation has begun")
            numerical_col = ['Delivery_person_Age', 'Delivery_person_Ratings',
       'Delivery_location_latitude', 'Delivery_location_longitude',
       'Vehicle_condition', 'multiple_deliveries', 'Order month', 'Order day',
       'Time_ordered_hr', 'Time_ordered_min', 'Time_Order_picked_hr',
       'Time_Order_picked_min']
            categorical_col = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']
            
            Weather_conditions = ['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny']
            Road_traffic = ['Jam', 'High', 'Medium', 'Low']
            Type_of_order = ['Snack', 'Meal', 'Drinks', 'Buffet']
            Type_of_vehicle = ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']
            Festival = ['No', 'Yes']
            City = ['Metropolitian', 'Urban', 'Semi-Urban']

            # Numerical Pipeline

            Numerical_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="median", missing_values=np.nan)),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline

            categorical_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
                    ("Encoder", OrdinalEncoder(categories=[Weather_conditions, Road_traffic,
                                                           Type_of_order,
                                                           Type_of_vehicle,
                                                           Festival,
                                                           City]))
                ]
            )

            preprocessor = ColumnTransformer([
                ("numerical_pipeline", Numerical_pipeline, numerical_col),
                ("categorical_pipeline", categorical_pipeline, categorical_col)
            ])

            # Transformation Completed
            logging.info("Data Transformation Completed")
            #self.preprocessor.preprocessor_file_obj_path
        except Exception as e:
            logging.info()
            raise CustomException(e, sys)