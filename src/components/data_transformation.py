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
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_file_obj_path:str = os.path.join("artifacts", "preprocessor.pkl")



class DataTransformation:
    
    def __init__(self):
        self.preprocessor = DataTransformationConfig()
    
    # Data Transformation Method

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

            logging.info("Data Transformation Completed")
            return preprocessor
            # Transformation Completed

        except Exception as e:
            logging.info("An Error has occured data_transformation method")
            raise CustomException(e, sys)
    
    # Initiate Data Transformation method

    def initate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading in train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading completed")
            logging.info(f"Training data : \n {train_df.head(20)}")
            logging.info(f"Testing data \n {test_df.head(20)}")
            
            target_column = "Time_taken (min)"
            drop_columns = [target_column]
            # collecting training dataset

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[drop_columns]

            # collecting testing dataset

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[drop_columns]

            # Transforming using the preprocessing object

            preprocessor_obj = self.data_transformation_obj()

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr =np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path = self.preprocessor.preprocessor_file_obj_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.preprocessor.preprocessor_file_obj_path
            )


        except Exception as e:
            logging.info("An error has occured at initiate_data_transformation")
            raise CustomException(e, sys)