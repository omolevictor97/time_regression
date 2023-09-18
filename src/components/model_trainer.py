import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
import pandas as pd
import numpy as np
from src.utils import save_obj, evaluate_model


@dataclass

class ModelTrainerConfig:
    training_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_config(self, train_arr, test_arr):
        try:
            X_train, y_train, X_test, y_test = (
                train_arr.drop("Time_taken (min)"),
                train_arr["Time_taken (min)"],
                test_arr.drop("Time_taken (min)"),
                test_arr["Time_taken (min)"]
            )

            models = {
                "Lasso_Regression" : Lasso(),
                "Linear_Regression" : LinearRegression(),
                "Ridge_Regression" : Ridge(),
                "Elastic_Net" : ElasticNet()
            }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            save_obj(
                self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info("An Error has occured at ModelTrainer method")
            raise CustomException(e, sys)