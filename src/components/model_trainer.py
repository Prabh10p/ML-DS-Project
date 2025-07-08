import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRFRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object,evaluate_object

@dataclass
class Modeltrainerconfig:
    model_trainer_file_path = os.path.join("datasets","model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_file = Modeltrainerconfig()

    def inititate_model_training(self,train_arr,test_arr):
           try:
                
            logging.info("Model training started")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]


            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "XGBRFRegressor": XGBRFRegressor()
            }

            
            params = {
    "DecisionTreeRegressor": {
        "max_depth": [10, 20, 40, 50],
        "splitter": ['best', 'random'],
        "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    },

    "RandomForestRegressor": {
        "n_estimators": [10, 50, 60, 80, 100],
        "max_depth": [10, 20, 40, 50],
        "bootstrap": [True, False],
        "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    },

    "AdaBoostRegressor": {
        "n_estimators": [10, 50, 60, 80, 100],
        "learning_rate": [0.01, 0.001, 0.1, 0.5, 1.0]
        # Note: No 'loss' param for regression
    },

    "GradientBoostingRegressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 10]
        # No 'splitter' or 'criterion' for GradientBoostingRegressor
    },

    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ['uniform', 'distance'],
        "algorithm": ['auto', 'ball_tree', 'kd_tree']
        # max_depth/splitter/criterion are invalid here
    },

    "CatBoostRegressor": {
        "depth": [4, 6, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "iterations": [100, 200]
        # No splitter/criterion
    },

    "XGBRFRegressor": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 10],
        "learning_rate": [0.01, 0.1]
        # No splitter/criterion
    }
}


            model_list:dict = evaluate_object(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models,
                                        param=params)
            
            best_model_name = max(model_list, key=lambda k: model_list[k][1])
            best_model, best_score = model_list[best_model_name]
            logging.info("Best model found on training and test data")



            save_object(
                file_path=self.model_file.model_trainer_file_path,
                obj=best_model
            )


            
            
            



           except Exception as e:
               raise CustomException(e,sys)