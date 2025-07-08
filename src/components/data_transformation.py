import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from src.utils import save_object

@dataclass
class DataConfig:
    obj_file  = os.path.join("datasets","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_config = DataConfig()

    def transformation(self):
     try:
       num_features = ["reading_score","writing_score"]
       cat_features = ["gender","race_ethnicity","parental_level_of_education",
                      "lunch","test_preparation_course"]
       
       num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

       cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first")),
            ("scaler", StandardScaler(with_mean=False))  # with_mean=False for sparse data
        ])

       
       logging.info(f"Numerical encoding and scaling done :{num_features}")
       logging.info(f"Categorical encoding and clealing done :{cat_features}")
       
       preprocessor = ColumnTransformer(transformers=[
    ("numerical", num_pipeline, num_features),
    ("categorical", cat_pipeline, cat_features)
])
       return preprocessor
     except Exception as e:
        raise CustomException(e,sys)
     

    def initiate_transformation(self,train_path,test_path):
       try:
          train_path = pd.read_csv("datasets/train.csv")
          test_path  = pd.read_csv("datasets/test.csv")
          logging.info("Training data loaded succesfully")
          logging.info("Testing Dta loaded succesfully")

          tranform_obj = self.transformation()
          target_column = "math_score"
          X_train_obj = train_path.drop(target_column,axis=1)
          y_train_obj =  train_path[target_column]


          X_test_obj = test_path.drop(target_column,axis=1)
          y_test_obj = test_path[target_column]


          logging.info("X,y column created")



          X_train  = tranform_obj.fit_transform(X_train_obj)
          X_test = tranform_obj.transform(X_test_obj)



          X_train_array = np.c_[np.array(X_train), y_train_obj]
          X_test_array = np.c_[np.array(X_test), y_test_obj]


          save_object(
            file_path = self.data_config.obj_file,
            obj = tranform_obj
          )
          logging.info("Train test transfomration cokmpleted")

          return (X_train_array,X_test_array,self.data_config.obj_file)

        

     
       except Exception as e:
            raise CustomException(e,sys)

