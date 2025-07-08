import pandas as pd
import numpy as np
import os 
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class Pipeline:
    def __init__(self):
        pass
    def MakePipeline(self,features):
     try:
        logging.info("Loading the dataset started")
        model_path = os.path.join("datasets","model.pkl")
        prep_path = os.path.join("datasets","preprocessor.pkl")
        model = load_object(model_path)
        preprocessor = load_object(prep_path)
        logging.info("model loaded succesfully")
        data_scaled = preprocessor.transform(features)
        y_prediction = model.predict(data_scaled)
        return y_prediction
     except Exception as e:
        raise CustomException(e,sys)
    
class CustomData:
    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score
    ):
        self.gender = gender                      # ✅ no comma
        self.race_ethnicity = race_ethnicity      # ✅ no comma
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def DataFrame(self):
     try:
        custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
        

        return pd.DataFrame(custom_data_input_dict)
     except Exception as e :
        raise CustomException(e,sys)


       
    

    

    
