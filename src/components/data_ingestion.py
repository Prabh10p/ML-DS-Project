import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation,DataConfig


@dataclass
class DataIngestionConfig:
      raw_data_path:str=os.path.join("datasets",'data.csv')
      train_data_path:str=os.path.join("datasets",'train.csv')
      test_datapath:str=os.path.join("datasets",'test.csv')


class DataIngestion:
      def __init__(self):
            self.ingestion = DataIngestionConfig()
      def initiate_ingestion(self):
            logging.info("Loading data into DataFrame Started")
            try:
                df = pd.read_csv("src/notebook/notebook/data/stud.csv")
                logging.info("Data converted to Datasframe succesfully")
                os.makedirs(os.path.dirname(self.ingestion.train_data_path),exist_ok=True)
                df.to_csv(self.ingestion.raw_data_path,index=False,header=True)
                logging.info("Train_test split started")
                train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
               

                train_set.to_csv(self.ingestion.train_data_path,index=False,header=True)

                test_set.to_csv(self.ingestion.test_datapath,index=False,header=True)
                logging.info("Train_test split completed")

                return (self.ingestion.train_data_path,self.ingestion.test_datapath)
            except Exception as e:
                  raise CustomException(e,sys)
            
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_ingestion()
    
    data_transform = DataTransformation()
    data_transform.initiate_transformation(train_data, test_data)



