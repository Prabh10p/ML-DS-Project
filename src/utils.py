import sys
import os
from src.exception import CustomException
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score




def save_object(file_path,obj):
    try:   
       path = os.path.dirname(file_path)
       os.makedirs(path,exist_ok=True)

       with open(file_path,"wb") as file_obj:
              dill.dump(obj,file_obj)

    except Exception as e:
         raise CustomException(e,sys)



def evaluate_object(X_train, y_train, X_test, y_test, models: dict, param: dict):
    try:
        report = {}

        for model_name, model in models.items():
            model_params = param.get(model_name, {})

            gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            # Store model and test score
            report[model_name] = (best_model, test_score)

        return report

    except Exception as e:
        raise CustomException(e, sys)
          
           
          