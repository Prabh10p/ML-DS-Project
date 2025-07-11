import pickle
from flask import Flask,render_template,request
import os
import sys
from src.Pipelines.test_pipeline import Pipeline,CustomData
from src.exception import CustomException

application  =Flask(__name__)

app = application



@app.route("/")
def index():
    return render_template('index.html')

@app.route('/PredictData',methods = ["GET","POST"])
def predicta_data():
    try:
        if request.method == "GET":
            return render_template("home.html")
        
        else:
           
           data = CustomData(
           gender = request.form.get('gender'),
           race_ethnicity = request.form.get('race_ethnicity'),
           parental_level_of_education = request.form.get("parental_level_of_education"),
           lunch = request.form.get("lunch"),
           test_preparation_course = request.form.get("test_preparation_course"),
           reading_score = int(request.form.get("reading_score")),
            writing_score = int(request.form.get("writing_score"))
)


           data_frame = data.DataFrame()
           pred = Pipeline()

           results = pred.MakePipeline(data_frame)
           return render_template("home.html",results=results[0])
             
    except Exception as e:
        raise CustomException(e,sys)
    


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=1000,debug=True)




