from flask import Flask,render_template,request,redirect
# from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
# cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=list(map(str,sorted(car['year'].unique(),reverse=True)))
    fuel_type=car['fuel_type'].unique().tolist()

    # companies.insert(0,'Select Company')
    # car_models.insert(0,'Select Car model')
    # year.insert(0,'Select Year')
    # fuel_type.insert(0,'Select Fuel Type')
    return {'companies':companies,'car_models':car_models,'year':year,'fuel_type':fuel_type}


@app.route('/predict',methods=['POST'])
# @cross_origin()
def predict():
    # print(request.form)
    company=request.form.get('company')
    car_model=request.form.get('car_model')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    driven=int(request.form.get('kilos_driven'))

    # print(type(int(year)))
    # return "sadfas"
    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))

    return "<h1>{}</h1>".format(str(np.round(prediction[0],2)))



if __name__=='__main__':
    app.run()