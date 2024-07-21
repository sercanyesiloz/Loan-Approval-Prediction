import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = joblib.load('models/model.pkl')
label_encoder_city = joblib.load('models/label_encoder_city.pkl')
label_encoder_education = joblib.load('models/label_encoder_education.pkl')
label_encoder_self_employed = joblib.load('models/label_encoder_self_employed.pkl')

city_cat = ['Ankara', 'Istanbul', 'Izmir', 'Bursa', 'Erzurum']
education_cat = ['Graduate', 'Not Graduate']
self_employed_cat = ['No', 'Yes']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = -1
    if request.method == 'POST':
        no_of_dependents = int(request.form.get('no_of_dependents'))
        income_annum = int(request.form.get('income_annum'))
        loan_amount = int(request.form.get('loan_amount'))
        loan_term = int(request.form.get('loan_term'))
        commercial_assets_value = int(request.form.get('commercial_assets_value'))
        luxury_assets_value = int(request.form.get('luxury_assets_value'))
        bank_asset_value = int(request.form.get('bank_asset_value'))

        cibil_score = float(request.form.get('cibil_score'))
        residential_assets_value = float(request.form.get('residential_assets_value'))

        city = str(request.form.get('city'))
        education = str(request.form.get('education'))
        self_employed = str(request.form.get('self_employed'))

        print(f"no_of_dependents: {no_of_dependents}")
        print(f"city: {city}")
        print(f"education: {education}")
        print(f"self_employed: {self_employed}")
        print(f"income_annum: {income_annum}")
        print(f"loan_amount: {loan_amount}")
        print(f"loan_term: {loan_term}")
        print(f"cibil_score: {cibil_score}")
        print(f"residential_assets_value: {residential_assets_value}")
        print(f"commercial_assets_value: {commercial_assets_value}")
        print(f"luxury_assets_value: {luxury_assets_value}")
        print(f"bank_asset_value: {bank_asset_value}")

        pred_df = pd.DataFrame({
            'no_of_dependents': [no_of_dependents],
            'city': [city],
            'education': [education],
            'self_employed': [self_employed],
            'income_annum': [income_annum],
            'loan_amount': [loan_amount],
            'loan_term': [loan_term],
            'cibil_score': [cibil_score],
            'residential_assets_value': [residential_assets_value],
            'commercial_assets_value': [commercial_assets_value],
            'luxury_assets_value': [luxury_assets_value],
            'bank_asset_value': [bank_asset_value],

        })

        pred_df["city"] = label_encoder_city.transform(pred_df["city"])
        pred_df["education"] = label_encoder_education.transform(pred_df["education"])
        pred_df["self_employed"] = label_encoder_self_employed.transform(pred_df["self_employed"])

        #rejected_df = pd.read_csv("datasets/rejected.csv")
        #approved_df = pd.read_csv("datasets/approved.csv")

        prediction = model.predict_proba(pred_df).tolist()[0]
        prediction[0] = round(prediction[0], 2) * 100
        prediction[1] = round(prediction[1], 2) * 100
        print(prediction)
        
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)