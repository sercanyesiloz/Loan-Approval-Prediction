import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

model = joblib.load('models/random_forest_model.pkl')

@app.route('/predict', methods=['GET', 'POST'])
def home():
    prediction = -1
    if request.method == 'POST':
        print(request)

        no_of_dependents = int(request.form.get('no_of_dependents'))
        city = int(request.form.get('city'))
        education = int(request.form.get('education'))
        self_employed = int(request.form.get('self_employed'))
        income_annum = int(request.form.get('income_annum'))
        loan_amount = int(request.form.get('loan_amount'))
        loan_term = int(request.form.get('loan_term'))
        cibil_score = float(request.form.get('cibil_score'))
        residential_assets_value = float(request.form.get('residential_assets_value'))
        commercial_assets_value = int(request.form.get('commercial_assets_value'))
        luxury_assets_value = int(request.form.get('luxury_assets_value'))
        bank_asset_value = int(request.form.get('bank_asset_value'))

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

        input_features = np.array([[
                            no_of_dependents,
                            city,
                            education,
                            self_employed,
                            income_annum,
                            loan_amount,
                            loan_term,
                            cibil_score,
                            residential_assets_value,
                            commercial_assets_value,
                            luxury_assets_value,
                            bank_asset_value,
                            ]])

        prediction = model.predict_proba(input_features).tolist()
        
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)