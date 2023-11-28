from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model using pickle
# with open("./gradient_boosting_regressor_model.pkl", 'rb') as file:
model = pd.read_pickle('gradient_boosting_regressor_model.pkl')

data = pd.read_csv('./clean_data.csv')
data.head()

@app.route('/')
def index():
    sex = sorted(data['sex'].unique())
    smoker = sorted(data['smoker'].unique())
    region = sorted(data['region'].unique())
    return render_template('try.html', sex=sex, smoker=smoker, region=region)

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    # Use the model for prediction
    prediction = model.predict(pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                             columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))

    return str(prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
