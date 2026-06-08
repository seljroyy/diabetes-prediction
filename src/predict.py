import joblib
from preprocess import x_test

model = joblib.load('../models/logistic_regression.pkl')

predictions = model.predict(x_test)