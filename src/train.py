import joblib
from sklearn.linear_model import LogisticRegression
from preprocess import x_train, y_train

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

joblib.dump(model, '../models/logistic_regression.pkl')