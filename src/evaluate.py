from sklearn.metrics import accuracy_score
from preprocess import y_test
from predict import predictions

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.3f}")