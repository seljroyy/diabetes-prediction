import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/diabetes.csv")

x = df.drop("Outcome", axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt

df.hist(figsize=(12, 8))
plt.savefig('results/confusion_matrix.png')  # saves it instead of just showing
plt.show()