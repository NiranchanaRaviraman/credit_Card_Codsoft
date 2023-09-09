from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("creditcard.csv")
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)
y_pred = log.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
