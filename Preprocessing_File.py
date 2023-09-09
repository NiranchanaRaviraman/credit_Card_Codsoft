import pandas as pd
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("creditcard.csv")
pd.options.display.max_columns = None


sc = StandardScaler()
amount = data['Amount'].values.reshape(-1, 1)
scaled_amount = sc.fit_transform(amount)
data['Amount'] = scaled_amount

print(data.head())

# data.shape()

data.duplicated().any()

Class=data['Class'].value_counts();
print(Class)


data1=sns.countplot(data['Class'])
print(data1)
