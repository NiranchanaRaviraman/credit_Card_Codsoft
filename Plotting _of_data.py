import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("creditcard.csv")
pd.options.display.max_columns = None
sns.set(style="white")
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title('Distribution of Class (Fraud:1, Not Fraud:0)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

X=data.drop('Class',axis=1)
y=data['Class']
