import pandas as pd
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("creditcard.csv")
pd.options.display.max_columns = None
print("Display 5 rows:")
print(data.head())


print("\nLast 5 rows:")
print(data.tail())

print("\nShape of the data:")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])


print("\nNull values:")
print(data.isnull())


sc = StandardScaler()


amount = data['Amount'].values.reshape(-1, 1)


scaled_amount = sc.fit_transform(amount)


data['Amount'] = scaled_amount
