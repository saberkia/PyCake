from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.ExcelFile('Data of khakzad.xlsx', engine='openpyxl').parse(1)
X = data.drop("Degradation(%)", axis=1).values
y = data['Degradation(%)'].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)
print(X_train.shape)
print(y_train.shape)
column_idx = 1
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(reg.score(X_test, y_test))
plt.scatter(X_train[:,column_idx], y_train)
plt.ylabel(data.columns[-1])
plt.xlabel(data.columns[column_idx])
plt.show()
print(y_pred,'\n\n', y_test)
