# per capita income Canada
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("canada_per_capita_income.csv")

X = df[['year']]
y = df['per capita income (US$)']

model = LinearRegression()
model.fit(X, y)

prediction_2020 = model.predict([[2020]])

print("Predicted Per Capita Income in 2020:", prediction_2020[0])

plt.scatter(df['year'], df['per capita income (US$)'], color='red')
plt.plot(df['year'], model.predict(X))
plt.xlabel("Year")
plt.ylabel("Per Capita Income")
plt.show()
