import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('data/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn import linear_model
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

print("Regression Coefficients:",regressor.coef_)
print("Intercept:",regressor.intercept_)

y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Evaluate on Train Data
mse = mean_squared_error(y_train, regressor.predict(X_train))
print(f'Mean Squared Error: {mse}')

#Evaluate on Test Data
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict the value at a given point (e.g., 5 years of experience)
years_of_experience = [[5]]
predicted_salary = regressor.predict(years_of_experience)
print(f'Predicted Salary for {years_of_experience[0][0]} years of experience: {predicted_salary[0]}')
