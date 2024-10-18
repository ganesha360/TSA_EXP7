# Ex.No: 07                                       AUTO REGRESSIVE MODEL
## DEVELOPED BY: GANESH R
## REGISTER NO: 212222240029
## Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```PY
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

file_path = '/content/Amazon.csv'
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

default_column_name = df.columns[0]  # Assuming rainfall data is in the first column
series = df[default_column_name]
print(f"Column 'Volume' not found. Using '{default_column_name}' instead.")

adf_result = adfuller(series)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(series, lags=30, ax=plt.gca())
plt.subplot(122)
plot_pacf(series, lags=30, ax=plt.gca())
plt.show()

model = AutoReg(train, lags=13)
model_fitted = model.fit()

predictions = model_fitted.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.legend()
plt.title('AR Model - Actual vs Predicted')
plt.show()

mse = mean_squared_error(test, predictions)
print(f"Mean Squared Error: {mse}")
```

### OUTPUT:
## PACF - ACF
![image](https://github.com/user-attachments/assets/94db075e-f6e6-4479-b04c-1a09a5e82554)

## PREDICTION
![image](https://github.com/user-attachments/assets/49b06084-dd10-497d-a6a3-be939235ccee)

## FINIAL PREDICTION
![image](https://github.com/user-attachments/assets/a31ad491-a5e4-4084-89c8-8c20403526bf)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
