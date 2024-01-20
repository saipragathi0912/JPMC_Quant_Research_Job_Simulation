import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
file_path = "/Users/pragathi/Nat Gas.csv"
date_column = "Dates"
date_format = "%m/%d/%y"
price_data = pd.read_csv(file_path,index_col=date_column,parse_dates=[date_column],date_format= date_format)
def differencing(series,period=2):
    return series.diff(periods=period)

def adf_test(series):
    """
    # Using ADF test for stationarity
    # Null hypothesis: The lag distribution has unit root (Which implies the series is nonstationary)
    # Alternate hypothesis: Negation of null hypothesis
    """
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    p_value = result[1]
    LOI = 0.05
    if(p_value <= LOI):
        print("Null hypothesis is rejected: The series is stationary")
    else:
        print("There is no enough evidence to reject the null hypothesis: The series is non stationary")

lag_1_series = differencing(price_data['Prices'],1)
lag_1_series.fillna(0, inplace=True)
adf_test(lag_1_series)

p = 10  # Order of autoregression
d = 0   # Order of differencing
q = 0  # Order of moving average
model = ARIMA(price_data['Prices'], order=(p, d, q))
results = model.fit()

# Forecast future values
n_forecast = 10  # Number of periods to forecast
forecast = results.get_forecast(steps=n_forecast)