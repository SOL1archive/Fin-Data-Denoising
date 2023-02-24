import numpy as np
import pandas as pd

def moving_avg(series: pd.Series, window):
    return series.rolling(window)

def exponential_moving_avg(series: pd.Series, alpha, adjust):
    return series.ewm(alpha=alpha, adjust=adjust)

def bilateral_filter():
    pass