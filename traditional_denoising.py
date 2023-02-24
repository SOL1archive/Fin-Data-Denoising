import numpy as np
import pandas as pd
from tsfilt.filt import BilateralFilter

def moving_avg(series: pd.Series, window):
    return series.rolling(window).sum()

def exp_moving_avg(series: pd.Series, **kwarg):
    return series.ewm(**kwarg).mean()

def bilateral_filter(series: pd.Series, window, sigma_d, sigma_i):
    return BilateralFilter(win_size=window, sigma_d=sigma_d, sigma_i=sigma_i).fit_transform(series)