import datetime

import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc

def candlechart(fig, ax, df):
    candlestick2_ohlc(ax, df['Open'], df['High'], df['Low'], df['Close'], colorup='r', colordown='b')