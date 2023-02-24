import datetime

import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import mplfinance as mpf

def candlechart(df, volume=False):
    mc = mpf.make_marketcolors(
    up='r',
    down='b',
    )
    mco = [mc] * len(df)

    return mpf.plot(
        df[['Open', 'High', 'Low', 'Close', 'Volume']], 
        style='yahoo', 
        type='candle', 
        marketcolor_overrides=mco, volume=volume,
        returnfig=True
    )