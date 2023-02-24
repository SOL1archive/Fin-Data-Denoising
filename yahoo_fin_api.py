import yaml
import pandas as pd
import yfinance as yf

def download():
    option = yaml.load('data-load.yaml')
    data_df = yf.download(**option)

    data_df.to_csv(option['ticker'] + option['start'] + option['end'] + option['interval'] + '.csv')

if __name__ == '__main__':
    download()