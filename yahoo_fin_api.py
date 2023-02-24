import yaml
import pandas as pd
import yfinance as yf

def option2filename(option):
    return option['ticker'] + option['start'] + option['end'] + option['interval'] + '.csv'

def download(option=None):
    if option is None:
        option = yaml.load('data-load.yaml')
    df = yf.download(**option)
    
    df['Mean'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)

    df.to_csv(option2filename(option))

    return df

def load_df(filepath_or_buffer=None):
    if filepath_or_buffer is None:
        option = yaml.load('data-load.yaml')
        df = pd.read_csv(option2filename(option))
    else:
        df = pd.read_csv(filepath_or_buffer).set_index('Date')

    return df

if __name__ == '__main__':
    df = download()
    print(df.head())

