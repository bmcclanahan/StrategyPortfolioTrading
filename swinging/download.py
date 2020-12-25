import pandas as pd
import datetime as dt
#import requests_cache
from eod_historical_data import get_eod_data
import os


def get_symbol_data(symbol, history_length, apikey):
    today = dt.datetime.now().strftime('%Y-%m-%d')
    history = (dt.datetime.now() - dt.timedelta(days=365 * history_length)).strftime('%Y-%m-%d')
    df = get_eod_data(symbol, 'US', start=history, end=today, api_key=apikey)
    df.loc[:, 'symbol'] = symbol
    factor = df.Adjusted_close / df.Close
    df.loc[:, 'adj_close'] = df.Adjusted_close
    df.loc[:, 'adj_open'] = df.Open * factor
    df.loc[:, 'adj_high'] = df.High * factor
    df.loc[:, 'adj_low'] = df.Low * factor
    return df

def download_symbols(api_key, history_length, output_dir, symbol_df,
                     indices, failed_indices):
    for index in indices:
        row = symbol_df.iloc[index]
        symbol = row.symbol
        sector = row.sector
        try:
            print('\r%d: %s' % (index, symbol))
            df = get_symbol_data(symbol, history_length, api_key)
            df.loc[:, 'sector'] = sector
            df.to_parquet('%s/%s.parquet' % (output_dir, symbol))
        except:
            print('failure occured')
            failed_indices.append(index)

def download_data(history_length, api_key, symbol_df, output_dir='data', limit=1500):
    #expire_after = dt.timedelta(days=1)
    #session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)
    failed_indices = []
    download_symbols(api_key, history_length, output_dir, symbol_df,
                     list(range(symbol_df.shape[0])), failed_indices)
    if len(failed_indices) != 0:
        print('retrying on failed symbols')
        failed_indices2 = []
        download_symbols(api_key, history_length, output_dir, symbol_df,
                         failed_indices, failed_indices2)

def combine_data(datadir='data', output_file='all_symbols.parquet'):
    files = os.listdir(datadir)
    dfs = []
    cols = ['adj_close', 'adj_open', 'adj_low', 'adj_high', 'Close', 'symbol', 'sector', 'Date', 'Volume']
    for f in files:
        dfs.append(pd.read_parquet('%s/%s' % (datadir, f)).reset_index()[cols])
    ttl_df = pd.concat(dfs, axis=0, ignore_index=True)
    ttl_df.to_parquet(output_file)
    return ttl_df

def writer_market_index_spy(apikey, output_file='market_index.parquet', history_length=5):
    df = get_symbol_data('SPY', history_length, apikey)
    df.reset_index().to_parquet(output_file)

def writer_sector(all_df, output_file='sector_index.parquet'):
    sector_indices = all_df.groupby(['sector', 'Date']).agg({'adj_close': 'sum'}).reset_index()
    sector_indices.to_parquet(output_file)

def write_data_and_indices(
    api_key, symbol_df, datadir='data',
    symbol_output_file='all_symbols.parquet',
    index_output_file='market_index.parquet',
    sector_output_file='sector_index.parquet', history_length=5
):
    download_data(history_length, api_key, symbol_df)
    combine_df = combine_data(datadir=datadir, output_file=symbol_output_file)
    writer_market_index_spy(
        api_key, output_file=index_output_file, history_length=history_length
    )
    writer_sector(combine_df, output_file=sector_output_file)



