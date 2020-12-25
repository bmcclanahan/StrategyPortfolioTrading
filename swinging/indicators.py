import ta
import numpy as np
import pandas as pd
import logging

def prep_regime_filter(regime_df, roc_col_name='regime_roc', mv_col_name='regime_ma',
                       close_name='regime_close',
                       ma_period=200,
                       roc_period=45):
    regime_df.loc[:, roc_col_name] = ta.momentum.ROCIndicator(regime_df.adj_close, n=roc_period).roc()
    regime_df.loc[:, mv_col_name] = regime_df.set_index(
        'Date'
    ).adj_close.rolling('%dd' % ma_period, min_periods=1).mean().values
    regime_df.loc[:, close_name] = regime_df.adj_close
    return regime_df

def mean_atr(df, atr_period=14):
    df.loc[:, 'last_close'] = df.adj_close.shift(1)
    atr_high = np.maximum(df.high_adj, df.last_close)
    atr_low = np.minimum(df.high_adj, df.last_close)
    atr = atr_high - atr_low
    return atr, atr.ewm(span=atr_period, adjust=False).mean()

def mean_close_diff_norm(close, ma):
    return (close - ma) / ma

def manual_mfi(df, period):
    mfi_df = df[['adj_close', 'Date', 'Volume']].set_index('Date')
    mfi_df.loc[:, 'prev_close'] = mfi_df.adj_close.shift(1)
    mfi_df.loc[:, 'perc_change'] = (mfi_df.prev_close - mfi_df.adj_close).abs() / mfi_df.adj_close
    up_index = mfi_df.adj_close > mfi_df.prev_close
    down_index = mfi_df.adj_close < mfi_df.prev_close
    mfi_df.loc[:, 'avg_up'] = 0
    mfi_df.loc[:, 'avg_down'] = 0
    mfi_df.loc[up_index, 'avg_up'] = mfi_df.loc[up_index, 'perc_change'] * mfi_df.loc[up_index, 'Volume']
    mfi_df.loc[down_index, 'avg_down'] = mfi_df.loc[down_index, 'perc_change'] * mfi_df.loc[down_index, 'Volume']
    mfi_df.loc[:, 'avg_up'] = mfi_df.loc[:, 'avg_up'].ewm(alpha=1.0 / period, adjust=False).mean()
    mfi_df.loc[:, 'avg_down'] = mfi_df.loc[:, 'avg_down'].ewm(alpha=1.0 / period, adjust=False).mean()
    mfi = (100.0 - (100.0 / (1 + (mfi_df.avg_up / mfi_df.avg_down)))).values
    return mfi

def generate_ta_features(sym_df, rsi_period=5, roc_period=45, roc_short_period=4,
                         roc_long_period=(20 * 5), break_out_period=(20 * 5),
                         fut_roc_period=5, mfi_period=5,
                         sto_period=14, atr_period=14, volitility_short_period=2, bba_period=20,
                         dch_period=20, ma_period=200,
                         macd_fast=12, macd_slow=26, macd_sig=9, bb_wide_dev=3, bb_slim_dev=1):
    sym_df = sym_df.sort_values('Date')
    sym_df.loc[:, 'mv_avg'] = sym_df.set_index(
        'Date'
    ).adj_close.rolling('%dd' % ma_period, min_periods=1).mean().values
    sym_df.loc[:, '200_day_high'] = sym_df.set_index(
        'Date'
    ).adj_close.rolling(
        '%dd' % break_out_period, min_periods=1
    ).max().shift().values
    rsi = ta.momentum.RSIIndicator(close=sym_df.adj_close, n=rsi_period).rsi()
    roc = ta.momentum.ROCIndicator(sym_df.adj_close, n=roc_period).roc()
    roc_long = ta.momentum.ROCIndicator(sym_df.adj_close, n=roc_long_period).roc()
    roc_short = ta.momentum.ROCIndicator(sym_df.adj_close, n=roc_short_period).roc()
    roc_fut = ta.momentum.ROCIndicator(sym_df.adj_close, n=fut_roc_period).roc()
    mfi = ta.volume.MFIIndicator(
        high=sym_df.adj_high, low=sym_df.adj_low,
        close=sym_df.adj_close, volume=sym_df.Volume,
        n=mfi_period
    ).money_flow_index()
    macd_diff = ta.trend.macd_diff(
        sym_df.adj_close, n_slow=macd_slow, n_fast=macd_fast, n_sign=macd_sig
    )
    sto = ta.momentum.StochasticOscillator(high=sym_df.adj_high, low=sym_df.adj_low, close=sym_df.adj_close,
                                     n=sto_period).stoch_signal()
    bb = ta.volatility.BollingerBands(close=sym_df.adj_close, n=bba_period)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    bba = bb_high - bb_low
    bb_wide = ta.volatility.BollingerBands(close=sym_df.adj_close, n=bba_period, ndev=bb_wide_dev)
    bb_wide_high = bb_wide.bollinger_hband()
    bb_slim = ta.volatility.BollingerBands(close=sym_df.adj_close, n=bba_period, ndev=bb_slim_dev)
    bb_slim_low = bb_slim.bollinger_lband()
    dc = ta.volatility.DonchianChannel(high=sym_df.adj_high, low=sym_df.adj_low, close=sym_df.adj_close, n=dch_period)
    dc_high = dc.donchian_channel_hband()
    dc_low = dc.donchian_channel_lband()
    dch = dc_high - dc_low
    sym_df.loc[:, 'rsi'] = rsi
    sym_df.loc[:, 'roc'] = roc
    sym_df.loc[:, 'roc_long'] = roc_long
    sym_df.loc[:, 'roc_short'] = roc_short
    sym_df.loc[:, 'fut_roc'] = roc_fut.shift(-fut_roc_period)
    sym_df.loc[:, 'mfi'] = mfi
    sym_df.loc[:, 'sto'] = sto
    sym_df.loc[:, 'bba'] = bba
    sym_df.loc[:, 'bb_high'] = bb_high
    sym_df.loc[:, 'bb_low'] = bb_low
    sym_df.loc[:, 'bb_wide_high'] = bb_wide_high
    sym_df.loc[:, 'bb_slim_low'] = bb_slim_low
    sym_df.loc[:, 'dch'] = dch
    sym_df.loc[:, 'bba_norm'] = bba / sym_df.adj_close
    sym_df.loc[:, 'dch_norm'] = dch / sym_df.adj_close
    sym_df.loc[:, 'ma_diff_norm'] = mean_close_diff_norm(sym_df.adj_close, sym_df.mv_avg)
    sym_df.loc[:, 'regime_ma_diff_norm'] = mean_close_diff_norm(sym_df.regime_close, sym_df.regime_ma)
    sym_df.loc[:, 'sector_ma_diff_norm'] = mean_close_diff_norm(sym_df.sector_close, sym_df.sector_ma)
    volatility_base = (sym_df.adj_close.diff() / sym_df.adj_close.shift(1)).abs()
    sym_df.loc[:, 'volatility'] = volatility_base.ewm(span=atr_period, adjust=False).mean()
    sym_df.loc[:, 'volatility_short'] = volatility_base.ewm(span=volitility_short_period, adjust=False).mean()
    sym_df.loc[:, 'directional_strength'] = -(sym_df.adj_close.diff() / sym_df.adj_close.shift(1)).ewm(
        span=atr_period,
        adjust=False
    ).mean()
    sym_df.loc[:, 'macd_diff'] = macd_diff
    return sym_df

def fill_outstanding_shares(sym_df):
    sym_df.loc[:, 'CommonStockSharesOutstanding'] = sym_df['CommonStockSharesOutstanding'].fillna(method='ffill')
    return sym_df

def get_quarter(date):
    quarter = np.ceil(date.dt.month / 3)
    return quarter

def get_prev_quart_and_num(year, quarter):
    prev_quarter = quarter - 1
    year_diff = np.zeros(year.shape[0])
    year_diff[prev_quarter == 0] = -1
    prev_year = year - year_diff
    prev_quarter.loc[prev_quarter == 0] = 4
    return prev_year, prev_quarter

def get_fundamental_data(ta_feats, fundamental_df):
    ta_feats.loc[:, 'quarter'] = get_quarter(ta_feats['Date'])
    prev_year, prev_quarter = get_prev_quart_and_num(ta_feats['Date'].dt.year, ta_feats.quarter)
    ta_feats.loc[:, 'prev_year'] = prev_year
    ta_feats.loc[:, 'prev_quarter'] = prev_quarter
    fundamental_cols = ['CommonStockSharesOutstanding', 'EarningsPerShareBasic',
                        'year', 'quarter', 'symbol']
    join_df = ta_feats.reset_index(drop=True).merge(
        fundamental_df[fundamental_cols].drop_duplicates(),
        left_on=['prev_year', 'prev_quarter', 'symbol'],
        right_on=['year', 'quarter', 'symbol'],
        how='left'
    )
    join_df = join_df.groupby('symbol').apply(fill_outstanding_shares).reset_index(drop=True)
    join_df.loc[:, 'MarketCap'] = join_df.CommonStockSharesOutstanding * join_df.Close
    return join_df

def get_ta_and_fundamental_data(
        data_path='all_symbols.parquet',
        market_index='market_index.parquet',
        sector_index='sector_index.parquet',
        fundamental_data_path='fundamentals.parquet',
        output_file='ta_data.parquet'):
    df = pd.read_parquet(data_path)
    regime_df = pd.read_parquet(market_index)
    sector_df = pd.read_parquet(sector_index)
    regime_df = prep_regime_filter(regime_df)
    sector_df = sector_df.groupby('sector').apply(
        lambda x: prep_regime_filter(x, roc_col_name='sector_roc',
        mv_col_name='sector_ma',
        close_name='sector_close')
    )
    df = df.merge(regime_df[['Date', 'regime_close', 'regime_ma']], on='Date', how='left')
    df = df.merge(sector_df[['Date', 'sector', 'sector_close', 'sector_ma']], on=['Date', 'sector'], how='left')
    logging.info('computing technical indicators')
    ta_feats = df.groupby('symbol').apply(generate_ta_features)
    logging.info('adding fundamental data')
    fundamental_df = pd.read_parquet(fundamental_data_path)
    all_data = get_fundamental_data(ta_feats, fundamental_df)
    all_data.to_parquet(output_file)


