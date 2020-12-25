import os
import pandas as pd
import pickle
from joblib import load
import datetime as dt
import numpy as np
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
from . import ibapp


def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_positions():
    positions_app = ibapp.PositionsApp()
    positions_app.connect('127.0.0.1',  7497, clientId=0)
    positions_app.get_positions()
    positions_app.run()
    return positions_app.positions_df.loc[positions_app.positions_df.position != 0]

def load_profits(fill_na_dict, strategy_index, feat_col, profit_folder='profit_reports/mean_reversion'):
    profit_files = [x for x in os.listdir(profit_folder) if x.endswith('.parquet')]
    all_cols = feat_col + ['symbol', 'date', 'norm_profit', 'actual_enter', 'actual_exit', 'exits', 'volatility_short']
    dfs = {x.split('_')[0]: pd.read_parquet('%s/%s' % (profit_folder, x))[all_cols] for x in profit_files}
    for key in dfs:
        dfs[key].loc[:, 'strategy_ind'] = strategy_index[key]
    df = pd.concat([dfs[x].fillna(fill_na_dict) for x in dfs], axis=0, ignore_index=True)
    return df


def make_symbol_plot(sym_df, indicator, folder):
    indicator_dict = {
        'macd': 'macd_diff',
        'breakout': '200_day_high',
        'bollinger': 'bb_wide_high'
    }
    indicator = indicator if indicator not in indicator_dict else indicator_dict[indicator]
    sym_df = sym_df.sort_values('Date')
    fig = make_subplots(rows=2, cols=1, row_heights=[0.5, 0.5], shared_xaxes=True)

    fig.add_trace(
        go.Candlestick(
            x=sym_df.Date, open=sym_df.adj_open, high=sym_df.adj_high, low=sym_df.adj_low, close=sym_df.adj_close, name='bars'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sym_df.Date, y=sym_df[indicator], name=indicator, marker=dict(color='purple')
        ),
        row=2, col=1
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    plotly.offline.plot(fig, filename=f'{folder}/{sym_df.symbol.iloc[0]}.html', auto_open=False)

def plot_entry_symbols(entry_symbols, df, folder):
    create_dir(folder)
    for index in entry_symbols.index:
        symbol = entry_symbols.loc[index, 'symbol']
        strategy =  entry_symbols.loc[index, 'strategy']
        make_symbol_plot(df.loc[df.symbol == symbol], strategy.lower(), folder)

def generate_signals(
        cols,
        model_file='rfc_mr.joblib' ,ml_metafile='ml_metadata_mr.pkl',
        days_lookback=5, pred_thresh=.65, num_positions=2, indicator_data='ta_data.parquet',
        strategy_type='mean_reversion'
):
    with open(ml_metafile, 'rb') as ml_meta_file:
        ml_meta = pickle.load(ml_meta_file)
    index_to_strat = {ml_meta['strategy_index'][x]: x for x in ml_meta['strategy_index']}
    df = load_profits(ml_meta['fill_na_dict'], ml_meta['strategy_index'], ml_meta['feat_col'],
                      profit_folder=f'profit_reports/{strategy_type}')
    with open(model_file, 'rb') as rfc_file:
        rfc = load(rfc_file)
    max_date = df['date'].max()
    min_date = max_date - dt.timedelta(days=days_lookback)
    df_date_filt = df.loc[df['date'].between(min_date, max_date) & (df.roc_short != np.inf)]
    preds = rfc.predict_proba(df_date_filt[ml_meta['feat_col']].fillna(0))[0][:, 1]
    df_date_filt.loc[:, 'pred'] = 0
    df_date_filt.loc[preds >= pred_thresh, 'pred'] = 1
    df_date_filt.loc[:, 'pred_score'] = preds
    df_date_filt.loc[:, 'strategy'] = df_date_filt.strategy_ind.map(index_to_strat)
    entries = df_date_filt.loc[df_date_filt.actual_enter == 1].sort_values(['date', 'pred', 'volatility'],
                                                                           ascending=[False, False, False])
    exits = df_date_filt.loc[df_date_filt.exits == 1]
    actual_exits = df_date_filt.loc[df_date_filt.actual_exit == 1]
    entries[cols].to_csv(f'signals/{strategy_type}/entries.csv', index=False)
    exits[cols].to_csv(f'signals/{strategy_type}/exits.csv', index=False)
    positions = pd.read_csv(f'signals/{strategy_type}/{strategy_type}_positions.csv', parse_dates=['date'])
    actual_exits = positions.merge(exits, on=['symbol', 'strategy'], suffixes=["", "_y"])
    actual_exits = actual_exits.loc[(actual_exits.date < actual_exits.date_y)]
    actual_exits[cols].to_csv(f'signals/{strategy_type}/actual_exits.csv', index=False)
    entry_symbols = entries[['strategy', 'symbol']].drop_duplicates().iloc[:num_positions]
    df = pd.read_parquet(indicator_data)
    folder = f'signals/{strategy_type}/plots'
    plot_entry_symbols(entry_symbols, df, folder)
    positions = get_positions()
    exits.loc[exits.symbol.isin(positions.symbol.values)][cols].to_csv(f'signals/{strategy_type}/position_exits.csv',
                                                                       index=False)




