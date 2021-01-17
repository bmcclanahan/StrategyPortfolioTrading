from abc import ABC, abstractmethod
from numba import jit
import datetime as dt
import numpy as np
import pandas as pd


def run_strategy_on_data(CustStrat, data_loc='ta_data.parquet', strategy_type='mean_reversion', run_length=30):
    df = pd.read_parquet(data_loc).reset_index(drop=True)
    df.loc[:, 'date'] = df.Date
    strat = CustStrat(df)
    strat.run_backtest_generate_reports(strategy_type=strategy_type, run_length=run_length)
    del df
    del strat

@jit(nopython=True)
def backtest_numba(enter_exit, close_price, open_price, df_index, stop_thresh,
                   run_length, inv_price, equity_signal, bool_date, mv_avg_ratio_thresh,
                   position, profit_target):
        in_trade = False
        n = len(enter_exit)
        actual_enter_exit = np.zeros(n)
        shares_arr = np.zeros(n)
        equity = np.zeros(n)
        profit = np.zeros(n)
        exit_profit = np.zeros(n)
        exit_index = np.zeros(n) - 1
        start_price = 1e-18
        enter_price = 1e-18
        top_price = start_price
        shares = 0
        for index in range(0, n):
            signal = enter_exit[index]
            equity_stop_signal = equity_signal[index] < mv_avg_ratio_thresh and bool_date[index] == True
            if in_trade and close_price[index] > top_price:
                top_price = close_price[index]
            # compute stop loss hit or profit target reached
            profit_or_stop = ((top_price - close_price[index]) / top_price) * position >= stop_thresh
            profit_or_stop = profit_or_stop or ((((close_price[index] - enter_price) / enter_price) * position) >= profit_target)
            if not in_trade and signal == 1 and not equity_stop_signal:
                enter_price = open_price[index]
                start_price = close_price[index]
                top_price = start_price
                shares = int(inv_price / start_price) #need condition here to see if you can afford shares
                shares_arr[index] = shares
                shares_cost = enter_price * shares
                actual_enter_exit[index] = 1
                in_trade = True
                enter_index = index
            elif in_trade and ((signal == -1) or ((index - enter_index) >= run_length) or equity_stop_signal): #exit signal
                profit[enter_index] = (open_price[index] - enter_price) * shares * position
                exit_profit[index] = profit[enter_index]
                exit_index[enter_index] = df_index[index]
                actual_enter_exit[index] = -1
                in_trade = False
                shares = 0
            elif in_trade and profit_or_stop:
                profit[enter_index] = (open_price[index] - enter_price) * shares * position
                exit_profit[index] = profit[enter_index]
                exit_index[enter_index] = df_index[index]
                actual_enter_exit[index] = -1
                in_trade = False
                shares = 0
            elif index == (n - 1) and in_trade:
                profit[enter_index] = (open_price[index] - enter_price) * shares * position
                exit_profit[index] = profit[enter_index]
                exit_index[enter_index] = df_index[index]
                actual_enter_exit[index] = -1
                in_trade = False
                shares = 0 # lots of duplication here
            equity[index] = (shares * close_price[index]) - (shares * enter_price)
            shares_arr[index] = shares
        return profit, exit_profit, exit_index, actual_enter_exit, shares_arr, equity # don't really need exit profit here


class Strategy(ABC):

    def __init__(self, df, strategy_name):
        self.df = df
        self.strategy_name = strategy_name

    @abstractmethod
    def get_entrances():
        pass

    @abstractmethod
    def get_exits():
        pass

    @abstractmethod
    def run_backtest():
        pass

    @abstractmethod
    def run_backtest_generate_reports():
        pass

    def backtest_seq(self, df, stop_thresh=0.1, run_length=30, inv_price=10000,
                     prof_avg_offset=30, ewm_prof_offset=100, mv_avg=None, equity=None,
                     pickup_dt=dt.datetime(1990, 10, 18), mv_avg_ratio_thresh=.97,
                     position=1, profit_target=np.inf):
        if mv_avg is None or equity is None:
            mv_avg = np.zeros(df.shape[0])
            equity = np.zeros(df.shape[0])
        df.loc[:, 'enter_exit_sig'] = df.entrances - df.exits
        df.loc[:, 'next_open'] = df.adj_open.shift(-1)
        profit, exit_profit, exit_index, actual_enter_exit, shares_arr, equity = backtest_numba(
            df.enter_exit_sig.values, df.adj_close.values,
            df.next_open.values, df.index.values, stop_thresh, run_length,
            inv_price, equity / mv_avg, (df['date'] >= pickup_dt).values,
            mv_avg_ratio_thresh, position, profit_target
        )
        df.loc[:, 'profit'] = profit
        df.loc[:, 'exit_profit'] = exit_profit
        df.loc[:, 'cum_exit_profit'] = df.exit_profit.fillna(0).cumsum()
        df.loc[:, 'equity_curve'] = equity + df.cum_exit_profit
        bool_index = exit_index != -1
        df.loc[bool_index, 'exit_date'] = df.loc[exit_index[bool_index], 'date'].values
        df.loc[:, 'cum_profit'] = df.profit.fillna(0).cumsum()
        df.loc[:, 'purch_shares'] = shares_arr
        df.loc[:, 'norm_profit'] = profit / (df.next_open * shares_arr)
        df.loc[df.profit == 0, 'norm_profit'] = np.nan
        df.loc[:, 'avg_profit'] = df.norm_profit.rolling(prof_avg_offset, min_periods=1).mean()
        df.loc[:, 'avg_profit_std'] = df.norm_profit.rolling(prof_avg_offset, min_periods=1).std()
        df.loc[:, 'eavg_profit'] = df.avg_profit.ewm(ewm_prof_offset, ignore_na=True).mean()
        df.loc[:, 'avg_profit'] = df.avg_profit.fillna(0)
        df.loc[:, 'actual_enter_exit'] = actual_enter_exit
        df.loc[:, 'actual_enter'] = 0
        df.loc[:, 'actual_exit'] = 0
        df.loc[df.actual_enter_exit == 1, 'actual_enter'] = 1
        df.loc[df.actual_enter_exit == -1, 'actual_exit'] = 1
        df.loc[:, 'trade_count'] = df.actual_enter_exit.rolling(prof_avg_offset).sum()
        return df

    def get_profit_metrics(self, df_profits):
        wins_losses = {}
        col_name = 'profit'
        win_index = df_profits[col_name] > 0
        loss_index = df_profits[col_name] < 0
        mean_win = df_profits.loc[win_index, col_name].mean()
        mean_loss = df_profits.loc[loss_index, col_name].mean()
        mean_norm_profit_win = df_profits.loc[win_index, 'norm_profit'].mean()
        mean_norm_profit_loss = df_profits.loc[loss_index, 'norm_profit'].mean()
        mean_norm_profit = df_profits.norm_profit.mean()
        sum_win = df_profits.loc[win_index, col_name].sum()
        sum_loss = df_profits.loc[loss_index, col_name].sum()

        wins_losses[col_name] = [win_index.sum(), loss_index.sum(), win_index.sum() + loss_index.sum(),
                                 mean_win, mean_loss,
                                 mean_norm_profit_win, mean_norm_profit_loss,
                                 mean_norm_profit,
                                 sum_win, sum_loss
                                ]

        df_win_loss = pd.DataFrame(wins_losses, index=['wins', 'losses', 'ttl_trades', 'mean_win',
                                                       'mean_loss',
                                                       'mean_norm_profit_win', 'mean_norm_profit_loss',
                                                       'mean_norm_profit',
                                                       'ttl_win', 'ttl_loss']).transpose()
        df_win_loss.loc[:, 'win_loss_rate'] =  df_win_loss.wins / (df_win_loss.losses + df_win_loss.wins)
        df_win_loss.loc[:, 'win_loss_ratio'] = df_win_loss.mean_win / np.abs(df_win_loss.mean_loss)

        df_win_loss.loc[:, 'profit_factor'] = df_win_loss.ttl_win / np.abs(df_win_loss.ttl_loss)
        df_win_loss.loc[:, 'net_profit'] = df_win_loss.ttl_win + df_win_loss.ttl_loss
        return df_win_loss

    def equity_curve(self, df_profits):
        strategy_stats_df = pd.DataFrame({'date': df_profits['date'].unique()}).sort_values('date')
        date_equity = df_profits.groupby('date').equity_curve.sum().reset_index('date')
        strategy_stats_df = strategy_stats_df.merge(date_equity, on='date', how='left')
        strategy_stats_df.loc[:, 'equity_curve_mv_avg'] = strategy_stats_df.set_index('date').equity_curve.rolling('200d', min_periods=1)\
                                                                           .mean().fillna(method='ffill').values
        strategy_stats_df.loc[:, 'equity_curve_agg'] = strategy_stats_df.equity_curve
        strategy_stats_df.loc[:, 'equity_curve_ratio'] = strategy_stats_df.equity_curve_agg / (strategy_stats_df.equity_curve_mv_avg + np.finfo(float).eps)
        return strategy_stats_df

    def generate_profit_metrics_reports(self, df_profits,
                                        profit_report_folder_location='profit_reports', strategy_type='mean_reversion'):
        df_profits.loc[:, 'year'] = df_profits.Date.dt.year
        df_profits.loc[:, 'month'] = df_profits.Date.dt.month
        overall = self.get_profit_metrics(df_profits)
        year = df_profits.groupby('year').apply(self.get_profit_metrics).reset_index()
        month = df_profits.groupby(['year', 'month']).apply(self.get_profit_metrics).reset_index()
        overall.to_excel('%s/%s/%s_overall.xlsx' % (profit_report_folder_location, strategy_type, self.strategy_name), index=False)
        year.to_excel('%s/%s/%s_year.xlsx' % (profit_report_folder_location, strategy_type, self.strategy_name), index=False)
        month.to_excel('%s/%s/%s_month.xlsx' % (profit_report_folder_location, strategy_type, self.strategy_name), index=False)

    def generate_entrance_exit_reports(self, df_profits, entry_exit_folder_location='strategy_entrances_exits',
                                       strategy_type='mean_reversion',
                                       lag_days=5):
        max_date = df_profits.Date.max()
        old_date = max_date - dt.timedelta(days=lag_days)
        df_dt_range = df_profits.loc[df_profits.Date.between(old_date, max_date)]
        entrances = df_dt_range.loc[df_dt_range.entrances ==1]
        exits = df_dt_range.loc[df_dt_range.exits ==1]
        entrances.sort_values('Date', ascending=False).to_excel('%s/%s/%s_entrances.xlsx' % (entry_exit_folder_location, strategy_type, self.strategy_name))
        exits.sort_values('Date', ascending=False).to_excel('%s/%s/%s_exits.xlsx' % (entry_exit_folder_location, strategy_type, self.strategy_name))

    def run_backtest_generate_reports(self, profit_report_folder_location='profit_reports',
                                      strategy_type='mean_reversion',
                                      equity_curve_location='equity_curves', entry_exit_folder_location='strategy_entrances_exits',
                                      run_length=30, lag_days=5):
        df_profits = self.run_backtest(run_length=run_length)
        equity_curve = self.equity_curve(df_profits)
        equity_curve.to_excel('%s/%s/%s_eq_curve.xlsx' % (equity_curve_location, strategy_type, self.strategy_name), index=False)
        df_profits = df_profits.merge(equity_curve, on='date')
        enter_exit = df_profits.loc[(df_profits.actual_enter == 1) | (df_profits.exits == 1)]
        enter_exit.to_parquet('%s/%s/%s_profits.parquet' % (profit_report_folder_location, strategy_type, self.strategy_name))
        self.generate_profit_metrics_reports(df_profits, profit_report_folder_location=profit_report_folder_location, strategy_type=strategy_type)
        self.generate_entrance_exit_reports(df_profits, entry_exit_folder_location=entry_exit_folder_location, strategy_type=strategy_type,
                                            lag_days=lag_days)


class RSIStrategy(Strategy):

    def __init__(self, df):
        Strategy.__init__(self, df, 'RSI')

    def get_entrances(self, df,
                  rsi_oversold=30,
                  roc_change=0,
                  regime_roc_change=0):
        df.loc[:, 'last_rsi'] = df.rsi.shift(1)
        df.loc[:, 'rsi_oversold_enter'] = 0
        df.loc[:, 'rsi_roc'] = df.rsi - df.last_rsi
        df.loc[:, 'last_rsi_roc'] = df.rsi_roc.shift(1)
        bool_index = (df.rsi <= rsi_oversold)
        bool_index &= (df.rsi_roc > df.last_rsi_roc)
        bool_index &= (df.roc > roc_change)
        bool_index &= (df.adj_close > df.mv_avg)
        bool_index &= (df.regime_close > df.regime_ma)
        bool_index &= (df.sector_close > df.sector_ma)
        df.loc[bool_index, 'rsi_oversold_enter'] = 1
        enter_cols = ['rsi_oversold_enter']
        df.loc[:, 'entrances'] = df[enter_cols].sum(axis=1).clip(upper=1)
        return df

    def get_exits(self, df, rsi_overbought=70):
        df.loc[:, 'rsi_overbought_exit'] = 0
        bool_index = df.rsi >= rsi_overbought
        df.loc[bool_index, 'rsi_overbought_exit'] = 1
        exit_cols = ['rsi_overbought_exit']
        df.loc[:, 'exits'] = df[exit_cols].sum(axis=1).clip(upper=1)
        return df

    def run_backtest(self, run_length=30):
        df_enter_exit = self.df.groupby('symbol').apply(
            lambda x: self.get_entrances(
                self.get_exits(x), roc_change=10, regime_roc_change=0
            )
        )
        df_profits = df_enter_exit.groupby('symbol').apply(lambda x: self.backtest_seq(x, stop_thresh=1.0, inv_price=10000, run_length=run_length))
        return df_profits


class MFIStrategy(Strategy):

    def __init__(self, df):
        Strategy.__init__(self, df, 'MFI')

    def get_entrances(self, df,
                  mfi_oversold=20,
                  roc_change=0,
                  regime_roc_change=0):
        df.loc[:, 'last_mfi'] = df.mfi.shift(1)
        df.loc[:, 'mfi_roc'] = df.mfi - df.last_mfi
        df.loc[:, 'last_mfi_roc'] = df.mfi_roc.shift(1)
        df.loc[:, 'mfi_oversold_enter'] = 0
        bool_index = (df.adj_close > df.mv_avg)
        bool_index &= (df.sector_close > df.sector_ma)
        bool_index &= (df.mfi <= mfi_oversold) & (df.roc > roc_change)
        df.loc[bool_index, 'mfi_oversold_enter'] = 1
        enter_cols = ['mfi_oversold_enter']
        df.loc[:, 'entrances'] = df[enter_cols].sum(axis=1).clip(upper=1)
        return df

    def get_exits(self, df, mfi_overbought=80):
        df.loc[:, 'close_adj_last'] = df.adj_close.shift(1)
        df.loc[:, 'mfi_overbought_exit'] = 0
        df.loc[df.mfi >= mfi_overbought, 'mfi_overbought_exit'] = 1
        exit_cols = ['mfi_overbought_exit']
        df.loc[:, 'exits'] = df[exit_cols].sum(axis=1).clip(upper=1)
        return df

    def run_backtest(self, run_length=30):
        df_enter_exit = self.df.groupby('symbol').apply(
            lambda x: self.get_entrances(
                self.get_exits(x), roc_change=7
            )
        )
        df_profits = df_enter_exit.groupby('symbol').apply(lambda x: self.backtest_seq(x, stop_thresh=1.0, inv_price=10000, run_length=run_length))
        return df_profits


class STOStrategy(Strategy):

    def __init__(self, df):
        Strategy.__init__(self, df, 'STO')

    def get_entrances(self, df,
                  sto_oversold=20,
                  roc_change=0,
                  regime_roc_change=0,
                  volatility_thresh=0.05):
        df.loc[:, 'rsi_oversold_enter'] = 0
        df.loc[:, 'last_sto'] = df.sto.shift(1)
        df.loc[:, 'sto_roc'] = df.sto - df.last_sto
        df.loc[:, 'last_sto_roc'] = df.sto_roc.shift(1)
        df.loc[:, 'sto_oversold_enter'] = 0
        bool_index = df.sto <= sto_oversold
        bool_index &= df.roc > roc_change
        bool_index &= (df.regime_close > df.regime_ma)
        bool_index &= (df.sector_close > df.sector_ma)
        bool_index &= (df.volatility < volatility_thresh)
        df.loc[bool_index, 'sto_oversold_enter'] = 1
        enter_cols = ['sto_oversold_enter']
        df.loc[:, 'entrances'] = df[enter_cols].sum(axis=1).clip(upper=1)
        return df

    def get_exits(self, df, sto_overbought=80):
        df.loc[:, 'sto_overbought_exit'] = 0
        df.loc[df.sto >= sto_overbought, 'sto_overbought_exit'] = 1
        exit_cols = ['sto_overbought_exit']
        df.loc[:, 'exits'] = df[exit_cols].sum(axis=1).clip(upper=1)
        return df

    def run_backtest(self, run_length=30):
        df_enter_exit = self.df.groupby('symbol').apply(
            lambda x: self.get_entrances(
                self.get_exits(x), roc_change=10
            )
        )
        df_profits = df_enter_exit.groupby('symbol').apply(lambda x: self.backtest_seq(x, stop_thresh=1.0, inv_price=10000, run_length=run_length))
        return df_profits


class BreakoutStrategy(Strategy):

    def __init__(self, df):
        Strategy.__init__(self, df, 'Breakout')

    def get_entrances(self, df,
                  roc_change=30):
        bool_index = df.adj_close >= df['200_day_high']
        bool_index &= df.roc_long > roc_change
        bool_index &= (df.regime_close > df.regime_ma)
        bool_index &= (df.sector_close > df.sector_ma)
        df.loc[bool_index, 'breakout_enter'] = 1
        enter_cols = ['breakout_enter']
        df.loc[:, 'entrances'] = df[enter_cols].sum(axis=1).clip(upper=1)
        return df

    def get_exits(self, df):
        df.loc[:, 'exits'] = 0
        return df

    def run_backtest(self, run_length=300):
        df_enter_exit = self.df.groupby('symbol').apply(
            lambda x: self.get_entrances(
                self.get_exits(x), roc_change=30
            )
        )
        df_profits = df_enter_exit.groupby('symbol').apply(lambda x: self.backtest_seq(x, stop_thresh=0.2, inv_price=10000, run_length=run_length))
        return df_profits


class BollingerStrategy(Strategy):

    def __init__(self, df):
        Strategy.__init__(self, df, 'Bollinger')

    def get_entrances(self, df,
                      roc_change=10):
        bool_index = df.roc_long > roc_change
        bool_index &= (df.regime_close > df.regime_ma)
        bool_index &= (df.sector_close > df.sector_ma)
        bool_index &= (df.adj_close > df.bb_high)
        df.loc[bool_index, 'bollinger_enter'] = 1
        enter_cols = ['bollinger_enter']
        df.loc[:, 'entrances'] = df[enter_cols].sum(axis=1).clip(upper=1)
        return df

    def get_exits(self, df):
        df.loc[:, 'bollinger_exit'] = 0
        df.loc[df.adj_close < df.bb_slim_low, 'bollinger_exit'] = 1
        exit_cols = ['bollinger_exit']
        df.loc[:, 'exits'] = df[exit_cols].sum(axis=1).clip(upper=1)
        return df

    def run_backtest(self, run_length=300):
        df_enter_exit = self.df.groupby('symbol').apply(
            lambda x: self.get_entrances(
                self.get_exits(x), roc_change=30
            )
        )
        df_profits = df_enter_exit.groupby('symbol').apply(lambda x: self.backtest_seq(x, stop_thresh=0.2, inv_price=10000, run_length=run_length, profit_target=0.3))
        return df_profits


class MACDStrategy(Strategy):

    def __init__(self, df):
        Strategy.__init__(self, df, 'MACD')

    def get_entrances(self, df,
                      roc_change=10):
        bool_index = df.roc > roc_change
        bool_index &= (df.macd_diff > 0)
        df.loc[bool_index, 'macd_enter'] = 1
        enter_cols = ['macd_enter']
        df.loc[:, 'entrances'] = df[enter_cols].sum(axis=1).clip(upper=1)
        return df

    def get_exits(self, df):
        df.loc[:, 'macd_exit'] = 0
        df.loc[0 > df.macd_diff, 'macd_exit'] = 1
        exit_cols = ['macd_exit']
        df.loc[:, 'exits'] = df[exit_cols].sum(axis=1).clip(upper=1)
        return df

    def run_backtest(self, run_length=300):
        df_enter_exit = self.df.groupby('symbol').apply(
            lambda x: self.get_entrances(
                self.get_exits(x), roc_change=10
            )
        )
        df_profits = df_enter_exit.groupby('symbol').apply(lambda x: self.backtest_seq(x, stop_thresh=0.2, inv_price=10000, run_length=run_length, profit_target=0.2))
        return df_profits
