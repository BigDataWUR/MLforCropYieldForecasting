import pandas as pd

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.config import CYPConfiguration
  from ..workflow.yield_trend import CYPYieldTrendEstimator

class TestYieldTrendEstimator():
  def __init__(self, yield_df):
    # TODO: Create a small yield data set
    self.yield_df = yield_df
    cyp_config = CYPConfiguration()
    self.verbose = 2
    cyp_config.setDebugLevel(self.verbose)
    self.trend_est = CYPYieldTrendEstimator(cyp_config)

  def testYieldTrendTwoRegions(self):
    print('\nFind the optimal trend window and estimate trend for first 2 regions')
    pd_yield_df = self.yield_df.toPandas()
    regions = sorted(pd_yield_df['IDREGION'].unique())
    reg1 = regions[0]
    pd_reg1_df = pd_yield_df[pd_yield_df['IDREGION'] == reg1]
    reg1_num_years = len(pd_reg1_df.index)
    reg1_max_year = pd_reg1_df['FYEAR'].max()
    reg1_min_year = pd_reg1_df['FYEAR'].min()

    if (self.verbose > 2):
      print('\nPrint Yield Trend Rounds')
      print('------------------------')

    trend_windows = [5]
    self.trend_est.printYieldTrendRounds(self.yield_df, reg1, trend_windows)

    if (self.verbose > 1):
      print('\n Fixed Trend Window prediction for region 1')
      print('---------------------------------------------------')
    trend_window = 5
    pd_fixed_win_df = self.trend_est.getFixedWindowTrend(self.yield_df, reg1, reg1_max_year,
                                                         trend_window)
    if (self.verbose > 1):
      print(pd_fixed_win_df.head(1))

    if (self.verbose > 1):
      print('\n Optimal Trend Window and prediction for region 1')
      print('---------------------------------------------------')
  
    trend_windows = [5, 7]
    pd_opt_win_df = self.trend_est.getOptimalWindowTrend(self.yield_df, reg1, reg1_max_year,
                                                         trend_windows)
    if (self.verbose > 1):
      print(pd_opt_win_df.head(1))

    reg2 = regions[1]
    pd_reg2_df = pd_yield_df[pd_yield_df['IDREGION'] == reg2]
    reg2_num_years = len(pd_reg2_df.index)
    reg2_max_year = pd_reg2_df['FYEAR'].max()
    reg2_min_year = pd_reg2_df['FYEAR'].min()

    if (self.verbose > 1):
      print('\n Fixed Trend Window prediction for region 2')
      print('---------------------------------------------------')
    trend_window = 5
    pd_fixed_win_df = self.trend_est.getFixedWindowTrend(self.yield_df, reg2, reg2_max_year,
                                                         trend_window)
    if (self.verbose > 1):
      print(pd_fixed_win_df.head(1))

    if (self.verbose > 1):
      print('\n Optimal Trend Window and prediction for region 2')
      print('---------------------------------------------------')

    pd_opt_win_df = self.trend_est.getOptimalWindowTrend(self.yield_df, reg2, reg2_max_year,
                                                         trend_windows)
    if (self.verbose > 1):
      print(pd_opt_win_df.head(1))

  def testYieldTrendAllRegions(self):
    print('\nYield trend estimation for all regions')

    print('\nOptimal Trend Windows')
    pd_trend_df = self.trend_est.getOptimalWindowTrendFeatures(self.yield_df)
    print(pd_trend_df.head(5))

    print('\nFixed Trend Window')
    pd_trend_df = self.trend_est.getFixedWindowTrendFeatures(self.yield_df)
    print(pd_trend_df.head(5))

  def runAllTests(self):
    print('\nTest Yield Trend Estimator BEGIN\n')
    self.testYieldTrendTwoRegions()
    self.testYieldTrendAllRegions()
    print('\nTest Yield Trend Estimator END\n')
