import numpy as np
import functools
from pyspark.sql import Window
from sklearn.metrics import mean_squared_error

from ..common import globals

if (globals.test_env == 'pkg'):
  SparkT = globals.SparkT
  SparkF = globals.SparkF

class CYPYieldTrendEstimator:
  def __init__(self, cyp_config):
    self.verbose = cyp_config.getDebugLevel()
    self.trend_windows = cyp_config.getTrendWindows()

  def setTrendWindows(self, trend_windows):
    """Set trend window lengths"""
    assert isinstance(trend_windows, list)
    assert len(trend_windows) > 0
    assert isinstance(trend_windows[0], int)
    self.trend_windows = trend_windows

  def getTrendWindowYields(self, df, trend_window, reg_id=None):
    """Extract previous years' yield values to separate columns"""
    sel_cols = ['IDREGION', 'FYEAR', 'YIELD']
    my_window = Window.partitionBy('IDREGION').orderBy('FYEAR')

    yield_fts = df.select(sel_cols)
    if (reg_id is not None):
      yield_fts = yield_fts.filter(yield_fts.IDREGION == reg_id)

    for i in range(trend_window):
      yield_fts = yield_fts.withColumn('YIELD-' + str(i+1),
                                       SparkF.lag(yield_fts.YIELD, i+1).over(my_window))
      yield_fts = yield_fts.withColumn('YEAR-' + str(i+1),
                                       SparkF.lag(yield_fts.FYEAR, i+1).over(my_window))

    # drop columns withs null values
    for i in range(trend_window):
      yield_fts = yield_fts.filter(SparkF.col('YIELD-' + str(i+1)).isNotNull())

    prev_yields = [ 'YIELD-' + str(i) for i in range(trend_window, 0, -1)]
    prev_years = [ 'YEAR-' + str(i) for i in range(trend_window, 0, -1)]
    sel_cols = ['IDREGION', 'FYEAR'] + prev_years + prev_yields
    yield_fts = yield_fts.select(sel_cols)

    return yield_fts

  # Christos, Ante's suggestion
  # - To avoid overfitting, trend estimation could use a window which skips a year in between
  # So a window of 3 will use 6 years of data
  def printYieldTrendRounds(self, df, reg_id, trend_windows=None):
    """Print the sequence of years used for yield trend estimation"""
    reg_years = sorted([yr[0] for yr in df.filter(df['IDREGION'] == reg_id).select('FYEAR').distinct().collect()])
    num_years = len(reg_years)
    if (trend_windows is None):
      trend_windows = self.trend_windows

    for trend_window in trend_windows:
      rounds = (num_years - trend_window)
      if ((self.verbose > 2) and (trend_window == trend_windows[0])):
        print('Trend window', trend_window)
    
      for rd in range(rounds):
        test_year = reg_years[-(rd + 1)]
        start_year = reg_years[-(rd + trend_window + 1)]
        end_year = reg_years[-(rd + 2)]

        if ((self.verbose > 2) and (trend_window == trend_windows[0])):
          print('Round:', rd, 'Test year:', test_year,
                'Trend Window:', start_year, '-', end_year)

  def getLinearYieldTrend(self, window_years, window_yields, pred_year):
    """Linear yield trend prediction"""
    coefs = np.polyfit(window_years, window_yields, 1)
    return float(np.round(coefs[0] * pred_year + coefs[1], 2))

  def getFixedWindowTrendFeatures(self, df, trend_window=None, pred_year=None):
    """Predict the yield trend for each IDREGION and FYEAR using a fixed window"""
    join_cols = ['IDREGION', 'FYEAR']
    if (trend_window is None):
      trend_window = self.trend_windows[0]

    yield_ft_df = self.getTrendWindowYields(df, trend_window)
    if (pred_year is not None):
      yield_ft_df = yield_ft_df.filter(yield_ft_df['FYEAR'] == pred_year)

    pd_yield_ft_df = yield_ft_df.toPandas()
    region_years = pd_yield_ft_df[join_cols].values
    prev_year_cols = ['YEAR-' + str(i) for i in range(1, trend_window + 1)]
    prev_yield_cols = ['YIELD-' + str(i) for i in range(1, trend_window + 1)]
    window_years = pd_yield_ft_df[prev_year_cols].values
    window_yields = pd_yield_ft_df[prev_yield_cols].values

    yield_trend = []
    for i in range(region_years.shape[0]):
      yield_trend.append(self.getLinearYieldTrend(window_years[i, :],
                                                  window_yields[i, :],
                                                  region_years[i, 1]))

    pd_yield_ft_df['YIELD_TREND'] = yield_trend
    return pd_yield_ft_df

  def getFixedWindowTrend(self, df, reg_id, pred_year, trend_window=None):
    """
    Return linear trend prediction for given region and year
    using fixed trend window.
    """
    if (trend_window is None):
      trend_window = self.trend_windows[0]

    reg_df = df.filter((df['IDREGION'] == reg_id) & (df['FYEAR'] <= pred_year))
    pd_yield_ft_df = self.getFixedWindowTrendFeatures(reg_df, trend_window, pred_year)
    if (len(pd_yield_ft_df.index) == 0):
      print('No data to estimate yield trend')
      return None

    if (len(pd_yield_ft_df.index) == 0):
      return None

    reg_year_filter = (df['IDREGION'] == reg_id) & (df['FYEAR'] == pred_year)
    pd_yield_ft_df['ACTUAL'] = df.filter(reg_year_filter).select('YIELD').collect()[0][0]
    pd_yield_ft_df = pd_yield_ft_df.rename(columns={'YIELD_TREND' : 'PREDICTED'})

    return pd_yield_ft_df

  def getL1OutCVPredictions(self, pd_yield_ft_df, trend_window, join_cols, iter):
    """1 iteration of leave-one-out cross-validation"""
    iter_year_cols = ['YEAR-' + str(i) for i in range(1, trend_window + 1) if i != iter]
    iter_yield_cols = ['YIELD-' + str(i) for i in range(1, trend_window + 1) if i != iter]
    window_years = pd_yield_ft_df[iter_year_cols].values
    window_yields = pd_yield_ft_df[iter_yield_cols].values

    # We are going to predict yield value for YEAR-<iter>.
    pred_years = pd_yield_ft_df['YEAR-' + str(iter)].values
    predicted_trend = []
    for i in range(pred_years.shape[0]):
      predicted_trend.append(self.getLinearYieldTrend(window_years[i, :],
                                                      window_yields[i, :],
                                                      pred_years[i]))

    pd_iter_preds = pd_yield_ft_df[join_cols]
    pd_iter_preds['YTRUE' + str(iter)] = pd_yield_ft_df['YIELD-' + str(iter)]
    pd_iter_preds['YPRED' + str(iter)] = predicted_trend

    if (self.verbose > 2):
      print('Leave-one-out cross-validation: iteration', iter)
      print(pd_iter_preds.head(5))

    return pd_iter_preds

  def getL1outRMSE(self, cv_actual, cv_predicted):
    """Compute RMSE for leave-one-out predictions"""
    return float(np.round(np.sqrt(mean_squared_error(cv_actual, cv_predicted)), 2))

  def getMinRMSEIndex(self, cv_rmses):
    """Index of min rmse values"""
    return np.nanargmin(cv_rmses)

  def getL1OutCVRMSE(self, df, trend_window, join_cols, pred_year=None):
    """Run leave-one-out cross-validation and compute RMSE"""
    join_cols = ['IDREGION', 'FYEAR']
    pd_yield_ft_df = self.getFixedWindowTrendFeatures(df, trend_window, pred_year)
    pd_l1out_preds = None
    for i in range(1, trend_window + 1):
      pd_iter_preds = self.getL1OutCVPredictions(pd_yield_ft_df, trend_window,
                                                 join_cols, i)
      if (pd_l1out_preds is None):
        pd_l1out_preds = pd_iter_preds
      else:
        pd_l1out_preds = pd_l1out_preds.merge(pd_iter_preds, on=join_cols)

    region_years = pd_l1out_preds[join_cols].values
    ytrue_cols = ['YTRUE' + str(i) for i in range(1, trend_window + 1)]
    ypred_cols = ['YPRED' + str(i) for i in range(1, trend_window + 1)]
    l1out_ytrue = pd_l1out_preds[ytrue_cols].values
    l1out_ypred = pd_l1out_preds[ypred_cols].values
    cv_rmse = []
    for i in range(region_years.shape[0]):
      cv_rmse.append(self.getL1outRMSE(l1out_ytrue[i, :],
                                       l1out_ypred[i, :]))

    pd_l1out_rmse = pd_yield_ft_df[join_cols]
    pd_l1out_rmse['YIELD_TREND' + str(trend_window)] = pd_yield_ft_df['YIELD_TREND']
    pd_l1out_rmse['CV_RMSE' + str(trend_window)] = cv_rmse

    return pd_l1out_rmse

  def getOptimalTrendWindows(self, df, pred_year=None, trend_windows=None):
    """
    Compute optimal yield trend values based on leave-one-out
    cross validation errors for different trend windows.
    """
    join_cols = ['IDREGION', 'FYEAR']
    if (trend_windows is None):
      trend_windows = self.trend_windows

    pd_tw_rmses = None
    for tw in trend_windows:
      pd_l1out_rmse = self.getL1OutCVRMSE(df, tw, join_cols, pred_year)
      if (pd_tw_rmses is None):
        pd_tw_rmses = pd_l1out_rmse
      else:
        pd_tw_rmses = pd_tw_rmses.merge(pd_l1out_rmse, on=join_cols, how='left')

    if (self.verbose > 2):
      print('Leave-one-out cross-validation: RMSE')
      print(pd_tw_rmses.sort_values(by=join_cols).head(5))

    region_years = pd_tw_rmses[join_cols].values
    tw_rmse_cols = ['CV_RMSE' + str(tw) for tw in trend_windows]
    tw_trend_cols = ['YIELD_TREND' + str(tw) for tw in trend_windows]
    tw_cv_rmses = pd_tw_rmses[tw_rmse_cols].values
    tw_yield_trend = pd_tw_rmses[tw_trend_cols].values

    opt_windows = []
    yield_trend_preds = []
    for i in range(region_years.shape[0]):
      min_rmse_index = self.getMinRMSEIndex(tw_cv_rmses[i, :])
      opt_windows.append(trend_windows[min_rmse_index])
      yield_trend_preds.append(tw_yield_trend[i, min_rmse_index])

    pd_opt_win_df = pd_tw_rmses[join_cols]
    pd_opt_win_df['OPT_TW'] = opt_windows
    pd_opt_win_df['YIELD_TREND'] = yield_trend_preds
    if (self.verbose > 2):
      print('Optimal trend windows')
      print(pd_opt_win_df.sort_values(by=join_cols).head(5))

    return pd_opt_win_df

  def getOptimalWindowTrendFeatures(self, df, trend_windows=None):
    """
    Get previous year yield values and predicted yield trend
    by determining optimal trend window for each region and year.
    NOTE: We have to select the same number of features, so we
    select previous trend_windows[0] yield values.
    """
    join_cols = ['IDREGION', 'FYEAR']
    if (trend_windows is None):
      trend_windows = self.trend_windows

    pd_yield_ft_df = self.getTrendWindowYields(df, trend_windows[0]).toPandas()
    pd_opt_win_df = self.getOptimalTrendWindows(df, trend_windows=trend_windows)
    pd_opt_win_df = pd_opt_win_df.drop(columns=['OPT_TW'])
    pd_yield_ft_df = pd_yield_ft_df.merge(pd_opt_win_df, on=join_cols)

    return pd_yield_ft_df

  def getOptimalWindowTrend(self, df, reg_id, pred_year, trend_windows=None):
    """
    Compute the optimal trend window for given region and year based on
    leave-one-out cross validation errors for different trend windows.
    """
    df = df.filter(df['IDREGION'] == reg_id)
    pd_opt_win_df = self.getOptimalTrendWindows(df, pred_year, trend_windows)
    if (len(pd_opt_win_df.index) == 0):
      return None

    reg_year_filter = (df['IDREGION'] == reg_id) & (df['FYEAR'] == pred_year)
    pd_opt_win_df['ACTUAL'] = df.filter(reg_year_filter).select('YIELD').collect()[0][0]
    pd_opt_win_df = pd_opt_win_df.rename(columns={'YIELD_TREND' : 'PREDICTED'})

    return pd_opt_win_df
