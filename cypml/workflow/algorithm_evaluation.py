import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import printFeatures
  from ..common.util import getPredictionScores

class CYPAlgorithmEvaluator:
  def __init__(self, cyp_config, custom_cv):
    self.scaler = cyp_config.getFeatureScaler()
    self.estimators = cyp_config.getEstimators()
    self.custom_cv = custom_cv
    self.cv_metric = cyp_config.getAlgorithmTrainingCVMetric()
    self.metrics = cyp_config.getEvaluationMetrics()
    self.verbose = cyp_config.getDebugLevel()
    self.use_yield_trend = cyp_config.useYieldTrend()
    self.trend_windows = cyp_config.getTrendWindows()
    self.predict_residuals = cyp_config.predictYieldResiduals()

  def setCustomCV(self, custom_cv):
    """Set custom K-Fold validation splits"""
    self.custom_cv = custom_cv

  # Nash-Sutcliffe Model Efficiency
  def nse(self, Y_true, Y_pred):
    """
        Nash Sutcliffe efficiency coefficient
        input:
          Y_pred: predicted
          Y_true: observed
        output:
          nse: Nash Sutcliffe efficient coefficient
        """
    return (1 - np.sum(np.square(Y_pred - Y_true))/np.sum(np.square(Y_true - np.mean(Y_true))))

  def updateAlgorithmsSummary(self, alg_summary, alg_name,
                              train_scores, test_scores):
    """Update algorithms summary with scores for given algorithm"""
    alg_row = [alg_name]
    alg_index = len(alg_summary)
    for met in train_scores:
      alg_row += [train_scores[met], test_scores[met]]

    alg_summary['row' + str(alg_index)] = alg_row

  def createPredictionDataFrames(self, Y_train_pred, Y_test_pred, data_cols):
    """"Create pandas data frames from true and predicted values"""
    pd_train_df = pd.DataFrame(data=Y_train_pred, columns=data_cols)
    pd_test_df = pd.DataFrame(data=Y_test_pred, columns=data_cols)

    return pd_train_df, pd_test_df

  def printPredictionDataFrames(self, pd_train_df, pd_test_df, log_fh):
    """"Print true and predicted values from pandas data frames"""
    train_info = '\nYield Predictions Training Set'
    train_info += '\n--------------------------------'
    train_info += '\n' + pd_train_df.head(6).to_string(index=False)
    log_fh.write(train_info + '\n')
    print(train_info)

    test_info = '\nYield Predictions Test Set'
    test_info += '\n--------------------------------'
    test_info += '\n' + pd_test_df.head(6).to_string(index=False)
    log_fh.write(test_info + '\n')
    print(test_info)

  def getNullMethodPredictions(self, Y_train_full, Y_test_full, log_fh):
    """
    The Null method or poor man's prediction. Y_*_full includes IDREGION, FYEAR.
    If using yield trend, Y_*_full also include YIELD_TREND.
    The null method predicts the YIELD_TREND or the average of the training set.
    """
    Y_train = Y_train_full[:, 2]
    if (self.use_yield_trend):
      Y_train = Y_train_full[:, 3]

    min_yield = np.round(np.min(Y_train), 2)
    max_yield = np.round(np.max(Y_train), 2)
    avg_yield = np.round(np.mean(Y_train), 2)
    median_yield = np.round(np.median(np.ravel(Y_train)), 2)

    null_method_label = 'Null Method: '
    if (self.use_yield_trend):
      null_method_label += 'Predicting linear yield trend:'
      data_cols = ['IDREGION', 'FYEAR', 'YIELD_TREND', 'YIELD']
      pd_train_df, pd_test_df = self.createPredictionDataFrames(Y_train_full, Y_test_full,
                                                                data_cols)
    else:
      Y_train_full_n = np.insert(Y_train_full, 2, avg_yield, axis=1)
      Y_test_full_n = np.insert(Y_test_full, 2, avg_yield, axis=1)
      null_method_label += 'Predicting average of the training set:'
      data_cols = ['IDREGION', 'FYEAR', 'YIELD_PRED', 'YIELD']
      pd_train_df, pd_test_df = self.createPredictionDataFrames(Y_train_full_n, Y_test_full_n,
                                                                data_cols)

    null_method_info = '\n' + null_method_label
    null_method_info += '\nMin Yield: ' + str(min_yield) + ', Max Yield: ' + str(max_yield)
    null_method_info += '\nMedian Yield: ' + str(median_yield) + ', Mean Yield: ' + str(avg_yield)
    log_fh.write(null_method_info + '\n')
    print(null_method_info)
    self.printPredictionDataFrames(pd_train_df, pd_test_df, log_fh)

    result = {
        'train' : pd_train_df,
        'test' : pd_test_df,
    }

    return result

  def evaluateNullMethodPredictions(self, pd_train_df, pd_test_df, alg_summary):
    """Evaluate the predictions of the null method and add an entry to alg_summary"""
    Y_train = pd_train_df['YIELD'].values
    Y_test = pd_test_df['YIELD'].values

    if (self.use_yield_trend):  
      alg_name = 'trend'
      Y_pred_train = pd_train_df['YIELD_TREND'].values
      Y_pred_test = pd_test_df['YIELD_TREND'].values
    else:
      alg_name = 'average'
      Y_pred_train = pd_train_df['YIELD_PRED'].values
      Y_pred_test = pd_test_df['YIELD_PRED'].values

    train_scores = getPredictionScores(Y_train, Y_pred_train, self.metrics)
    test_scores = getPredictionScores(Y_test, Y_pred_test, self.metrics)
    self.updateAlgorithmsSummary(alg_summary, alg_name, train_scores, test_scores)

  def getYieldTrendML(self, X_train, Y_train, X_test):
    """
    Predict yield trend using a linear model.
    No need to scale features. They are all yield values.
    """
    est = Ridge(alpha=1, random_state=42, max_iter=1000,
                copy_X=True, fit_intercept=True)
    est_param_grid = dict(alpha=[1e-3, 1e-2, 1e-1, 1, 10])

    grid_search = GridSearchCV(estimator=est, param_grid=est_param_grid,
                               scoring=self.cv_metric, cv=self.custom_cv)
    X_train_copy = np.copy(X_train)
    with parallel_backend('spark', n_jobs=-1):
      grid_search.fit(X_train_copy, np.ravel(Y_train))

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    if (self.verbose > 1):
      for param in est_param_grid:
        print(param + '=', best_params[param])

    Y_pred_train = np.reshape(best_estimator.predict(X_train), (X_train.shape[0], 1))
    Y_pred_test = np.reshape(best_estimator.predict(X_test), (X_test.shape[0], 1))

    return Y_pred_train, Y_pred_test

  def estimateYieldTrendAndDetrend(self, X_train, Y_train, X_test, Y_test, features):
    """Estimate yield trend using machine learning and detrend"""
    trend_window = self.trend_windows[0]
    # NOTE assuming previous years' yield values are at the end
    X_train_trend = X_train[:, -trend_window:]
    X_test_trend = X_test[:, -trend_window:]

    Y_train_trend, Y_test_trend = self.getYieldTrendML(X_train_trend, Y_train, X_test_trend)
    # New features exclude previous years' yield and include YIELD_TREND
    features_n = features[:-trend_window] + ['YIELD_TREND']
    X_train_n = np.append(X_train[:, :-trend_window], Y_train_trend, axis=1)
    X_test_n = np.append(X_test[:, :-trend_window], Y_test_trend, axis=1)
    Y_train_res = np.reshape(Y_train, (X_train.shape[0], 1)) - Y_train_trend
    Y_test_res = np.reshape(Y_test, (X_test.shape[0], 1)) - Y_test_trend

    result =  {
        'X_train' : X_train_n,
        'Y_train' : Y_train_res,
        'Y_train_trend' : Y_train_trend,
        'X_test' : X_test_n,
        'Y_test' : Y_test_res,
        'Y_test_trend' : Y_test_trend,
        'features' : features_n,
    }

    return result

  def yieldPredictionsFromResiduals(self, pd_train_df, Y_train, pd_test_df, Y_test):
    """Predictions are residuals. Add trend back to get yield predictions."""
    pd_train_df['YIELD_RES'] = pd_train_df['YIELD']
    pd_train_df['YIELD'] = Y_train
    pd_test_df['YIELD_RES'] = pd_test_df['YIELD']
    pd_test_df['YIELD'] = Y_test
    
    for alg in self.estimators:
      pd_train_df['YIELD_RES_PRED_' + alg] = pd_train_df['YIELD_PRED_' + alg]
      pd_train_df['YIELD_PRED_' + alg] = pd_train_df['YIELD_RES_PRED_' + alg] + pd_train_df['YIELD_TREND']
      pd_test_df['YIELD_RES_PRED_' + alg] = pd_test_df['YIELD_PRED_' + alg]
      pd_test_df['YIELD_PRED_' + alg] = pd_test_df['YIELD_RES_PRED_' + alg] + pd_test_df['YIELD_TREND']

    sel_cols = ['IDREGION', 'FYEAR', 'YIELD_TREND', 'YIELD_RES']
    for alg in self.estimators:
      sel_cols += ['YIELD_RES_PRED_' + alg, 'YIELD_PRED_' + alg]

    sel_cols.append('YIELD')
    result = {
        'train' : pd_train_df[sel_cols],
        'test' : pd_test_df[sel_cols],
    }

    return result

  def trainAndTest(self, X_train, Y_train, X_test,
                   est, est_name, est_param_grid):
    """
    Use k-fold validation to tune hyperparameters and evaluate performance.
    """
    if (self.verbose > 1):
      print('\nEstimator', est_name)
      print('---------------------------')
      print(est)

    pipeline = Pipeline([("scaler", self.scaler), ("estimator", est)])
    grid_search = GridSearchCV(estimator=pipeline, param_grid=est_param_grid,
                               scoring=self.cv_metric, cv=self.custom_cv)
    X_train_copy = np.copy(X_train)
    with parallel_backend('spark', n_jobs=-1):
      grid_search.fit(X_train_copy, np.ravel(Y_train))

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    if (self.verbose > 1):
      for param in est_param_grid:
        print(param + '=', best_params[param])

    Y_pred_train = np.reshape(best_estimator.predict(X_train), (X_train.shape[0], 1))
    Y_pred_test = np.reshape(best_estimator.predict(X_test), (X_test.shape[0], 1))

    return Y_pred_train, Y_pred_test

  def combineAlgorithmPredictions(self, pd_ml_predictions, pd_alg_predictions, alg):
    """Combine predictions of ML algorithms."""
    join_cols = ['IDREGION', 'FYEAR']

    if (pd_ml_predictions is None):
      pd_ml_predictions = pd_alg_predictions
      pd_ml_predictions = pd_ml_predictions.rename(columns={'YIELD_PRED': 'YIELD_PRED_' + alg })
    else:
      pd_alg_predictions = pd_alg_predictions[join_cols + ['YIELD_PRED']]
      pd_ml_predictions = pd_ml_predictions.merge(pd_alg_predictions, on=join_cols)
      pd_ml_predictions = pd_ml_predictions.rename(columns={'YIELD_PRED': 'YIELD_PRED_' + alg })
      # Put YIELD at the end
      all_cols = list(pd_ml_predictions.columns)
      col_order = all_cols[:-2] + ['YIELD_PRED_' + alg, 'YIELD']
      pd_ml_predictions = pd_ml_predictions[col_order]

    return pd_ml_predictions

  def getMLPredictions(self, X_train, Y_train_full, X_test, Y_test_full,
                       cyp_ftsel, ft_selectors, features, log_fh):
    """Train and evaluate crop yield prediction algorithms"""
    Y_train = Y_train_full[:, -1]
    Y_test = Y_test_full[:, -1]
    pd_test_predictions = None
    pd_train_predictions = None

    # feature selection frequency
    # NOTE must be in sync with crop calendar periods
    feature_selection_counts = {
        'static' : {},
        'p0' : {},
        'p1' : {},
        'p2' : {},
        'p3' : {},
        'p4' : {},
        'p5' : {},
    }

    for est_name in self.estimators:
      # feature selection
      est = self.estimators[est_name]['estimator']
      param_grid = self.estimators[est_name]['fs_param_grid']
      selected_indices = cyp_ftsel.selectOptimalFeatures(ft_selectors,
                                                         est, est_name, param_grid,
                                                         log_fh)
      sel_fts_info = '\nSelected Features:'
      sel_fts_info += '\n-------------------'
      log_fh.write(sel_fts_info)
      print(sel_fts_info)
      printFeatures(features, selected_indices, log_fh)

      # update feature selection counts
      for idx in selected_indices:
        ft_count = 0
        ft = features[idx]
        ft_period = None
        for p in feature_selection_counts:
          if p in ft:
            ft_period = p

        if (ft_period is None):
          ft_period = 'static'

        if (ft in feature_selection_counts[ft_period]):
          ft_count = feature_selection_counts[ft_period][ft]

        feature_selection_counts[ft_period][ft] = ft_count + 1

      X_train_sel = X_train[:, selected_indices]
      X_test_sel = X_test[:, selected_indices]

      # Training and testing
      param_grid = self.estimators[est_name]['param_grid']
      # yield/yield residual predictions
      Y_pred_train, Y_pred_test = self.trainAndTest(X_train_sel, Y_train, X_test_sel,
                                                    est, est_name, param_grid)
      data_cols = ['IDREGION', 'FYEAR']
      if (self.use_yield_trend):
        data_cols.append('YIELD_TREND')
        Y_train_full_n = np.insert(Y_train_full, 3, Y_pred_train[:, 0], axis=1)
        Y_test_full_n = np.insert(Y_test_full, 3, Y_pred_test[:, 0], axis=1)
      else:
        Y_train_full_n = np.insert(Y_train_full, 2, Y_pred_train[:, 0], axis=1)
        Y_test_full_n = np.insert(Y_test_full, 2, Y_pred_test[:, 0], axis=1)

      data_cols += ['YIELD_PRED', 'YIELD']
      pd_train_df, pd_test_df = self.createPredictionDataFrames(Y_train_full_n, Y_test_full_n, data_cols)
      pd_train_predictions = self.combineAlgorithmPredictions(pd_train_predictions, pd_train_df, est_name)
      pd_test_predictions = self.combineAlgorithmPredictions(pd_test_predictions, pd_test_df, est_name)

    ft_counts_info = '\nFeature Selection Frequencies'
    ft_counts_info += '\n-------------------------------'
    for ft_period in feature_selection_counts:
      ft_count_str = ft_period + ': '
      for ft in sorted(feature_selection_counts[ft_period],
                       key=feature_selection_counts[ft_period].get, reverse=True):
        ft_count_str += ft + '(' + str(feature_selection_counts[ft_period][ft]) + '), '

      if (len(feature_selection_counts[ft_period]) > 0):
        # drop ', ' from the end
        ft_count_str = ft_count_str[:-2]

      ft_counts_info += '\n' + ft_count_str

    ft_counts_info += '\n'
    log_fh.write(ft_counts_info)
    if (self.verbose > 1):
      print(ft_counts_info)

    result = {
        'train' : pd_train_predictions,
        'test' : pd_test_predictions,
    }

    return result

  def evaluateMLPredictions(self, pd_train_predictions, pd_test_predictions, alg_summary):
    """Evaluate predictions of ML algorithms and add entries to alg_summary."""
    Y_train = pd_train_predictions['YIELD'].values
    Y_test = pd_test_predictions['YIELD'].values

    for alg in self.estimators:
      alg_col = 'YIELD_PRED_' + alg
      Y_pred_train = pd_train_predictions[alg_col].values
      Y_pred_test = pd_test_predictions[alg_col].values
      train_scores = getPredictionScores(Y_train, Y_pred_train, self.metrics)
      test_scores = getPredictionScores(Y_test, Y_pred_test, self.metrics)

      self.updateAlgorithmsSummary(alg_summary, alg, train_scores, test_scores)
