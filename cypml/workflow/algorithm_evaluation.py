import numpy as np
import pandas as pd
from copy import deepcopy
import multiprocessing as mp

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.utils import parallel_backend

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import printInGroups
  from ..common.util import getPredictionScores
  from ..common.util import customFitPredict

class CYPAlgorithmEvaluator:
  def __init__(self, cyp_config, custom_cv=None,
               train_weights=None, test_weights=None):
    self.scaler = cyp_config.getFeatureScaler()
    self.estimators = cyp_config.getEstimators()
    self.custom_cv = custom_cv
    self.cv_metric = cyp_config.getAlgorithmTrainingCVMetric()
    self.train_weights = train_weights
    self.test_weights = test_weights
    self.metrics = cyp_config.getEvaluationMetrics()
    self.verbose = cyp_config.getDebugLevel()
    self.use_yield_trend = cyp_config.useYieldTrend()
    self.trend_windows = cyp_config.getTrendWindows()
    self.predict_residuals = cyp_config.predictYieldResiduals()
    self.retrain_per_test_year = cyp_config.retrainPerTestYear()
    self.use_sample_weights = cyp_config.useSampleWeights()

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

  def updateAlgorithmsSummary(self, alg_summary, alg_name, scores_list):
    """Update algorithms summary with scores for given algorithm"""
    alg_row = [alg_name]
    alg_index = len(alg_summary)
    assert (len(scores_list) > 0)
    for met in scores_list[0]:
      for pred_scores in scores_list:
        alg_row.append(pred_scores[met])

    alg_summary['row' + str(alg_index)] = alg_row

  def createPredictionDataFrames(self, Y_pred_arrays, data_cols):
    """"Create pandas data frames from true and predicted values"""
    pd_pred_dfs = []
    for ar in Y_pred_arrays:
      pd_df = pd.DataFrame(data=ar, columns=data_cols)
      pd_pred_dfs.append(pd_df)

    return pd_pred_dfs

  def printPredictionDataFrames(self, pd_pred_dfs, pred_set_info, log_fh):
    """"Print true and predicted values from pandas data frames"""
    for i in range(len(pd_pred_dfs)):
      pd_df = pd_pred_dfs[i]
      set_info = pred_set_info[i]
      df_info = '\n Yield Predictions ' + set_info
      df_info += '\n--------------------------------'
      df_info += '\n' + pd_df.head(6).to_string(index=False)
      log_fh.write(df_info + '\n')
      print(df_info)

  def getNullMethodPredictions(self, Y_train_full, Y_test_full, cv_test_years, log_fh):
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
    cv_test_years = np.array(cv_test_years)

    null_method_label = 'Null Method: '
    if (self.use_yield_trend):
      null_method_label += 'Predicting linear yield trend:'
      data_cols = ['IDREGION', 'FYEAR', 'YIELD_TREND', 'YIELD']
      Y_cv_full = Y_train_full[np.in1d(Y_train_full[:, 1], cv_test_years)]
      Y_pred_arrays = [Y_train_full, Y_cv_full, Y_test_full]
    else:
      Y_train_full_n = np.insert(Y_train_full, 2, avg_yield, axis=1)
      Y_test_full_n = np.insert(Y_test_full, 2, avg_yield, axis=1)
      null_method_label += 'Predicting average of the training set:'
      data_cols = ['IDREGION', 'FYEAR', 'YIELD_PRED', 'YIELD']
      Y_cv_full = Y_train_full_n[np.in1d(Y_train_full_n[:, 1], cv_test_years)]
      Y_pred_arrays = [Y_train_full_n, Y_cv_full, Y_test_full_n]

    pd_pred_dfs = self.createPredictionDataFrames(Y_pred_arrays, data_cols)
    null_method_info = '\n' + null_method_label
    null_method_info += '\nMin Yield: ' + str(min_yield) + ', Max Yield: ' + str(max_yield)
    null_method_info += '\nMedian Yield: ' + str(median_yield) + ', Mean Yield: ' + str(avg_yield)
    log_fh.write(null_method_info + '\n')
    print(null_method_info)
    pred_set_info = ['Training Set', 'Validation Test Set', 'Test Set']
    self.printPredictionDataFrames(pd_pred_dfs, pred_set_info, log_fh)

    result = {
        'train' : pd_pred_dfs[0],
        'custom_cv' : pd_pred_dfs[1],
        'test' : pd_pred_dfs[2],
    }

    return result

  def evaluateNullMethodPredictions(self, pd_pred_dfs, alg_summary):
    """Evaluate predictions of the Null method"""
    if (self.use_yield_trend):
      alg_name = 'trend'
      pred_col_name = 'YIELD_TREND'
    else:
      alg_name = 'average'
      pred_col_name = 'YIELD_PRED'

    scores_list = []
    for pred_set in pd_pred_dfs:
      pred_df = pd_pred_dfs[pred_set]
      Y_true = pred_df['YIELD'].values
      Y_pred = pred_df[pred_col_name].values
      pred_scores = getPredictionScores(Y_true, Y_pred, self.metrics)
      scores_list.append(pred_scores)

    self.updateAlgorithmsSummary(alg_summary, alg_name, scores_list)

  def getYieldTrendML(self, X_train, Y_train, X_test):
    """
    Predict yield trend using a linear model.
    No need to scale features. They are all yield values.
    """
    est = Ridge(alpha=1, random_state=42, max_iter=1000,
                copy_X=True, fit_intercept=True)
    param_space = dict(estimator__alpha=loguniform(1e-1, 1e+2))
    pipeline = Pipeline([("scaler", self.scaler), ("estimator", est)])
    nparams_sampled = pow(3, len(param_space))

    X_train_copy = np.copy(X_train)
    rand_search = RandomizedSearchCV(estimator=pipeline,
                                     param_distributions=param_space,
                                     n_iter=nparams_sampled,
                                     scoring=self.cv_metric,
                                     cv=self.custom_cv,
                                     return_train_score=True,
                                     refit=modelRefitMeanVariance)

    fit_params = {}
    if (self.use_sample_weights):
      fit_params = { 'estimator__sample_weight' : self.train_weights }

    with parallel_backend('spark', n_jobs=-1):
      rand_search.fit(X_train_copy, np.ravel(Y_train), **fit_params)

    best_params = rand_search.best_params_
    best_estimator = rand_search.best_estimator_
    if (self.verbose > 1):
      print('\nYield Trend: Ridge best parameters:')
      for param in param_space:
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

  def yieldPredictionsFromResiduals(self, pd_train_df, Y_train, pd_test_df, Y_test,
                                    pd_cv_df, cv_test_years):
    """Predictions are residuals. Add trend back to get yield predictions."""
    pd_train_df['YIELD_RES'] = pd_train_df['YIELD']
    pd_train_df['YIELD'] = Y_train
    Y_custom_cv = pd_train_df[pd_train_df['FYEAR'].isin(cv_test_years)]['YIELD'].values
    pd_cv_df['YIELD_RES'] = pd_cv_df['YIELD']
    pd_cv_df['YIELD'] = Y_custom_cv
    pd_test_df['YIELD_RES'] = pd_test_df['YIELD']
    pd_test_df['YIELD'] = Y_test

    for alg in self.estimators:
      pd_train_df['YIELD_RES_PRED_' + alg] = pd_train_df['YIELD_PRED_' + alg]
      pd_train_df['YIELD_PRED_' + alg] = pd_train_df['YIELD_RES_PRED_' + alg] + pd_train_df['YIELD_TREND']
      pd_test_df['YIELD_RES_PRED_' + alg] = pd_test_df['YIELD_PRED_' + alg]
      pd_test_df['YIELD_PRED_' + alg] = pd_test_df['YIELD_RES_PRED_' + alg] + pd_test_df['YIELD_TREND']
      pd_cv_df['YIELD_RES_PRED_' + alg] = pd_cv_df['YIELD_PRED_' + alg]
      pd_cv_df['YIELD_PRED_' + alg] = pd_cv_df['YIELD_RES_PRED_' + alg] + pd_cv_df['YIELD_TREND']

    sel_cols = ['IDREGION', 'FYEAR', 'YIELD_TREND', 'YIELD_RES']
    for alg in self.estimators:
      sel_cols += ['YIELD_RES_PRED_' + alg, 'YIELD_PRED_' + alg]

    sel_cols.append('YIELD')
    result = {
        'train' : pd_train_df[sel_cols],
        'custom_cv' : pd_cv_df[sel_cols],
        'test' : pd_test_df[sel_cols],
    }

    return result

  def updateFeatureSelectionInfo(self, est_name, features, selected_indices,
                                 ft_selection_counts, ft_importances, log_fh):
    """
    Update feature selection counts.
    Print selected features and importance.
    """
    # update feature selection counts
    for idx in selected_indices:
      ft_count = 0
      ft = features[idx]
      ft_period = 'static'
      for p in ft_selection_counts:
        if p in ft:
          ft_period = p

      if (ft in ft_selection_counts[ft_period]):
        ft_count = ft_selection_counts[ft_period][ft]

      ft_selection_counts[ft_period][ft] = ft_count + 1

    if (ft_importances is not None):
      ft_importance_indices = []
      ft_importance_values = [0.0 for i in range(len(features))]
      for idx in reversed(np.argsort(ft_importances)):
        ft_importance_indices.append(selected_indices[idx])
        ft_importance_values[selected_indices[idx]] = str(np.round(ft_importances[idx], 2))

      ft_importance_info = '\nSelected features with importance:'
      ft_importance_info += '\n----------------------------------'
      log_fh.write(ft_importance_info)
      print(ft_importance_info)
      printInGroups(features, ft_importance_indices, ft_importance_values, log_fh)
    else:
      sel_fts_info = '\nSelected Features:'
      sel_fts_info += '\n-------------------'
      log_fh.write(sel_fts_info)
      print(sel_fts_info)
      printInGroups(features, selected_indices, log_fh=log_fh)

  def getCustomCVPredictions(self, est_name, best_est,
                             X_train, Y_train, Y_cv_full):
    """Get predictions for custom cv test years"""
    Y_pred_cv = np.zeros(Y_cv_full.shape[0])
    fit_predict_args = []
    for i in range(len(self.custom_cv)):
      cv_train_idxs, cv_test_idxs = self.custom_cv[i]
      sample_weights = None
      if (self.use_sample_weights):
        sample_weights = np.copy(self.train_weights[cv_train_idxs])

      fit_params = {}
      if (self.use_sample_weights and (est_name != 'KNN')):
        fit_params = { 'estimator__sample_weight' : sample_weights }

      fit_predict_args.append(
          {
              'X_train' : np.copy(X_train[cv_train_idxs, :]),
              'Y_train' : np.copy(Y_train[cv_train_idxs]),
              'X_test' : np.copy(X_train[cv_test_idxs, :]),
              'fit_params' : fit_params,
              'estimator' : deepcopy(best_est),
          }
      )

    pool = mp.Pool(len(self.custom_cv))
    Y_preds = pool.map(customFitPredict, fit_predict_args)
    for i in range(len(self.custom_cv)):
      cv_train_idxs, cv_test_idxs = self.custom_cv[i]
      Y_pred_cv[cv_test_idxs] = Y_preds[i]

    # clean up
    pool.close()
    pool.join()

    return Y_pred_cv

  def getPerTestYearPredictions(self, est_name, best_est,
                                X_train, Y_train_full,
                                X_test, Y_test_full, test_years):
    """For each test year, fit best_est on all previous years and predict"""
    Y_train = Y_train_full[:, -1]
    Y_test = Y_test_full[:, -1]
    Y_pred_test = np.zeros(Y_test_full.shape[0])

    # For the first test year, X_train and Y_train do not change. No need to refit.
    test_indexes = np.where(Y_test_full[:, 1] == test_years[0])[0]
    Y_pred_first_yr = best_est.predict(X_test[test_indexes, :])
    Y_pred_test[test_indexes] = Y_pred_first_yr

    if (est_name == 'GBDT'):
      best_est.named_steps['estimator'].set_params(**{ 'warm_start' : True })

    fit_predict_args = []
    for i in range(1, len(test_years)):
      extra_train_years = test_years[:i]
      test_indexes = np.where(Y_test_full[:, 1] == test_years[i])[0]
      sample_weights = None
      if (self.use_sample_weights):
        sample_weights = self.train_weights

      train_indexes_n = np.ravel(np.nonzero(np.isin(Y_test_full[:, 1], extra_train_years)))
      X_train_n = np.append(X_train, X_test[train_indexes_n, :], axis=0)
      Y_train_n = np.append(Y_train, Y_test[train_indexes_n])
      if (self.use_sample_weights):
        sample_weights = np.append(sample_weights, self.test_weights[train_indexes_n], axis=0)

      fit_params = {}
      if (self.use_sample_weights and (est_name != 'KNN')):
        fit_params['estimator__sample_weight'] = sample_weights

      fit_predict_args.append(
          {
              'X_train' : np.copy(X_train_n),
              'Y_train' : np.copy(Y_train_n),
              'X_test' : np.copy(X_test[test_indexes, :]),
              'fit_params' : fit_params,
              'estimator' : deepcopy(best_est),
          }
      )

    pool = mp.Pool(len(test_years) - 1)
    Y_preds = pool.map(customFitPredict, fit_predict_args)
    for i in range(1, len(test_years)):
      test_indexes = np.where(Y_test_full[:, 1] == test_years[i])[0]
      Y_pred_test[test_indexes] = Y_preds[i-1]

    # clean up
    pool.close()
    pool.join()

    return Y_pred_test

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
                       cv_test_years, cyp_ftsel, features, log_fh):
    """Train and evaluate crop yield prediction algorithms"""
    # Y_*_full
    # IDREGION, FYEAR, YIELD_TREND, YIELD_PRED_Ridge, ..., YIELD_PRED_GBDT, YIELD
    # NL11, NL12, NL13 (some regions can have missing values)
    # 1999, ..., 2011 => training
    # 2012, ..., 2018 => Test
    # We need to aggregate to national level. Need predictions from all regions.
    # cv_test_years
    # 1999, ..., 2006 => 2007
    # 1999, ..., 2007 => 2008
    # ...
    # cv_test_years : [2007, 2008, ..., 2011]

    Y_train = Y_train_full[:, -1]
    train_years = sorted(np.unique(Y_train_full[:, 1]))
    Y_cv_full = np.copy(Y_train_full)
    Y_test = Y_test_full[:, -1]
    test_years = sorted(np.unique(Y_test_full[:, 1]))

    pd_test_predictions = None
    pd_cv_predictions = None
    pd_train_predictions = None

    # feature selection frequency
    # NOTE must be in sync with crop calendar periods
    ft_selection_counts = {
        'static' : {}, 'p0' : {}, 'p1' : {}, 'p2' : {}, 'p3' : {}, 'p4' : {}, 'p5' : {},
    }

    for est_name in self.estimators:
      # feature selection
      est = self.estimators[est_name]['estimator']
      param_space = self.estimators[est_name]['param_space']
      sel_indices, best_est = cyp_ftsel.selectOptimalFeatures(est, est_name, param_space, log_fh)

      # feature importance
      ft_importances = None
      if ((est_name == 'Ridge') or (est_name == 'Lasso')):
        ft_importances = best_est.named_steps['estimator'].coef_
      elif (est_name == 'SVR'):
        try:
          ft_importances = np.ravel(best_est.named_steps['estimator'].coef_)
        except AttributeError as e:
          ft_importances = None
      elif ((est_name == 'GBDT') or (est_name == 'RF') or (est_name == 'ERT')):
        ft_importances = best_est.named_steps['estimator'].feature_importances_

      self.updateFeatureSelectionInfo(est_name, features, sel_indices, ft_selection_counts,
                                      ft_importances, log_fh)

      # Predictions
      Y_pred_train = best_est.predict(X_train)
      Y_pred_test = best_est.predict(X_test)

      # custom cv predictions for cv metrics
      Y_pred_cv = self.getCustomCVPredictions(est_name, best_est,
                                              X_train, Y_train, Y_cv_full)

      # per test year predictions
      # 1999, ..., 2011 => 2012
      # 1999, ..., 2012 => 2013
      # ...
      if (self.retrain_per_test_year):
        Y_pred_test = self.getPerTestYearPredictions(est_name, best_est,
                                                     X_train, Y_train_full,
                                                     X_test, Y_test_full, test_years)

      data_cols = ['IDREGION', 'FYEAR']
      if (self.use_yield_trend):
        data_cols.append('YIELD_TREND')
        Y_train_full_n = np.insert(Y_train_full, 3, Y_pred_train, axis=1)
        Y_cv_full_n = np.insert(Y_cv_full, 3, Y_pred_cv, axis=1)
        Y_test_full_n = np.insert(Y_test_full, 3, Y_pred_test, axis=1)
      else:
        Y_train_full_n = np.insert(Y_train_full, 2, Y_pred_train, axis=1)
        Y_cv_full_n = np.insert(Y_cv_full, 2, Y_pred_cv, axis=1)
        Y_test_full_n = np.insert(Y_test_full, 2, Y_pred_test, axis=1)

      data_cols += ['YIELD_PRED', 'YIELD']
      Y_cv_full_n = Y_cv_full_n[np.in1d(Y_cv_full_n[:, 1], cv_test_years)]
      Y_pred_arrays = [Y_train_full_n, Y_cv_full_n, Y_test_full_n]
      pd_pred_dfs = self.createPredictionDataFrames(Y_pred_arrays, data_cols)
      pd_train_predictions = self.combineAlgorithmPredictions(pd_train_predictions,
                                                              pd_pred_dfs[0], est_name)
      pd_cv_predictions = self.combineAlgorithmPredictions(pd_cv_predictions,
                                                           pd_pred_dfs[1], est_name)
      pd_test_predictions = self.combineAlgorithmPredictions(pd_test_predictions,
                                                             pd_pred_dfs[2], est_name)

    ft_counts_info = '\nFeature Selection Frequencies'
    ft_counts_info += '\n-------------------------------'
    for ft_period in ft_selection_counts:
      ft_count_str = ft_period + ': '
      for ft in sorted(ft_selection_counts[ft_period],
                       key=ft_selection_counts[ft_period].get, reverse=True):
        ft_count_str += ft + '(' + str(ft_selection_counts[ft_period][ft]) + '), '

      if (len(ft_selection_counts[ft_period]) > 0):
        # drop ', ' from the end
        ft_count_str = ft_count_str[:-2]

      ft_counts_info += '\n' + ft_count_str

    ft_counts_info += '\n'
    log_fh.write(ft_counts_info)
    if (self.verbose > 1):
      print(ft_counts_info)

    result = {
        'train' : pd_train_predictions,
        'custom_cv' : pd_cv_predictions,
        'test' : pd_test_predictions,
    }

    return result

  def evaluateMLPredictions(self, pd_pred_dfs, alg_summary):
    """Evaluate predictions of ML algorithms and add entries to alg_summary."""
    for alg in self.estimators:
      pred_col = 'YIELD_PRED_' + alg
      scores_list = []
      for pred_set in pd_pred_dfs:
        pred_df = pd_pred_dfs[pred_set]
        Y_true = pred_df['YIELD'].values
        Y_pred = pred_df[pred_col].values
        pred_scores = getPredictionScores(Y_true, Y_pred, self.metrics)
        scores_list.append(pred_scores)

      self.updateAlgorithmsSummary(alg_summary, alg, scores_list)
