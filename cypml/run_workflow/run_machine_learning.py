import pandas as pd
import numpy as np
from joblibspark import register_spark

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import printInGroups
  from ..common.util import getPredictionFilename
  from ..workflow.train_test_split import CYPTrainTestSplitter
  from ..workflow.feature_selection import CYPFeatureSelector
  from ..workflow.algorithm_evaluation import CYPAlgorithmEvaluator

def dropHighlyCorrelatedFeatures(cyp_config, pd_train_df, pd_test_df,
                                 log_fh, corr_method='pearson', corr_thresh=0.95):
  """Plot correlations. Drop columns that are highly correlated."""
  debug_level = cyp_config.getDebugLevel()
  all_cols = list(pd_train_df.columns)[2:]
  avg_cols = [c for c in all_cols if 'avg' in c] + ['YIELD']
  max_cols = [c for c in all_cols if 'max' in c] + ['YIELD']
  lt_th_cols = [c for c in all_cols if 'Z-' in c] + ['YIELD']
  gt_th_cols = [c for c in all_cols if 'Z+' in c] + ['YIELD']
  yt_cols = ['YIELD-' + str(i) for i in range(1, 6)]  + ['YIELD']

  if (debug_level > 2):
    plotCorrelation(pd_train_df, avg_cols)
    plotCorrelation(pd_train_df, max_cols)
    plotCorrelation(pd_train_df, lt_th_cols)
    plotCorrelation(pd_train_df, gt_th_cols)
    plotCorrelation(pd_train_df, yt_cols)

  # drop highly correlated features
  # Based on https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

  corr_columns = [c for c in all_cols if ((c != 'YIELD') and (c != 'YIELD_TREND'))]
  corr_matrix = pd_train_df[corr_columns].corr(method=corr_method).abs()
  ut_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
  ut_matrix = corr_matrix.mask(ut_mask)
  to_drop = [c for c in ut_matrix.columns if any(ut_matrix[c] > corr_thresh)]
  drop_info = '\nDropping highly correlated features'
  drop_info += '\n' + ', '.join(to_drop)

  log_fh.write(drop_info + '\n')
  if ((debug_level > 1) and (to_drop)):
    print(drop_info)

  pd_train_df = pd_train_df.drop(columns=to_drop)
  pd_test_df = pd_test_df.drop(columns=to_drop)

  return pd_train_df, pd_test_df

def getValidationSplits(cyp_config, pd_train_df, pd_test_df, log_fh):
  """Split features and label into training and test sets"""
  use_yield_trend = cyp_config.useYieldTrend()
  use_sample_weights = cyp_config.useSampleWeights()
  debug_level = cyp_config.getDebugLevel()

  regions = [reg for reg in pd_train_df['IDREGION'].unique()]
  num_regions = len(regions)

  original_headers = list(pd_train_df.columns.values)
  features = []
  labels = []
  if (use_yield_trend):
    if (use_sample_weights):
      features = original_headers[2:-3]
      labels = original_headers[:2] + original_headers[-3:-1]
    else:
      features = original_headers[2:-2]
      labels = original_headers[:2] + original_headers[-2:]
  else:
    if (use_sample_weights):
      features = original_headers[2:-2]
      labels = original_headers[:2] + original_headers[-2:-1]
    else:
      features = original_headers[2:-1]
      labels = original_headers[:2] + original_headers[-1:]

  X_train = pd_train_df[features].values
  Y_train = pd_train_df[labels].values
  train_weights = None
  if (use_sample_weights):
    train_weights = pd_train_df['SAMPLE_WEIGHT'].values

  train_info = '\nTraining Data Size: ' + str(len(pd_train_df.index)) + ' rows'
  train_info += '\nX cols: ' + str(X_train.shape[1]) + ', Y cols: ' + str(Y_train.shape[1])
  train_info += '\n' + pd_train_df.head(5).to_string(index=False)
  log_fh.write(train_info + '\n')
  if (debug_level > 1):
    print(train_info)

  X_test = pd_test_df[features].values
  Y_test = pd_test_df[labels].values
  test_weights = None
  if (use_sample_weights):
    test_weights = pd_test_df['SAMPLE_WEIGHT'].values

  test_info = '\nTest Data Size: ' + str(len(pd_test_df.index)) + ' rows'
  test_info += '\nX cols: ' + str(X_test.shape[1]) + ', Y cols: ' + str(Y_test.shape[1])
  test_info += '\n' + pd_test_df.head(5).to_string(index=False)
  log_fh.write(test_info + '\n')
  if (debug_level > 1):
    print(test_info)

  # print feature names
  num_features = len(features)
  indices = [idx for idx in range(num_features)]
  feature_info = '\nAll features'
  feature_info += '\n-------------'
  log_fh.write(feature_info)
  print(feature_info)
  printInGroups(features, indices, log_fh=log_fh)

  # num_folds for k-fold cv
  num_folds = 5
  custom_cv = num_folds
  cv_test_years = []
  if (use_yield_trend):
    cyp_cv_splitter = CYPTrainTestSplitter(cyp_config)
    custom_cv, cv_test_years = cyp_cv_splitter.customKFoldValidationSplit(Y_train, num_folds, log_fh)

  result = {
      'X_train' : X_train,
      'Y_train_full' : Y_train,
      'train_weights' : train_weights,
      'X_test' : X_test,
      'Y_test_full' : Y_test,
      'test_weights' : test_weights,
      'custom_cv' : custom_cv,
      'cv_test_years' : cv_test_years,
      'features' : features,
  }

  return result

def printAlgorithmsEvaluationSummary(cyp_config, null_preds, ml_preds,
                                     log_fh, country_code=None):
  """Print summary of algorithm evaluation"""
  metrics = cyp_config.getEvaluationMetrics()
  country = country_code
  if (country_code is None):
    country = cyp_config.getCountryCode()

  alg_summary = {}
  cyp_algeval = CYPAlgorithmEvaluator(cyp_config)
  cyp_algeval.evaluateNullMethodPredictions(null_preds, alg_summary)
  cyp_algeval.evaluateMLPredictions(ml_preds, alg_summary)
  pd_pred_dfs = [ml_preds['train'], ml_preds['custom_cv'], ml_preds['test']]
  pred_sets_info = ['Training Set', 'Validation Test Set', 'Test Set']
  cyp_algeval.printPredictionDataFrames(pd_pred_dfs, pred_sets_info, log_fh)

  alg_df_columns = ['algorithm']
  for met in metrics:
    alg_df_columns += ['train_' + met, 'cv_' + met, 'test_' + met]

  alg_df = pd.DataFrame.from_dict(alg_summary, orient='index', columns=alg_df_columns)

  eval_summary_info = '\nAlgorithm Evaluation Summary for ' + country
  eval_summary_info += '\n-----------------------------------------'
  eval_summary_info += '\n' + alg_df.to_string(index=False) + '\n'
  log_fh.write(eval_summary_info)
  print(eval_summary_info)

def getMachineLearningPredictions(cyp_config, pd_train_df, pd_test_df, log_fh):
  """Train and evaluate algorithms"""
  metrics = cyp_config.getEvaluationMetrics()
  use_yield_trend = cyp_config.useYieldTrend()
  predict_residuals = cyp_config.predictYieldResiduals()
  use_sample_weights = cyp_config.useSampleWeights()
  alg_names = list(cyp_config.getEstimators().keys())
  debug_level = cyp_config.getDebugLevel()
  country_code = cyp_config.getCountryCode()

  # register spark parallel backend
  register_spark()

  eval_info = '\nTraining and Evaluation'
  eval_info += '\n-------------------------'
  log_fh.write(eval_info)
  if (debug_level > 1):
    print(eval_info)

  data_splits = getValidationSplits(cyp_config, pd_train_df, pd_test_df, log_fh)
  X_train = data_splits['X_train']
  Y_train_full = data_splits['Y_train_full']
  X_test = data_splits['X_test']
  Y_test_full = data_splits['Y_test_full']
  features = data_splits['features']
  custom_cv = data_splits['custom_cv']
  cv_test_years = data_splits['cv_test_years']

  train_weights = None
  test_weights = None
  if (use_sample_weights):
    train_weights = data_splits['train_weights']
    test_weights = data_splits['test_weights']

  cyp_algeval = CYPAlgorithmEvaluator(cyp_config, custom_cv, train_weights, test_weights)
  null_preds = cyp_algeval.getNullMethodPredictions(Y_train_full, Y_test_full,
                                                    cv_test_years, log_fh)

  Y_train_full_n = Y_train_full
  Y_test_full_n = Y_test_full
  if (use_yield_trend and predict_residuals):
    result = cyp_algeval.estimateYieldTrendAndDetrend(X_train, Y_train_full[:, -1],
                                                      X_test, Y_test_full[:, -1], features)
    X_train = result['X_train']
    Y_train_full_n = np.copy(Y_train_full)
    Y_train_full_n[:, -1] = result['Y_train'][:, 0]
    Y_train_full_n[:, -2] = result['Y_train_trend'][:, 0]
    X_test = result['X_test']
    Y_test_full_n = np.copy(Y_test_full)
    Y_test_full_n[:, -1] = result['Y_test'][:,0]
    Y_test_full_n[:, -2] = result['Y_test_trend'][:, 0]
    features = result['features']

  cyp_ftsel = CYPFeatureSelector(cyp_config, X_train, Y_train_full_n[:, -1], features,
                                 custom_cv, train_weights)
  ml_preds = cyp_algeval.getMLPredictions(X_train, Y_train_full_n, X_test, Y_test_full_n,
                                          cv_test_years, cyp_ftsel, features, log_fh)

  if (use_yield_trend and predict_residuals):
    ml_preds = cyp_algeval.yieldPredictionsFromResiduals(ml_preds['train'],
                                                         Y_train_full[:, -1],
                                                         ml_preds['test'],
                                                         Y_test_full[:, -1],
                                                         ml_preds['custom_cv'],
                                                         cv_test_years)

  if (debug_level > 1):
    sel_cols = ['IDREGION', 'FYEAR', 'YIELD'] + ['YIELD_PRED_' + alg for alg in alg_names]
    print('\n', ml_preds['test'][sel_cols].head(5))

  null_preds_train = null_preds['train']
  null_preds_cv = null_preds['custom_cv']
  null_preds_test = null_preds['test']
  ml_preds_train = ml_preds['train']
  ml_preds_cv = ml_preds['custom_cv']
  ml_preds_test = ml_preds['test']

  # print per country evaluation summary
  if (country_code is not None):
    printAlgorithmsEvaluationSummary(cyp_config, null_preds, ml_preds, log_fh)
    ml_preds_test['COUNTRY'] = country_code
  else:
    ml_preds_test['COUNTRY'] = ml_preds_test['IDREGION'].str[:2]
    countries = ml_preds_test['COUNTRY'].unique()
    for c in countries:
      null_preds_country = {
          'train' : null_preds_train[null_preds_train['IDREGION'].str[:2] == c],
          'custom_cv' : null_preds_cv[null_preds_cv['IDREGION'].str[:2] == c],
          'test' : null_preds_test[null_preds_test['IDREGION'].str[:2] == c]
      }

      ml_preds_country = {
          'train' : ml_preds_train[ml_preds_train['IDREGION'].str[:2] == c],
          'custom_cv' : ml_preds_cv[ml_preds_cv['IDREGION'].str[:2] == c],
          'test' : ml_preds_test[ml_preds_test['COUNTRY'] == c]
      }

      printAlgorithmsEvaluationSummary(cyp_config, null_preds_country, ml_preds_country,
                                       log_fh, c)

  return ml_preds_test

def saveMLPredictions(cyp_config, sqlCtx, pd_ml_predictions):
  """Save ML predictions to a CSV file"""
  debug_level = cyp_config.getDebugLevel()
  crop = cyp_config.getCropName()
  country = cyp_config.getCountryCode()
  nuts_level = cyp_config.getNUTSLevel()
  use_yield_trend = cyp_config.useYieldTrend()
  early_season_prediction = cyp_config.earlySeasonPrediction()
  early_season_end = cyp_config.getEarlySeasonEndDekad()
  ml_algs = cyp_config.getEstimators()

  output_path = cyp_config.getOutputPath()
  output_file = getPredictionFilename(crop, use_yield_trend,
                                      early_season_prediction, early_season_end,
                                      country, nuts_level)

  save_pred_path = output_path + '/' + output_file
  if (debug_level > 1):
    print('\nSaving predictions to', save_pred_path + '.csv')
    print(pd_ml_predictions.head(5))

  pd_ml_predictions.to_csv(save_pred_path + '.csv', index=False, header=True)

  # NOTE: In some environments, Spark can write, but pandas cannot.
  # In such cases, use the following code.
  # spark_predictions_df = sqlCtx.createDataFrame(pd_ml_predictions)
  # spark_predictions_df.coalesce(1)\
  #                     .write.option('header','true')\
  #                     .mode("overwrite").csv(save_pred_path)
