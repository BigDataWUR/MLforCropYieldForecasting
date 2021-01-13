import pandas as pd
import numpy as np
from joblibspark import register_spark

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import printFeatures
  from ..common.util import getPredictionFilename
  from ..workflow.train_test_split import CYPTrainTestSplitter
  from ..workflow.feature_selection import CYPFeatureSelector
  from ..workflow.algorithm_evaluation import CYPAlgorithmEvaluator

def getValidationSplits(cyp_config, pd_train_df, pd_test_df, log_fh):
  """Split features and label into training and test sets"""
  use_yield_trend = cyp_config.useYieldTrend()
  debug_level = cyp_config.getDebugLevel()

  regions = [reg for reg in pd_train_df['IDREGION'].unique()]
  num_regions = len(regions)

  original_headers = list(pd_train_df.columns.values)
  features = []
  labels = []
  if (use_yield_trend):
    features = original_headers[2:-2]
    labels = original_headers[:2] + original_headers[-2:]
  else:
    features = original_headers[2:-1]
    labels = original_headers[:2] + original_headers[-1:]

  X_train = pd_train_df[features].values
  Y_train = pd_train_df[labels].values

  train_info = '\nTraining Data Size: ' + str(len(pd_train_df.index)) + ' rows'
  train_info += '\nX cols: ' + str(X_train.shape[1]) + ', Y cols: ' + str(Y_train.shape[1])
  train_info += '\n' + pd_train_df.head(5).to_string(index=False)
  log_fh.write(train_info + '\n')
  if (debug_level > 1):
    print(train_info)

  X_test = pd_test_df[features].values
  Y_test = pd_test_df[labels].values

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
  printFeatures(features, indices, log_fh)

  # num_folds for k-fold cv
  num_folds = 5
  custom_cv = num_folds
  if (use_yield_trend):
    cyp_cv_splitter = CYPTrainTestSplitter(cyp_config)
    custom_cv = cyp_cv_splitter.customKFoldValidationSplit(Y_train, num_folds, log_fh)

  # L1 penalty = error (y_pred - y_obs)^2 + alpha * sum (|w_i|) = sparsity regularization
  # L2 penalty = error (y_pred - y_obs)^2 + alpha * sqrt ( sum (w_i^2) ) = weight decay regularization

  result = {
      'X_train' : X_train,
      'Y_train_full' : Y_train,
      'X_test' : X_test,
      'Y_test_full' : Y_test,
      'custom_cv' : custom_cv,
      'features' : features,
  }

  return result

def getMachineLearningPredictions(cyp_config, pd_train_df, pd_test_df, log_fh):
  """Train and evaluate algorithms"""
  metrics = cyp_config.getEvaluationMetrics()
  use_yield_trend = cyp_config.useYieldTrend()
  predict_residuals = cyp_config.predictYieldResiduals()
  debug_level = cyp_config.getDebugLevel()

  # register spark parallel backend
  register_spark()

  eval_info = '\nTraining and Evaluation'
  eval_info += '\n-------------------------'
  log_fh.write(eval_info)
  if (debug_level > 1):
    print(eval_info)

  data_splits = getValidationSplits(cyp_config, pd_train_df, pd_test_df, log_fh)
  X_train = data_splits['X_train']
  Y_train_full = np.copy(data_splits['Y_train_full'])
  X_test = data_splits['X_test']
  Y_test_full = np.copy(data_splits['Y_test_full'])
  features = data_splits['features']
  custom_cv = data_splits['custom_cv']

  alg_summary = {}
  cyp_algeval = CYPAlgorithmEvaluator(cyp_config, custom_cv)
  null_preds = cyp_algeval.getNullMethodPredictions(Y_train_full, Y_test_full, log_fh)

  if (use_yield_trend and predict_residuals):
    result = cyp_algeval.estimateYieldTrendAndDetrend(X_train, Y_train_full[:, -1],
                                                      X_test, Y_test_full[:, -1], features)
    X_train = result['X_train']
    Y_train_full[:, -1] = result['Y_train'][:, 0]
    Y_train_full[:, -2] = result['Y_train_trend'][:, 0]
    X_test = result['X_test']
    Y_test_full[:, -1] = result['Y_test'][:,0]
    Y_test_full[:, -2] = result['Y_test_trend'][:, 0]
    features = result['features']

  # feature selection methods
  num_features = len(features)
  ft_selectors = cyp_config.getFeatureSelectors(X_train, Y_train_full[:, -1],
                                                num_features, custom_cv)
  cyp_ftsel = CYPFeatureSelector(cyp_config, X_train, Y_train_full[:, -1], custom_cv, features)
  ml_preds = cyp_algeval.getMLPredictions(X_train, Y_train_full, X_test, Y_test_full,
                                          cyp_ftsel, ft_selectors, features, log_fh)
  if (use_yield_trend and predict_residuals):
    # NOTE Y_train_full, Y_test_full can get modified above
    ml_preds = cyp_algeval.yieldPredictionsFromResiduals(ml_preds['train'],
                                                         data_splits['Y_train_full'][:, -1],
                                                         ml_preds['test'],
                                                         data_splits['Y_test_full'][:, -1])

  cyp_algeval.evaluateNullMethodPredictions(null_preds['train'], null_preds['test'], alg_summary)
  cyp_algeval.evaluateMLPredictions(ml_preds['train'], ml_preds['test'], alg_summary)
  cyp_algeval.printPredictionDataFrames(ml_preds['train'], ml_preds['test'], log_fh)

  alg_df_columns = ['algorithm']
  for met in metrics:
    alg_df_columns += ['train_' + met, 'test_' + met]

  alg_df = pd.DataFrame.from_dict(alg_summary, orient='index', columns=alg_df_columns)

  eval_summary_info = '\nAlgorithm Evaluation Summary'
  eval_summary_info += '\n-----------------------------'
  eval_summary_info += '\n' + alg_df.to_string(index=False) + '\n'
  log_fh.write(eval_summary_info)
  print(eval_summary_info)

  return ml_preds['test']

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
  output_file = getPredictionFilename(crop, country, nuts_level, use_yield_trend,
                                      early_season_prediction, early_season_end)

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
