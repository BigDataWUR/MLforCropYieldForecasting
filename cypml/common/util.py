import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

from . import globals

if (globals.test_env == 'pkg'):
  SparkT = globals.SparkT
  SparkF = globals.SparkF

# crop name and id mappings
def cropNameToID(crop_id_dict, crop):
  """
  Return id of given crop. Relies on crop_id_dict.
  Return 0 if crop name is not in the dictionary.
  """
  crop_lcase = crop.lower()
  try:
    crop_id = crop_id_dict[crop_lcase]
  except KeyError as e:
    crop_id = 0

  return crop_id

def cropIDToName(crop_name_dict, crop_id):
  """
  Return crop name for given crop ID. Relies on crop_name_dict.
  Return 'NA' if crop id is not found in the dictionary.
  """
  try:
    crop_name = crop_name_dict[crop_id]
  except KeyError as e:
    crop_name = 'NA'

  return crop_name

def getYear(date_str):
  """Extract year from date in yyyyMMdd or dd/MM/yyyy format."""
  return SparkF.when(SparkF.length(date_str) == 8,
                     SparkF.year(SparkF.to_date(date_str, 'yyyyMMdd')))\
                     .otherwise(SparkF.year(SparkF.to_date(date_str, 'dd/MM/yyyy')))

def getMonth(date_str):
  """Extract month from date in yyyyMMdd or dd/MM/yyyy format."""
  return SparkF.when(SparkF.length(date_str) == 8,
                     SparkF.month(SparkF.to_date(date_str, 'yyyyMMdd')))\
                     .otherwise(SparkF.month(SparkF.to_date(date_str, 'dd/MM/yyyy')))

def getDay(date_str):
  """Extract day from date in yyyyMMdd or dd/MM/yyyy format."""
  return SparkF.when(SparkF.length(date_str) == 8,
                     SparkF.dayofmonth(SparkF.to_date(date_str, 'yyyyMMdd')))\
                     .otherwise(SparkF.dayofmonth(SparkF.to_date(date_str, 'dd/MM/yyyy')))

# 1-10: Dekad 1
# 11-20: Dekad 2
# > 20 : Dekad 3
def getDekad(date_str):
  """Extract dekad from date in YYYYMMDD format."""
  month = getMonth(date_str)
  day = getDay(date_str)
  return SparkF.when(day < 30, (month - 1)* 3 +
                     SparkF.ceil(day/10)).otherwise((month - 1) * 3 + 3)

# Machine Learning Utility Functions

# Hassanat Distance Metric for KNN
# See https://arxiv.org/pdf/1708.04321.pdf
# Code based on https://github.com/BrunoGomesCoelho/hassanat-distance-checker/blob/master/Experiments.ipynb
def hassanatDistance(a, b):
  total = 0
  for a_i, b_i in zip(a, b):
    min_value = min(a_i, b_i)
    max_value = max(a_i, b_i)
    total += 1
    if min_value >= 0:
      total -= (1 + min_value) / (1 + max_value)
    else:
      total -= (1 + min_value + abs(min_value)) / (1 + max_value + abs(min_value))

  return total

# This definition is from the suggested answer to:
# https://stats.stackexchange.com/questions/58391/mean-absolute-percentage-error-mape-in-scikit-learn/294069#294069
def meanAbsolutePercentageError(Y_true, Y_pred):
  """Mean Absolute Percentage Error"""
  Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
  return np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100

def modelRefitMeanVariance(cv_results):
  """
  Custom refit callable for hyperparameter optimization.
  Look at mean and variance of validation scores
  """
  # cv_results structure
  # {
  #   'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
  #   'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
  #   'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
  #   'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
  #   'rank_test_score'    : [2, 4, 3, 1],
  #   'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
  #   'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
  #   'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
  #   'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
  #    ...
  #   'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
  # }

  # For mean_score, higher is better
  # For std_score or variance, lower is better
  # We combine them by using mean_score - std_score
  mean_score = cv_results['mean_test_score']
  std_score = cv_results['std_test_score']
  refit_score = mean_score - std_score

  best_score, best_index = max((val, idx) for (idx, val) in enumerate(refit_score))
  return best_index

def modelRefitTrainValDiff(cv_results):
  """
  Custom refit callable for hyperparameter optimization.
  Look at difference between training and validation errors.
  NOTE: Hyperparameter search must be called with return_train_score=True.
  """
  # cv_results structure
  # {
  #   'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
  #   'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
  #   'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
  #   'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
  #   'rank_test_score'    : [2, 4, 3, 1],
  #   'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
  #   'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
  #   'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
  #   'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
  #    ...
  #   'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
  # }
  mean_test_score = cv_results['mean_test_score']
  mean_train_score = cv_results['mean_train_score']
  mean_score_diff = mean_test_score - mean_train_score

  best_score, best_index = min((val, idx) for (idx, val) in enumerate(mean_score_diff))
  return best_index

def customFitPredict(args):
  """
  We need this because scikit-learn does not support
  cross_val_predict for time series splits.
  """
  X_train = args['X_train']
  Y_train = args['Y_train']
  X_test = args['X_test']
  est = args['estimator']
  fit_params = args['fit_params']

  est.fit(X_train, Y_train, **fit_params)
  return est.predict(X_test)

def printInGroups(items, indices, item_values=None, log_fh=None):
  """Print elements at given indices in groups of 5"""
  num_items = len(indices)
  groups = int(num_items/5) + 1

  items_str = '\n'
  for g in range(groups):
    group_start = g * 5
    group_end = (g + 1) * 5
    if (group_end > num_items):
      group_end = num_items

    group_indices = indices[group_start:group_end]
    for idx in group_indices:
      items_str += str(idx+1) + ': ' + items[idx]
      if (item_values):
        items_str += '=' + item_values[idx]

      if (idx != group_indices[-1]):
          items_str += ', '

    items_str += '\n'

  print(items_str)
  if (log_fh is not None):
    log_fh.write(items_str)

def getPredictionScores(Y_true, Y_predicted, metrics):
  """Get values of metrics for given Y_predicted and Y_true"""
  pred_scores = {}

  for met in metrics:
    score_function = metrics[met]
    met_score = score_function(Y_true, Y_predicted)
    # for RMSE, score_function is mean_squared_error, take square root
    # normalize RMSE
    if (met == 'RMSE'):
      met_score = np.round(100*np.sqrt(met_score)/np.mean(Y_true), 2)
      pred_scores['NRMSE'] = met_score
    # normalize mean absolute errors except MAPE which is already a percentage
    elif ((met == 'MAE') or (met == 'MdAE')):
      met_score = np.round(100*met_score/np.mean(Y_true), 2)
      pred_scores['N' + met] = met_score
    # MAPE, R2, ... : no postprocessing
    else:
      met_score = np.round(met_score, 2)
      pred_scores[met] = met_score

  return pred_scores

def getFilename(crop, yield_trend, early_season, early_season_end,
                country=None, nuts_level=None):
  """Get filename based on input arguments"""
  suffix = crop.replace(' ', '_')

  if (country is not None):
    suffix += '_' + country

  if (nuts_level is not None):
    suffix += '_' + nuts_level

  if (yield_trend):
    suffix += '_trend'
  else:
    suffix += '_notrend'

  if (early_season):
    suffix += '_early' + str(early_season_end)

  return suffix

def getLogFilename(crop, yield_trend, early_season, early_season_end,
                   country=None):
  """Get filename for experiment log"""
  log_file = getFilename(crop, yield_trend, early_season, early_season_end, country)
  return log_file + '.log'

def getFeatureFilename(crop, yield_trend, early_season, early_season_end,
                       country=None):
  """Get unique filename for features"""
  feature_file = 'ft_'
  suffix = getFilename(crop, yield_trend, early_season, early_season_end, country)
  feature_file += suffix
  return feature_file

def getPredictionFilename(crop, yield_trend, early_season, early_season_end,
                          country=None, nuts_level=None):
  """Get unique filename for predictions"""
  pred_file = 'pred_'
  suffix = getFilename(crop, yield_trend, early_season, early_season_end,
                       country, nuts_level)
  pred_file += suffix
  return pred_file

def plotTrend(years, actual_values, trend_values, trend_label):
  """Plot a linear trend and scatter plot of actual values"""
  plt.scatter(years, actual_values, color="blue", marker="o")
  plt.plot(years, trend_values, '--')
  plt.xticks(np.arange(years[0], years[-1] + 1, step=len(years)/5))
  ax = plt.axes()
  plt.xlabel("YEAR")
  plt.ylabel(trend_label)
  plt.title(trend_label + ' Trend by YEAR')
  plt.show()

def plotTrueVSPredicted(actual, predicted):
  """Plot actual and predicted values"""
  fig, ax = plt.subplots()
  ax.scatter(np.asarray(actual), predicted)
  ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
  ax.set_xlabel('Actual')
  ax.set_ylabel('Predicted')
  plt.show()

def plotCVResultsGroup(pd_results_df, score_cols, param_cols):
  """Plot training and validation scores of given parameters"""
  # Metrics can be string or functions, skip them.
  if ('estimator__metric' in param_cols):
    param_cols.remove('estimator__metric')

  fig, ax = plt.subplots(1, len(param_cols), sharex='none', sharey='all',figsize=(20,5))
  fig.suptitle('Score per parameter')
  fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
  for i, p in enumerate(param_cols):
    pd_filtered_df = pd_results_df.copy()
    pd_filtered_df = pd_filtered_df.drop_duplicates(subset=[p])
    pd_param_df = pd_filtered_df[[p] + score_cols]
    if ((pd_filtered_df[p].dtype == 'int64') or (pd_filtered_df[p].dtype == 'float64')):
      pd_param_df = pd_param_df.sort_values(by=[p])

    x = pd_param_df[p].values
    y1 = pd_param_df['MEAN_TEST'].values
    y2 = pd_param_df['MEAN_TRAIN'].values
    e1 = pd_param_df['STD_TEST'].values
    e2 = pd_param_df['STD_TRAIN'].values
    if (len(param_cols) > 1):
      ax[i].errorbar(x, y1, e1, linestyle='--', marker='o', label='test')
      ax[i].errorbar(x, y2, e2, linestyle='solid', marker='o', label='train')
      ax[i].set_xlabel(p.upper())
    else:
      ax.errorbar(x, y1, e1, linestyle='--', marker='o', label='test')
      ax.errorbar(x, y2, e2, linestyle='solid', marker='o', label='train')
      ax.set_xlabel(p.upper())

  plt.legend()
  plt.show()

def plotCVResults(search_cv):
  """Plot training and validation scores of search parameters"""
  score_cols = ['MEAN_TEST', 'MEAN_TRAIN', 'STD_TEST', 'STD_TRAIN']
  pd_results_df = pd.concat([pd.DataFrame(search_cv.cv_results_['params']),
                             pd.DataFrame(search_cv.cv_results_['mean_test_score'],
                                          columns=['MEAN_TEST']),
                             pd.DataFrame(search_cv.cv_results_['std_test_score'],
                                          columns=['STD_TEST']),
                             pd.DataFrame(search_cv.cv_results_['mean_train_score'],
                                          columns=['MEAN_TRAIN']),
                             pd.DataFrame(search_cv.cv_results_['std_train_score'],
                                          columns=['STD_TRAIN'])
                             ], axis=1)

  param_cols = list(pd_results_df.columns)[:-len(score_cols)]
  # remove parameters with fixed value
  del_cols = []
  for p in param_cols:
    x = set(pd_results_df[p].values)
    if (len(x) == 1):
      del_cols.append(p)

  param_cols = [c for c in param_cols if c not in del_cols]
  if (len(param_cols) <= 3):
    plotCVResultsGroup(pd_results_df, score_cols, param_cols)
  else:
    num_groups = int(len(param_cols)/ 3) + 1
    for g in range(num_groups):
      param_cols_group = param_cols[g * 3 : (g + 1) * 3]
      if (not param_cols_group):
        break

      plotCVResultsGroup(pd_results_df, score_cols, param_cols_group)

# Based on
# https://stackoverflow.com/questions/39409866/correlation-heatmap
def plotCorrelation(df, sel_cols):
  corr = df[sel_cols].corr()
  mask = np.zeros_like(corr, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True
  f, ax = plt.subplots(figsize=(20, 18))
  cmap = sns.diverging_palette(220, 10, as_cmap=True)
  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5},
              annot=True, fmt='.1g')
