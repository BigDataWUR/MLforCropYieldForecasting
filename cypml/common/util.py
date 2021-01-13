import numpy as np
import pandas as pd
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

# This definition is from the suggested answer to:
# https://stats.stackexchange.com/questions/58391/mean-absolute-percentage-error-mape-in-scikit-learn/294069#294069
def mean_absolute_percentage_error(Y_true, Y_pred):
  """Mean Absolute Percentage Error"""
  Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
  return np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100

def printFeatures(features, indices, log_fh=None):
  """Print names of features in groups of 5"""
  num_features = len(indices)
  groups = int(num_features/5) + 1

  feature_str = '\n'
  for g in range(groups):
    group_start = g * 5
    group_end = (g + 1) * 5
    if (group_end > num_features):
      group_end = num_features

    group_indices = indices[group_start:group_end]
    for idx in group_indices:
      if (idx == group_indices[-1]):
        feature_str += str(idx+1) + ': ' + features[idx]
      else:
        feature_str += str(idx+1) + ': ' + features[idx] + ', '

    feature_str += '\n'

  print(feature_str)
  if (log_fh is not None):
    log_fh.write(feature_str)

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

def getFilename(crop, country, yield_trend,
                early_season, early_season_end, nuts_level=None):
  """Get filename based on input arguments"""
  suffix = crop.replace(' ', '_')
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

def getLogFilename(crop, country, yield_trend,
                   early_season, early_season_end):
  """Get filename for experiment log"""
  log_file = getFilename(crop, country, yield_trend,
                         early_season, early_season_end)
  return log_file + '.log'

def getFeatureFilename(crop, country, yield_trend,
                       early_season, early_season_end):
  """Get unique filename for features"""
  feature_file = 'ft_'
  suffix = getFilename(crop, country, yield_trend, early_season, early_season_end)
  feature_file += suffix
  return feature_file

def getPredictionFilename(crop, country, nuts_level, yield_trend,
                          early_season, early_season_end):
  """Get unique filename for predictions"""
  pred_file = 'pred_'
  suffix = getFilename(crop, country, yield_trend,
                       early_season, early_season_end, nuts_level)
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
