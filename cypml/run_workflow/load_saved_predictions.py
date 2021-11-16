import pandas as pd

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import getPredictionFilename

def loadSavedPredictions(cyp_config, spark):
  """Load machine learning predictions from saved CSV file"""
  crop = cyp_config.getCropName()
  country = cyp_config.getCountryCode()
  nuts_level = cyp_config.getNUTSLevel()
  use_yield_trend = cyp_config.useYieldTrend()
  early_season_prediction = cyp_config.earlySeasonPrediction()
  early_season_end = cyp_config.getEarlySeasonEndDekad()
  debug_level = cyp_config.getDebugLevel()

  pred_file_path = cyp_config.getOutputPath()
  pred_file = getPredictionFilename(crop, use_yield_trend,
                                    early_season_prediction, early_season_end,
                                    country, nuts_level)
  pred_file += '.csv'
  pd_ml_predictions = pd.read_csv(pred_file_path + '/' + pred_file, header=0)

  # NOTE: In some environments, Spark can read, but pandas cannot.
  # In such cases, use the following code.
  # all_pred_df = spark.read.csv(pred_file_path + '/' + pred_file, header=True, inferSchema=True)
  # pd_ml_predictions = all_pred_df.toPandas()

  if (debug_level > 1):
    print(pd_ml_predictions.head(5))

  return pd_ml_predictions
