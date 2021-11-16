import pandas as pd

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import getFeatureFilename

def loadSavedFeaturesLabels(cyp_config, spark):
  """Load saved features from a CSV file"""
  crop = cyp_config.getCropName()
  country = cyp_config.getCountryCode()
  use_yield_trend = cyp_config.useYieldTrend()
  early_season_prediction = cyp_config.earlySeasonPrediction()
  early_season_end = cyp_config.getEarlySeasonEndDekad()
  debug_level = cyp_config.getDebugLevel()

  feature_file_path = cyp_config.getOutputPath()
  feature_file = getFeatureFilename(crop, use_yield_trend,
                                    early_season_prediction, early_season_end,
                                    country)

  load_ft_path = feature_file_path + '/' + feature_file
  pd_train_df = pd.read_csv(load_ft_path + '_train.csv', header=0)
  pd_test_df = pd.read_csv(load_ft_path + '_test.csv', header=0)

  # NOTE: In some environments, Spark can read, but pandas cannot.
  # In such cases, use the following code.
  # spark_train_df = spark.read.csv(load_ft_path + '_train.csv', header=True, inferSchema=True)
  # spark_test_df = spark.read.csv(load_ft_path + '_test.csv', header=True, inferSchema=True)
  # pd_train_df = spark_train_df.toPandas()
  # pd_test_df = spark_test_df.toPandas()

  if (debug_level > 1):
    print('\nAll Features and labels')
    print(pd_train_df.head(5))
    print(pd_test_df.head(5))

  return pd_train_df, pd_test_df
