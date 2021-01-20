import sys
import argparse

import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from cypml.common import globals

if (globals.test_env == 'pkg'):
  from cypml.common.config import CYPConfiguration
  from cypml.common.util import getLogFilename
  from cypml.workflow.data_loading import CYPDataLoader
  from cypml.workflow.data_preprocessing import CYPDataPreprocessor
  from cypml.workflow.data_summary import CYPDataSummarizer
  from cypml.workflow.feature_design import CYPFeaturizer
  from cypml.workflow.yield_trend import CYPYieldTrendEstimator

  from cypml.run_workflow.run_data_preprocessing import preprocessData
  from cypml.run_workflow.run_data_summary import summarizeData
  from cypml.run_workflow.run_feature_design import createFeatures
  from cypml.run_workflow.run_trend_feature_design import createYieldTrendFeatures
  from cypml.run_workflow.run_train_test_split import splitDataIntoTrainingTestSets
  from cypml.run_workflow.combine_features import combineFeaturesLabels
  from cypml.run_workflow.load_saved_features import loadSavedFeaturesLabels
  from cypml.run_workflow.run_machine_learning import getMachineLearningPredictions
  from cypml.run_workflow.run_machine_learning import saveMLPredictions
  from cypml.run_workflow.load_saved_predictions import loadSavedPredictions
  from cypml.run_workflow.compare_with_mcyfs import comparePredictionsWithMCYFS

  from cypml.tests.test_util import TestUtil
  from cypml.tests.test_data_loading import TestDataLoader 
  from cypml.tests.test_data_preprocessing import TestDataPreprocessor 
  from cypml.tests.test_data_summary import TestDataSummarizer 
  from cypml.tests.test_yield_trend import TestYieldTrendEstimator 

def main():
  if (globals.test_env == 'pkg'):
    test_env = globals.test_env
    run_tests = globals.run_tests


  SparkContext.setSystemProperty('spark.executor.memory', '12g')
  SparkContext.setSystemProperty('spark.driver.memory', '6g')
  spark = SparkSession.builder.master("local[*]").getOrCreate()
  spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

  sc = SparkContext.getOrCreate()
  sqlContext = SQLContext(sc)

  print('##################')
  print('# Configuration  #')
  print('##################')

  parser = argparse.ArgumentParser(prog='mlbaseline_pkg.py')

  # Some command-line argument names are slightly different
  # from configuration option names for brevity.
  args_dict = {
      '--crop' : { 'type' : str,
                   'default' : 'potatoes',
                   'help' : 'crop name (default: potatoes)',
                 },
      '--crosses-calendar-year' : { 'type' : str,
                                    'default' : 'N',
                                    'choices' : ['Y', 'N'],
                                    'help' : 'crop growing season crosses calendar year boundary (default: N)',
                                  },
      '--country' : { 'type' : str,
                      'default' : 'NL',
                      'choices' : ['NL', 'DE', 'FR'],
                      'help' : 'country code (default: NL)',
                    },
      '--nuts-level' : { 'type' : str,
                         'default' : 'NUTS2',
                         'choices' : ['NUTS2', 'NUTS3'],
                         'help' : 'country code (default: NL)',
                       },
      '--data-path' : { 'type' : str,
                        'default' : '.',
                        'help' : 'path to data files (default: .)',
                       },
      '--output-path' : { 'type' : str,
                          'default' : '.',
                          'help' : 'path to output files (default: .)',
                        },
      '--yield-trend' : { 'type' : str,
                          'default' : 'N',
                          'choices' : ['Y', 'N'],
                          'help' : 'estimate and use yield trend (default: N)',
                        },
      '--optimal-trend-window' : { 'type' : str,
                                   'default' : 'N',
                                   'choices' : ['Y', 'N'],
                                   'help' : 'find optimal trend window for each year (default: N)',
                                 },
      '--predict-residuals' : { 'type' : str,
                                'default' : 'N',
                                'choices' : ['Y', 'N'],
                                'help' : 'predict yield residuals instead of full yield (default: N)',
                              },
      '--early-season' : { 'type' : str,
                           'default' : 'N',
                           'choices' : ['Y', 'N'],
                           'help' : 'early season prediction (default: N)',
                         },
      '--early-season-end' : { 'type' : int,
                               'default' : 15,
                               'help' : 'early season end dekad (default: 15)',
                             },
      '--centroids' : { 'type' : str,
                        'default' : 'N',
                        'choices' : ['Y', 'N'],
                        'help' : 'use centroid coordinates and distance to coast (default: N)',
                      },
      '--remote-sensing' : { 'type' : str,
                             'default' : 'Y',
                             'choices' : ['Y', 'N'],
                             'help' : 'use remote sensing data (default: Y)',
                           },
      '--save-features' : { 'type' : str,
                            'default' : 'N',
                            'choices' : ['Y', 'N'],
                            'help' : 'save features to a CSV file (default: N)',
                          },
      '--use-saved-features' : { 'type' : str,
                                 'default' : 'N',
                                 'choices' : ['Y', 'N'],
                                 'help' : 'use features from a CSV file (default: N). Set ',
                               },
      '--save-predictions' : { 'type' : str,
                               'default' : 'Y',
                               'choices' : ['Y', 'N'],
                               'help' : 'save predictions to a CSV file (default: Y)',
                             },
      '--use-saved-predictions' : { 'type' : str,
                                    'default' : 'N',
                                    'choices' : ['Y', 'N'],
                                    'help' : 'use predictions from a CSV file (default: N)',
                                  },
      '--compare-with-mcyfs' : { 'type' : str,
                                 'default' : 'N',
                                 'choices' : ['Y', 'N'],
                                 'help' : 'compare predictions with MCYFS (default: N)',
                               },
      '--debug-level' : { 'type' : int,
                          'default' : 0,
                          'choices' : range(4),
                          'help' : 'amount of debug information to print (default: 0)',
                        },
  }

  for arg in args_dict:
    arg_config = args_dict[arg]
    # add cases if other argument settings are used
    if ('choices' in arg_config):
      parser.add_argument(arg, type=arg_config['type'], default=arg_config['default'],
                          choices=arg_config['choices'], help=arg_config['help'])
    else:
      parser.add_argument(arg, type=arg_config['type'], default=arg_config['default'],
                          help=arg_config['help'])

  if (run_tests):
    test_util = TestUtil(spark)
    test_util.runAllTests()

  args = parser.parse_args()
  cyp_config = CYPConfiguration()

  # must be in sync with args_dict used to parse args
  config_update = {
      'crop_name' : args.crop,
      'season_crosses_calendar_year' : args.crosses_calendar_year,
      'country_code' : args.country,
      'nuts_level' : args.nuts_level,
      'data_path' : args.data_path,
      'output_path' : args.output_path,
      'use_yield_trend' : args.yield_trend,
      'find_optimal_trend_window' : args.optimal_trend_window,
      'predict_yield_residuals' : args.predict_residuals,
      'early_season_prediction' : args.early_season,
      'early_season_end_dekad' : args.early_season_end,
      'use_centroids' : args.centroids,
      'use_remote_sensing' : args.remote_sensing,
      'save_features' : args.save_features,
      'use_saved_features' : args.use_saved_features,
      'save_predictions' : args.save_predictions,
      'use_saved_predictions' : args.use_saved_predictions,
      'compare_with_mcyfs' : args.compare_with_mcyfs,
      'debug_level' : args.debug_level,
  }

  cyp_config.updateConfiguration(config_update)
  crop = cyp_config.getCropName()
  country = cyp_config.getCountryCode()
  nuts_level = cyp_config.getNUTSLevel()
  debug_level = cyp_config.getDebugLevel()
  use_saved_predictions = cyp_config.useSavedPredictions()
  use_saved_features = cyp_config.useSavedFeatures()
  use_yield_trend = cyp_config.useYieldTrend()
  early_season_prediction = cyp_config.earlySeasonPrediction()
  early_season_end = cyp_config.getEarlySeasonEndDekad()

  output_path = cyp_config.getOutputPath()
  log_file = getLogFilename(crop, country, use_yield_trend,
                            early_season_prediction, early_season_end)
  log_fh = open(output_path + '/' + log_file, 'w+')
  cyp_config.printConfig(log_fh)

  if (not use_saved_predictions):
    if (not use_saved_features):
      print('#################')
      print('# Data Loading  #')
      print('#################')

      if (run_tests):
        test_loader = TestDataLoader(spark)
        test_loader.runAllTests()

      cyp_loader = CYPDataLoader(spark, cyp_config)
      data_dfs = cyp_loader.loadAllData()

      print('#######################')
      print('# Data Preprocessing  #')
      print('#######################')

      if (run_tests):
        test_preprocessor = TestDataPreprocessor(spark)
        test_preprocessor.runAllTests()

      cyp_preprocessor = CYPDataPreprocessor(spark, cyp_config)
      data_dfs = preprocessData(cyp_config, cyp_preprocessor, data_dfs)

      print('###########################')
      print('# Training and Test Split #')
      print('###########################')

      if (run_tests):
        yield_df = data_dfs['YIELD']
        test_custom = TestCustomTrainTestSplit(yield_df)
        test_custom.runAllTests()

      prep_train_test_dfs, test_years = splitDataIntoTrainingTestSets(cyp_config, data_dfs, log_fh)

      print('#################')
      print('# Data Summary  #')
      print('#################')

      if (run_tests):
        test_summarizer = TestDataSummarizer(spark)
        test_summarizer.runAllTests()

      cyp_summarizer = CYPDataSummarizer(cyp_config)
      summary_dfs = summarizeData(cyp_config, cyp_summarizer, prep_train_test_dfs)

      print('###################')
      print('# Feature Design  #')
      print('###################')

      # WOFOST, Meteo and Remote Sensing features
      cyp_featurizer = CYPFeaturizer(cyp_config)
      pd_feature_dfs = createFeatures(cyp_config, cyp_featurizer,
                                      prep_train_test_dfs, summary_dfs, log_fh)

      # yield trend features
      if (use_yield_trend):
        yield_train_df = prep_train_test_dfs['YIELD'][0]
        yield_test_df = prep_train_test_dfs['YIELD'][1]

        if (run_tests):
          test_yield_trend = TestYieldTrendEstimator(yield_train_df)
          test_yield_trend.runAllTests()

        cyp_trend_est = CYPYieldTrendEstimator(cyp_config)
        pd_yield_train_ft, pd_yield_test_ft = createYieldTrendFeatures(cyp_config, cyp_trend_est,
                                                                       yield_train_df, yield_test_df,
                                                                       test_years)
        pd_feature_dfs['YIELD_TREND'] = [pd_yield_train_ft, pd_yield_test_ft]

      # combine features
      join_cols = ['IDREGION', 'FYEAR']
      pd_train_df, pd_test_df = combineFeaturesLabels(cyp_config, sqlContext,
                                                      prep_train_test_dfs, pd_feature_dfs,
                                                      join_cols, log_fh)
    # use saved features
    else:
      pd_train_df, pd_test_df = loadSavedFeaturesLabels(cyp_config, spark)

    print('###################################')
    print('# Machine Learning using sklearn  #')
    print('###################################')

    pd_ml_predictions = getMachineLearningPredictions(cyp_config, pd_train_df, pd_test_df, log_fh)
    save_predictions = cyp_config.savePredictions()
    if (save_predictions):
      saveMLPredictions(cyp_config, sqlContext, pd_ml_predictions)

  # use saved predictions
  else:
    pd_ml_predictions = loadSavedPredictions(cyp_config, spark)

  # compare with MCYFS
  compareWithMCYFS = cyp_config.compareWithMCYFS()
  if (compareWithMCYFS):
    comparePredictionsWithMCYFS(sqlContext, cyp_config, pd_ml_predictions, log_fh)

  log_fh.close()

if __name__ == '__main__':
    main()
