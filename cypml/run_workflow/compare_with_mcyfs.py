import numpy as np
import pandas as pd

from ..common import globals

if (globals.test_env == 'pkg'):
  SparkT = globals.SparkT
  SparkF = globals.SparkF
  run_tests = globals.run_tests

  from ..common.util import getPredictionScores
  from ..common.util import getPredictionFilename
  from ..workflow.data_loading import CYPDataLoader
  from ..workflow.data_preprocessing import CYPDataPreprocessor
  from ..workflow.data_summary import CYPDataSummarizer

  from ..tests.test_util import TestUtil
  from ..tests.test_data_loading import TestDataLoader 
  from ..tests.test_data_preprocessing import TestDataPreprocessor 
  from ..tests.test_data_summary import TestDataSummarizer 

def saveNUTS0Predictions(cyp_config, sqlCtx, nuts0_ml_predictions):
  """Save predictions aggregated to NUTS0"""
  crop = cyp_config.getCropName()
  country = cyp_config.getCountryCode()
  nuts_level = 'NUTS0'
  use_yield_trend = cyp_config.useYieldTrend()
  early_season_prediction = cyp_config.earlySeasonPrediction()
  early_season_end = cyp_config.getEarlySeasonEndDekad()
  debug_level = cyp_config.getDebugLevel()

  output_path = cyp_config.getOutputPath()
  output_file = getPredictionFilename(crop, country, nuts_level, use_yield_trend,
                                      early_season_prediction, early_season_end)

  save_pred_path = output_path + '/' + output_file
  if (debug_level > 1):
    print('\nNUTS0 Predictions of ML algorithms')
    print(nuts0_ml_predictions.head(5))
    print('\nSaving predictions to', save_pred_path + '.csv')

  nuts0_ml_predictions.to_csv(save_pred_path + '.csv', index=False, header=True)

  # NOTE: In some environments, Spark can write, but pandas cannot.
  # In such cases, use the following code.
  # spark_predictions_df = sqlCtx.createDataFrame(nuts0_ml_predictions)
  # spark_predictions_df.coalesce(1)\
  #                     .write.option('header','true')\
  #                     .mode("overwrite").csv(save_pred_path)

def getDataForMCYFSComparison(spark, cyp_config, test_years):
  """Load and preprocess data for MCYFS comparison"""
  data_path = cyp_config.getDataPath()
  crop_id = cyp_config.getCropID()
  nuts_level = cyp_config.getNUTSLevel()
  season_crosses_calyear = cyp_config.seasonCrossesCalendarYear()
  early_season_end = cyp_config.getEarlySeasonEndDekad()
  debug_level = cyp_config.getDebugLevel()
  area_nuts = ['NUTS' + str(i) for i in range(int(nuts_level[-1]), 0, -1)]
  data_sources = {
      'WOFOST' : nuts_level,
      'AREA_FRACTIONS' : area_nuts,
      'YIELD' : 'NUTS0',
      'YIELD_PRED_MCYFS' : 'NUTS0',
  }

  if (run_tests):
    test_util = TestUtil(spark)
    test_util.runAllTests()

  print('##############')
  print('# Load Data  #')
  print('##############')

  if (run_tests):
    test_loader = TestDataLoader(spark)
    test_loader.runAllTests()

  cyp_config.setDataSources(data_sources)
  cyp_loader = CYPDataLoader(spark, cyp_config)
  data_dfs = cyp_loader.loadAllData()

  wofost_df = data_dfs['WOFOST']
  area_dfs = data_dfs['AREA_FRACTIONS']
  nuts0_yield_df = data_dfs['YIELD']
  mcyfs_yield_df = data_dfs['YIELD_PRED_MCYFS']

  print('####################')
  print('# Preprocess Data  #')
  print('####################')

  if (run_tests):
    test_preprocessor = TestDataPreprocessor(spark)
    test_preprocessor.runAllTests()

  cyp_preprocessor = CYPDataPreprocessor(spark, cyp_config)
  wofost_df = wofost_df.filter(wofost_df['CROP_ID'] == crop_id).drop('CROP_ID')
  crop_season = cyp_preprocessor.getCropSeasonInformation(wofost_df, season_crosses_calyear)
  wofost_df = cyp_preprocessor.preprocessWofost(wofost_df, crop_season, season_crosses_calyear)

  for i in range(len(area_dfs)):
    af_df = area_dfs[i]
    af_df = cyp_preprocessor.preprocessAreaFractions(af_df, crop_id)
    af_df = af_df.filter(af_df['FYEAR'].isin(test_years))
    area_dfs[i] = af_df

  if (debug_level > 1):
    print('NUTS0 Yield before preprocessing')
    nuts0_yield_df.show(10)

  nuts0_yield_df = cyp_preprocessor.preprocessYield(nuts0_yield_df, crop_id)
  nuts0_yield_df = nuts0_yield_df.filter(nuts0_yield_df['FYEAR'].isin(test_years))
  if (debug_level > 1):
    print('NUTS0 Yield after preprocessing')
    nuts0_yield_df.show(10)

  if (debug_level > 1):
    print('MCYFS yield predictions before preprocessing')
    mcyfs_yield_df.show(10)

  mcyfs_yield_df = cyp_preprocessor.preprocessYieldMCYFS(mcyfs_yield_df, crop_id)
  mcyfs_yield_df = mcyfs_yield_df.filter(mcyfs_yield_df['FYEAR'].isin(test_years))
  if (debug_level > 1):
    print('MCYFS yield predictions after preprocessing')
    mcyfs_yield_df.show(10)

  # Check we have yield data for crop
  assert (nuts0_yield_df is not None)
  assert (mcyfs_yield_df is not None)

  if (run_tests):
    test_summarizer = TestDataSummarizer(spark)
    test_summarizer.runAllTests()

  cyp_summarizer = CYPDataSummarizer(cyp_config)
  dvs_summary = cyp_summarizer.wofostDVSSummary(wofost_df, early_season_end)
  dvs_summary = dvs_summary.filter(dvs_summary['CAMPAIGN_YEAR'].isin(test_years))

  data_dfs = {
      'WOFOST_DVS' : dvs_summary,
      'AREA_FRACTIONS' : area_dfs,
      'YIELD_NUTS0' : nuts0_yield_df,
      'YIELD_PRED_MCYFS' : mcyfs_yield_df
  }

  return data_dfs

def fillMissingDataWithAverage(pd_pred_df, print_debug):
  """Fill missing data with regional average or zero"""
  regions = pd_pred_df['IDREGION'].unique()

  for reg_id in regions:
    reg_filter = (pd_pred_df['IDREGION'] == reg_id)
    pd_reg_pred_df = pd_pred_df[reg_filter]

    if (len(pd_reg_pred_df[pd_reg_pred_df['YIELD_PRED'].notnull()].index) == 0):
      if (print_debug):
        print('No data for', reg_id)

      pd_pred_df.loc[reg_filter, 'FRACTION'] = 0.0
      pd_pred_df.loc[reg_filter, 'YIELD_PRED'] = 0.0
    else:
      reg_avg_yield_pred = pd_pred_df.loc[reg_filter, 'YIELD_PRED'].mean()
      pd_pred_df.loc[reg_filter, 'YIELD_PRED'] = pd_pred_df.loc[reg_filter, 'YIELD_PRED']\
                                                           .fillna(reg_avg_yield_pred)  

  return pd_pred_df

def recalculateAreaFractions(pd_pred_df, print_debug):
  """Recalculate area fractions by excluding regions with missing data"""
  join_cols = ['IDREG_PARENT', 'FYEAR']
  pd_af_sum = pd_pred_df.groupby(join_cols).agg(FRACTION_SUM=('FRACTION', 'sum')).reset_index()
  pd_pred_df = pd_pred_df.merge(pd_af_sum, on=join_cols, how='left')
  pd_pred_df['FRACTION'] = pd_pred_df['FRACTION'] / pd_pred_df['FRACTION_SUM']
  pd_pred_df = pd_pred_df.drop(columns=['FRACTION_SUM'])

  return pd_pred_df

def aggregatePredictionsToNUTS0(cyp_config, pd_ml_predictions,
                                area_dfs, test_years, join_cols):
  """Aggregate regional predictions to national level"""
  pd_area_dfs = []
  nuts_level = cyp_config.getNUTSLevel()
  use_yield_trend = cyp_config.useYieldTrend()
  crop_id = cyp_config.getCropID()
  alg_names = list(cyp_config.getEstimators().keys())
  debug_level = cyp_config.getDebugLevel()

  for af_df in area_dfs:
    pd_af_df = af_df.toPandas()
    pd_af_df = pd_af_df[pd_af_df['FYEAR'].isin(test_years)]
    pd_area_dfs.append(pd_af_df)

  nuts0_pred_df = None
  for alg in alg_names:
    sel_cols = ['IDREGION', 'FYEAR', 'YIELD_PRED_' + alg]
    pd_alg_pred_df = pd_ml_predictions[sel_cols]
    pd_alg_pred_df = pd_alg_pred_df.rename(columns={'YIELD_PRED_' + alg : 'YIELD_PRED'})

    for idx in range(len(pd_area_dfs)):
      pd_af_df = pd_area_dfs[idx]
      # merge with area fractions to get all regions and years
      pd_alg_pred_df = pd_af_df.merge(pd_alg_pred_df, on=join_cols)
      print_debug = (debug_level > 2) and (alg == alg_names[0])
      pd_alg_pred_df = fillMissingDataWithAverage(pd_alg_pred_df, print_debug)
      pd_alg_pred_df['IDREG_PARENT'] = pd_alg_pred_df['IDREGION'].str[:-1]
      pd_alg_pred_df = recalculateAreaFractions(pd_alg_pred_df, print_debug)
      if (print_debug):
        print('\nAggregation to NUTS' + str(len(pd_area_dfs) - (idx + 1)))
        print(pd_alg_pred_df[pd_alg_pred_df['FYEAR'] == test_years[0]].head(10))

      pd_alg_pred_df['YPRED_WEIGHTED'] = pd_alg_pred_df['YIELD_PRED'] * pd_alg_pred_df['FRACTION']
      pd_alg_pred_df = pd_alg_pred_df.groupby(by=['IDREG_PARENT', 'FYEAR'])\
                                     .agg(YPRED_WEIGHTED=('YPRED_WEIGHTED', 'sum')).reset_index()
      pd_alg_pred_df = pd_alg_pred_df.rename(columns={'IDREG_PARENT': 'IDREGION',
                                                      'YPRED_WEIGHTED': 'YIELD_PRED' })

    pd_alg_pred_df = pd_alg_pred_df.rename(columns={ 'YIELD_PRED': 'YIELD_PRED_' + alg })
    if (nuts0_pred_df is None):
      nuts0_pred_df = pd_alg_pred_df
    else:
      nuts0_pred_df = nuts0_pred_df.merge(pd_alg_pred_df, on=join_cols)

  return nuts0_pred_df

def getMCYFSPrediction(pd_mcyfs_pred_df, pred_year, pred_dekad, print_debug):
  """Get MCYFS prediction for given year with prediction date close to pred_dekad"""
  pd_pred_year = pd_mcyfs_pred_df[pd_mcyfs_pred_df['FYEAR'] == pred_year]
  mcyfs_pred_dekads = pd_pred_year['PRED_DEKAD'].unique()
  if (len(mcyfs_pred_dekads) == 0):
    return 0.0

  mcyfs_pred_dekads = sorted(mcyfs_pred_dekads)
  mcyfs_pred_dekad = mcyfs_pred_dekads[-1]
  if (pred_dekad < mcyfs_pred_dekad):
    for dek in mcyfs_pred_dekads:
      if dek >= pred_dekad:
        mcyfs_pred_dekad = dek
        break

  pd_pred_dek = pd_pred_year[pd_pred_year['PRED_DEKAD'] == mcyfs_pred_dekad]
  yield_pred_list = pd_pred_dek['YIELD_PRED'].values

  if (print_debug):
    print('\nAll MCYFS dekads for', pred_year, ':', mcyfs_pred_dekads)
    print('MCYFS prediction dekad', mcyfs_pred_dekad)
    print('ML Baseline prediction dekad', pred_dekad)
    print('MCYFS prediction:', yield_pred_list[0], '\n')

  return yield_pred_list[0]

def getNUTS0Yield(pd_nuts0_yield_df, pred_year, print_debug):
  """Get the true (reported) Eurostat yield value"""
  nuts0_yield_year = pd_nuts0_yield_df[pd_nuts0_yield_df['FYEAR'] == pred_year]
  pred_year_yield = nuts0_yield_year['YIELD'].values
  if (len(pred_year_yield) == 0):
    return 0.0

  if (print_debug):
    print(pred_year, 'Eurostat yield', pred_year_yield[0])

  return pred_year_yield[0]

def comparePredictionsWithMCYFS(sqlCtx, cyp_config, pd_ml_predictions, log_fh):
  """Compare ML Baseline predictions with MCYFS predictions"""
  # We need AREA_FRACTIONS, MCYFS yield predictions and NUTS0 Eurostat YIELD
  # for comparison with MCYFS
  country_code = cyp_config.getCountryCode()
  debug_level = cyp_config.getDebugLevel()
  alg_names = list(cyp_config.getEstimators().keys())
  test_years = list(pd_ml_predictions['FYEAR'].unique())
  early_season_prediction = cyp_config.earlySeasonPrediction()

  spark = sqlCtx.sparkSession
  data_dfs = getDataForMCYFSComparison(spark, cyp_config, test_years)
  pd_dvs_summary = data_dfs['WOFOST_DVS'].toPandas()
  pd_nuts0_yield_df = data_dfs['YIELD_NUTS0'].toPandas()
  pd_mcyfs_pred_df = data_dfs['YIELD_PRED_MCYFS'].toPandas()
  area_dfs = data_dfs['AREA_FRACTIONS']
  join_cols = ['IDREGION', 'FYEAR']
  test_years = pd_ml_predictions['FYEAR'].unique()
  metrics = cyp_config.getEvaluationMetrics()
  nuts0_pred_df = aggregatePredictionsToNUTS0(cyp_config, pd_ml_predictions,
                                              area_dfs, test_years, join_cols)

  crop_season_cols = ['IDREGION', 'CAMPAIGN_YEAR', 'CALENDAR_END_SEASON', 'CALENDAR_EARLY_SEASON']
  pd_dvs_summary = pd_dvs_summary[crop_season_cols].rename(columns={ 'CAMPAIGN_YEAR' : 'FYEAR' })
  pd_dvs_summary = pd_dvs_summary.groupby('FYEAR').agg(END_SEASON=('CALENDAR_END_SEASON', 'mean'),
                                                       EARLY_SEASON=('CALENDAR_EARLY_SEASON', 'mean'))\
                                                       .round(0).reset_index()
  if (debug_level > 1):
    print(pd_dvs_summary.head(5).to_string(index=False))

  alg_summary = {}
  Y_pred_mcyfs = []
  Y_true = []
  nuts0_pred_df['YIELD_PRED_MCYFS'] = 0.0
  nuts0_pred_df['YIELD'] = 0.0
  nuts0_pred_df = nuts0_pred_df.sort_values(by=join_cols)
  ml_pred_years = nuts0_pred_df['FYEAR'].unique()
  mcyfs_pred_years = []
  print_debug = (debug_level > 2)
  if (print_debug):
    print('\nPredictions and true values for', country_code)

  for yr in ml_pred_years:
    pred_dekad = pd_dvs_summary[pd_dvs_summary['FYEAR'] == yr]['END_SEASON'].values[0]
    if (early_season_prediction):
      pred_dekad = pd_dvs_summary[pd_dvs_summary['FYEAR'] == yr]['EARLY_SEASON'].values[0]

    mcyfs_pred = getMCYFSPrediction(pd_mcyfs_pred_df, yr, pred_dekad, print_debug)
    nuts0_yield = getNUTS0Yield(pd_nuts0_yield_df, yr, print_debug)
    if ((mcyfs_pred > 0.0) and (nuts0_yield > 0.0)):
      nuts0_pred_df.loc[nuts0_pred_df['FYEAR'] == yr, 'YIELD'] = nuts0_yield
      nuts0_pred_df.loc[nuts0_pred_df['FYEAR'] == yr, 'YIELD_PRED_MCYFS'] = mcyfs_pred
      mcyfs_pred_years.append(yr)

  nuts0_pred_df = nuts0_pred_df[nuts0_pred_df['FYEAR'].isin(mcyfs_pred_years)]
  Y_true = nuts0_pred_df['YIELD'].values

  if (print_debug):
    print(nuts0_pred_df.head(5))

  if (len(mcyfs_pred_years) > 0):
    for alg in alg_names:
      Y_pred_alg = nuts0_pred_df['YIELD_PRED_' + alg].values
      alg_nuts0_scores = getPredictionScores(Y_true, Y_pred_alg, metrics)

      alg_row = [alg]
      for met in alg_nuts0_scores:
        alg_row.append(alg_nuts0_scores[met])

      alg_index = len(alg_summary)
      alg_summary['row' + str(alg_index)] = alg_row

    Y_pred_mcyfs = nuts0_pred_df['YIELD_PRED_MCYFS'].values
    mcyfs_nuts0_scores = getPredictionScores(Y_true, Y_pred_mcyfs, metrics)
    alg_row = ['MCYFS_Predictions']
    for met in mcyfs_nuts0_scores:
      alg_row.append(mcyfs_nuts0_scores[met])

    alg_index = len(alg_summary)
    alg_summary['row' + str(alg_index)] = alg_row

    alg_df_columns = ['algorithm']
    for met in metrics:
      alg_df_columns += ['test_' + met]

    alg_df = pd.DataFrame.from_dict(alg_summary, orient='index',
                                    columns=alg_df_columns)
    eval_summary_info = '\nAlgorithm Evaluation Summary (NUTS0) for ' + country_code
    eval_summary_info += '\n-------------------------------------------'
    eval_summary_info += '\n' + alg_df.to_string(index=False) + '\n'
    log_fh.write(eval_summary_info)
    print(eval_summary_info)

  save_predictions = cyp_config.savePredictions()
  if (save_predictions):
    saveNUTS0Predictions(cyp_config, sqlCtx, nuts0_pred_df)
