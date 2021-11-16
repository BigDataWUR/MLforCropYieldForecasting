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
  output_file = getPredictionFilename(crop, use_yield_trend,
                                      early_season_prediction, early_season_end,
                                      country, nuts_level)

  save_pred_path = output_path + '/' + output_file
  if (debug_level > 1):
    print('\nNUTS0 Predictions of ML algorithms')
    print(nuts0_ml_predictions.head(5).to_string(index=False))
    print('\nSaving predictions to', save_pred_path + '.csv')

  nuts0_ml_predictions.to_csv(save_pred_path + '.csv', index=False, header=True)

  # NOTE: In some environments, Spark can write, but pandas cannot.
  # In such cases, use the following code.
  # spark_predictions_df = sqlCtx.createDataFrame(nuts0_ml_predictions)
  # spark_predictions_df.coalesce(1)\
  #                     .write.option('header','true')\
  #                     .mode("overwrite").csv(save_pred_path)

def getDataForMCYFSComparison(spark, cyp_config, pd_ml_predictions, test_years):
  """Load and preprocess data for MCYFS comparison"""
  data_path = cyp_config.getDataPath()
  crop_id = cyp_config.getCropID()
  nuts_level = cyp_config.getNUTSLevel()
  debug_level = cyp_config.getDebugLevel()
  country_code = cyp_config.getCountryCode()
  season_crosses_calyear = cyp_config.seasonCrossesCalendarYear()
  early_season_end = cyp_config.getEarlySeasonEndDekad()

  if (country_code is None):
    countries = pd_ml_predictions['COUNTRY'].unique()
    country_nuts = {}
    for c in countries:
      pd_ml_country_preds = pd_ml_predictions[pd_ml_predictions['IDREGION'].str[:2] == c]
      first_reg = pd_ml_country_preds['IDREGION'].iloc[0]
      nuts = 'NUTS' + str(len(first_reg[2:]))
      country_nuts[c] = nuts
  else:
    country_nuts = { country_code : nuts_level }

  if (run_tests):
    test_util = TestUtil(spark)
    test_util.runAllTests()

  print('\n##############')
  print('# Load Data  #')
  print('##############')

  if (run_tests):
    test_loader = TestDataLoader(spark)
    test_loader.runAllTests()

  data_sources = ['WOFOST', 'AREA_FRACTIONS', 'YIELD', 'YIELD_PRED_MCYFS']
  cyp_config.setDataSources(data_sources)
  cyp_loader = CYPDataLoader(spark, cyp_config)

  lowest_nuts = 0
  for c in country_nuts:
    nuts_level_int = int(country_nuts[c][-1])
    if (nuts_level_int > lowest_nuts):
      lowest_nuts = nuts_level_int

  area_nuts = ['NUTS' + str(i) for i in range(lowest_nuts, 0, -1)]
  area_fraction_dfs = cyp_loader.loadData('AREA_FRACTIONS', area_nuts)
  wofost_df = cyp_loader.loadData('WOFOST', nuts_level)
  wofost_df = wofost_df.filter(wofost_df['CROP_ID'] == crop_id).drop('CROP_ID')
  nuts0_yield_df = cyp_loader.loadData('YIELD', 'NUTS0')
  mcyfs_yield_df = cyp_loader.loadData('YIELD_PRED_MCYFS', 'NUTS0')

  print('\n####################')
  print('# Preprocess Data  #')
  print('####################')

  if (run_tests):
    test_preprocessor = TestDataPreprocessor(spark)
    test_preprocessor.runAllTests()

  if (run_tests):
    test_summarizer = TestDataSummarizer(spark)
    test_summarizer.runAllTests()

  cyp_preprocessor = CYPDataPreprocessor(spark, cyp_config)
  cyp_summarizer = CYPDataSummarizer(cyp_config)

  country_area_dfs = {}
  country_dvs_summaries = {}
  country_nuts0_yields = {}
  country_mcyfs_yields = {}
  for c in country_nuts:
    nuts_level_int = int(country_nuts[c][-1])
    sel_af_dfs = area_fraction_dfs[-nuts_level_int:]
    for i in range(len(sel_af_dfs)):
      nuts_af_df = cyp_preprocessor.preprocessAreaFractions(sel_af_dfs[i], crop_id)
      nuts_af_df = nuts_af_df.filter(SparkF.substring(nuts_af_df['IDREGION'], 1, 2) == c)
      nuts_af_df = nuts_af_df.filter(nuts_af_df['FYEAR'].isin(test_years))
      sel_af_dfs[i] = nuts_af_df

    country_area_dfs[c] = sel_af_dfs

    # DVS summary with crop season information
    sel_wofost_df = wofost_df.filter(SparkF.substring(wofost_df['IDREGION'], 1, 2) == c)
    crop_season = cyp_preprocessor.getCropSeasonInformation(sel_wofost_df, season_crosses_calyear)
    sel_wofost_df = cyp_preprocessor.preprocessWofost(sel_wofost_df, crop_season, season_crosses_calyear)
    dvs_summary = cyp_summarizer.wofostDVSSummary(sel_wofost_df, early_season_end)
    dvs_summary = dvs_summary.filter(dvs_summary['CAMPAIGN_YEAR'].isin(test_years))
    country_dvs_summaries[c] = dvs_summary

    # NUTS0 Yield and MCYFS predictions
    sel_nuts0_yield_df = nuts0_yield_df.filter(SparkF.substring(nuts0_yield_df['IDREGION'], 1, 2) == c)
    if (debug_level > 1):
      print('NUTS0 Yield for', c, 'before preprocessing')
      sel_nuts0_yield_df.show(10)

    sel_nuts0_yield_df = cyp_preprocessor.preprocessYield(sel_nuts0_yield_df, crop_id)
    sel_nuts0_yield_df = sel_nuts0_yield_df.filter(sel_nuts0_yield_df['FYEAR'].isin(test_years))
    if (debug_level > 1):
      print('NUTS0 Yield for', c, 'after preprocessing')
      sel_nuts0_yield_df.show(10)

    sel_mcyfs_yield_df = mcyfs_yield_df.filter(SparkF.substring(mcyfs_yield_df['IDREGION'], 1, 2) == c)
    if (debug_level > 1):
      print('MCYFS yield predictions for', c, 'before preprocessing')
      sel_mcyfs_yield_df.show(10)

    sel_mcyfs_yield_df = cyp_preprocessor.preprocessYieldMCYFS(sel_mcyfs_yield_df, crop_id)
    sel_mcyfs_yield_df = sel_mcyfs_yield_df.filter(sel_mcyfs_yield_df['FYEAR'].isin(test_years))
    if (debug_level > 1):
      print('MCYFS yield predictions for', c, 'after preprocessing')
      sel_mcyfs_yield_df.show(10)

    # Check we have yield data for crop
    assert (sel_nuts0_yield_df is not None)
    assert (sel_mcyfs_yield_df is not None)

    country_nuts0_yields[c] = sel_nuts0_yield_df
    country_mcyfs_yields[c] = sel_mcyfs_yield_df

  data_dfs = {
      'WOFOST_DVS' : country_dvs_summaries,
      'AREA_FRACTIONS' : country_area_dfs,
      'YIELD_NUTS0' : country_nuts0_yields,
      'YIELD_PRED_MCYFS' : country_mcyfs_yields,
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
  alg_names = list(cyp_config.getEstimators().keys())
  debug_level = cyp_config.getDebugLevel()

  pd_area_dfs = []
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
  debug_level = cyp_config.getDebugLevel()
  alg_names = list(cyp_config.getEstimators().keys())
  metrics = cyp_config.getEvaluationMetrics()
  # sometimes spark complains about int64 data types
  test_years = [int(yr) for yr in pd_ml_predictions['FYEAR'].unique()]
  early_season_prediction = cyp_config.earlySeasonPrediction()

  spark = sqlCtx.sparkSession
  # We need WOFOST summary, AREA_FRACTIONS, MCYFS yield predictions
  # and NUTS0 Eurostat YIELD for comparison with MCYFS
  data_dfs = getDataForMCYFSComparison(spark, cyp_config, pd_ml_predictions, test_years)
  dvs_summaries = data_dfs['WOFOST_DVS']
  area_fraction_dfs = data_dfs['AREA_FRACTIONS']
  nuts0_yield_dfs = data_dfs['YIELD_NUTS0']
  mcyfs_yield_dfs = data_dfs['YIELD_PRED_MCYFS']

  nuts0_preds_combined_df = None
  for country in dvs_summaries:
    pd_dvs_summary = dvs_summaries[country].toPandas()
    pd_nuts0_yield_df = nuts0_yield_dfs[country].toPandas()
    pd_mcyfs_pred_df = mcyfs_yield_dfs[country].toPandas()
    area_dfs = area_fraction_dfs[country]

    join_cols = ['IDREGION', 'FYEAR']
    pd_country_ml_preds = pd_ml_predictions[pd_ml_predictions['COUNTRY'] == country]
    pd_country_ml_preds = pd_country_ml_preds.drop(columns=['COUNTRY'])
    nuts0_pred_df = aggregatePredictionsToNUTS0(cyp_config, pd_country_ml_preds,
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
      print('\nPredictions and true values for', country)

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

    if (nuts0_preds_combined_df is None):
      nuts0_preds_combined_df = nuts0_pred_df
    else:
      nuts0_preds_combined_df = nuts0_preds_combined_df.append(nuts0_pred_df)

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
    eval_summary_info = '\nAlgorithm Evaluation Summary (NUTS0) for ' + country
    eval_summary_info += '\n-------------------------------------------'
    eval_summary_info += '\n' + alg_df.to_string(index=False) + '\n'
    log_fh.write(eval_summary_info)
    print(eval_summary_info)

  save_predictions = cyp_config.savePredictions()
  if (save_predictions):
    saveNUTS0Predictions(cyp_config, sqlCtx, nuts0_preds_combined_df)
