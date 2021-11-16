from ..common import globals

if (globals.test_env == 'pkg'):
  SparkT = globals.SparkT
  SparkF = globals.SparkF

def printDataSummary(df, data_source):
  """Print summary information"""
  if (data_source == 'WOFOST_DVS'):
    print('Crop calender information based on WOFOST data')
    max_year = df.select('CAMPAIGN_YEAR').agg(SparkF.max('CAMPAIGN_YEAR')).collect()[0][0]
    df.filter(df.CAMPAIGN_YEAR == max_year).orderBy('IDREGION').show(10)
  else:
    print(data_source, 'indicators summary')
    df.orderBy('IDREGION').show()

def getWOFOSTSummaryCols():
  """WOFOST columns used for data summary"""
  # only RSM has non-zero min values
  min_cols = ['IDREGION', 'RSM']
  max_cols = ['IDREGION'] + ['WLIM_YB', 'WLIM_YS', 'DVS',
                             'WLAI', 'RSM', 'TWC', 'TWR']
  # biomass and DVS values grow over time
  avg_cols = ['IDREGION', 'WLAI', 'RSM', 'TWC', 'TWR']

  return [min_cols, max_cols, avg_cols]

def getMeteoSummaryCols():
  """Meteo columns used for data summary"""
  col_names = ['TMAX', 'TMIN', 'TAVG', 'PREC', 'ET0', 'CWB', 'RAD']
  min_cols = ['IDREGION'] + col_names
  max_cols = ['IDREGION'] + col_names
  avg_cols = ['IDREGION'] + col_names

  return [min_cols, max_cols, avg_cols]

def getRemoteSensingSummaryCols():
  """Remote Sensing columns used for data summary"""
  col_names = ['FAPAR']
  min_cols = ['IDREGION'] + col_names
  max_cols = ['IDREGION'] + col_names
  avg_cols = ['IDREGION'] + col_names

  return [min_cols, max_cols, avg_cols]

def summarizeData(cyp_config, cyp_summarizer, train_test_dfs):
  """
  Summarize data. Create DVS summary to infer crop calendar.
  Summarize selected indicators for each data source.
  """
  wofost_train_df = train_test_dfs['WOFOST'][0]
  wofost_test_df = train_test_dfs['WOFOST'][1]
  meteo_train_df = train_test_dfs['METEO'][0]
  yield_train_df = train_test_dfs['YIELD'][0]

  use_remote_sensing = cyp_config.useRemoteSensing()
  debug_level = cyp_config.getDebugLevel()
  early_season = cyp_config.earlySeasonPrediction()
  early_season_end = None
  if (early_season):
    early_season_end = cyp_config.getEarlySeasonEndDekad()

  # DVS summary (crop calendar)
  # NOTE this summary of crops based on wofost data should be used with caution
  # 1. The summary is per region per year.
  # 2. The summary is based on wofost simulations not real sowing and harvest dates
  dvs_summary_train = cyp_summarizer.wofostDVSSummary(wofost_train_df, early_season_end)
  dvs_summary_train = dvs_summary_train.drop('CALENDAR_END_SEASON', 'CALENDAR_EARLY_SEASON')
  dvs_summary_test = cyp_summarizer.wofostDVSSummary(wofost_test_df, early_season_end)
  dvs_summary_test = dvs_summary_test.drop('CALENDAR_END_SEASON', 'CALENDAR_EARLY_SEASON')
  if (debug_level > 1):
    printDataSummary(dvs_summary_train, 'WOFOST_DVS')

  summary_cols = {
      'WOFOST' : getWOFOSTSummaryCols(),
      'METEO' : getMeteoSummaryCols(),
  }

  summary_sources_dfs = {
      'WOFOST' : wofost_train_df,
      'METEO' : meteo_train_df,
  }

  if (use_remote_sensing):
    rs_train_df = train_test_dfs['REMOTE_SENSING'][0]
    summary_cols['REMOTE_SENSING'] = getRemoteSensingSummaryCols()
    summary_sources_dfs['REMOTE_SENSING'] = rs_train_df

  summary_dfs = {}
  for sum_src in summary_sources_dfs:
    summary_dfs[sum_src] = cyp_summarizer.indicatorsSummary(summary_sources_dfs[sum_src],
                                                            summary_cols[sum_src][0],
                                                            summary_cols[sum_src][1],
                                                            summary_cols[sum_src][2])

  for src in summary_dfs:
    if (debug_level > 2):
      printDataSummary(summary_dfs[src], src)

  yield_summary = cyp_summarizer.yieldSummary(yield_train_df)
  if (debug_level > 2):
    printDataSummary(yield_summary, 'YIELD')

  summary_dfs['WOFOST_DVS'] = [dvs_summary_train, dvs_summary_test]
  summary_dfs['YIELD'] = yield_summary

  return summary_dfs
