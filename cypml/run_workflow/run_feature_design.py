def wofostMaxFeatureCols():
  """columns or indicators using max aggregation"""
  # must be in sync with crop calendar periods
  max_cols = {
      'p0' : [],
      'p1' : [],
      'p2' : ['WLIM_YB', 'TWC', 'WLAI'],
      'p3' : [],
      'p4' : ['WLIM_YB', 'WLIM_YS', 'TWC', 'WLAI'],
      'p5' : [],
  }

  return max_cols

def wofostAvgFeatureCols():
  """columns or indicators using avg aggregation"""
  # must be in sync with crop calendar periods
  avg_cols = {
      'p0' : [],
      'p1' : [],
      'p2' : ['RSM'],
      'p3' : [],
      'p4' : ['RSM'],
      'p5' : [],
  }

  return avg_cols

def wofostCountFeatureCols():
  """columns or indicators using count aggregation"""
  # must be in sync with crop calendar periods
  count_cols = {
      'p0' : [],
      'p1' : ['RSM'],
      'p2' : ['RSM'],
      'p3' : ['RSM'],
      'p4' : ['RSM'],
      'p5' : [],
  }

  return count_cols

# Meteo Feature ideas:
# Two dry summers caused drop in ground water level:
#   rainfall sums going back to second of half of previous year
# Previous year: high production, prices low, invest less in crop
#   Focus on another crop
def meteoMaxFeatureCols():
  """columns or indicators using max aggregation"""
  # must be in sync with crop calendar periods
  max_cols = { 'p0' : [], 'p1' : [], 'p2' : [], 'p3' : [], 'p4' : [], 'p5' : [] }

  return max_cols

def meteoAvgFeatureCols():
  """columns or indicators using avg aggregation"""
  # must be in sync with crop calendar periods
  avg_cols = {
      'p0' : ['TAVG', 'PREC', 'CWB'],
      'p1' : ['TAVG', 'PREC'],
      'p2' : ['TAVG', 'CWB'],
      'p3' : ['PREC'],
      'p4' : ['CWB'],
      'p5' : ['PREC'],
  }

  return avg_cols

def meteoCountFeatureCols():
  """columns or indicators using count aggregation"""
  # must be in sync with crop calendar periods
  count_cols = {
      'p0' : [],
      'p1' : ['TMIN', 'PREC'],
      'p2' : [],
      'p3' : ['PREC', 'TMAX'],
      'p4' : [],
      'p5' : ['PREC'],
  }

  return count_cols

def rsMaxFeatureCols():
  """columns or indicators using max aggregation"""
  # must be in sync with crop calendar periods
  max_cols = { 'p0' : [], 'p1' : [], 'p2' : [], 'p3' : [], 'p4' : [], 'p5' : [] }

  return max_cols

def rsAvgFeatureCols():
  """columns or indicators using avg aggregation"""
  # must be in sync with crop calendar periods
  avg_cols = {
      'p0' : [],
      'p1' : [],
      'p2' : ['FAPAR'],
      'p3' : [],
      'p4' : ['FAPAR'],
      'p5' : [],
  }

  return avg_cols

def convertFeaturesToPandas(ft_dfs, join_cols):
  """Convert features to pandas and merge"""
  train_ft_df = ft_dfs[0]
  test_ft_df = ft_dfs[1]
  train_ft_df = train_ft_df.withColumnRenamed('CAMPAIGN_YEAR', 'FYEAR')
  test_ft_df = test_ft_df.withColumnRenamed('CAMPAIGN_YEAR', 'FYEAR')
  pd_train_df = train_ft_df.toPandas()
  pd_test_df = test_ft_df.toPandas()

  return [pd_train_df, pd_test_df]

def dropZeroColumns(pd_ft_dfs):
  """Drop columns which have all zeros in training data"""
  pd_train_df = pd_ft_dfs[0]
  pd_test_df = pd_ft_dfs[1]

  pd_train_df = pd_train_df.loc[:, (pd_train_df != 0.0).any(axis=0)]
  pd_train_df = pd_train_df.dropna(axis=1)  
  pd_test_df = pd_test_df[pd_train_df.columns]

  return [pd_train_df, pd_test_df]

def printFeatureData(pd_feature_dfs, join_cols):
  for src in pd_feature_dfs:
    pd_train_fts = pd_feature_dfs[src][0]
    if (pd_train_fts is None):
      continue

    pd_test_fts = pd_feature_dfs[src][1]
    all_cols = list(pd_train_fts.columns)
    aggr_cols = [ c for c in all_cols if (('avg' in c) or ('max' in c))]
    if (len(aggr_cols) > 0):
      print('\n', src, 'Aggregate Features: Training')
      print(pd_train_fts[join_cols + aggr_cols].head(5))
      print('\n', src, 'Aggregate Features: Test')
      print(pd_test_fts[join_cols + aggr_cols].head(5))

    ext_cols = [ c for c in all_cols if (('Z+' in c) or ('Z-' in c) or
                                         ('lt' in c) or ('gt' in c))]
    if (len(ext_cols) > 0):
      print('\n', src, 'Features for Extreme Conditions: Training')
      print(pd_train_fts[join_cols + ext_cols].head(5))
      print('\n', src, 'Features for Extreme Conditions: Test')
      print(pd_test_fts[join_cols + ext_cols].head(5))

def createFeatures(cyp_config, cyp_featurizer, train_test_dfs,
                   summary_dfs, log_fh):
  """Create WOFOST, Meteo and Remote Sensing features"""
  nuts_level = cyp_config.getNUTSLevel()
  use_remote_sensing = cyp_config.useRemoteSensing()
  debug_level = cyp_config.getDebugLevel()

  wofost_train_df = train_test_dfs['WOFOST'][0]
  wofost_test_df = train_test_dfs['WOFOST'][1]
  meteo_train_df = train_test_dfs['METEO'][0]
  meteo_test_df = train_test_dfs['METEO'][1]
  yield_train_df = train_test_dfs['YIELD'][0]

  dvs_summary = summary_dfs['WOFOST_DVS']

  # Filter out years before min yield year
  yield_min_year = yield_train_df.agg({"FYEAR": "MIN"}).collect()[0][0]
  print('Yield min year', yield_min_year)

  wofost_train_df = wofost_train_df.filter(wofost_train_df.CAMPAIGN_YEAR >= yield_min_year)
  meteo_train_df = meteo_train_df.filter(meteo_train_df.CAMPAIGN_YEAR >= yield_min_year)
  dvs_summary = dvs_summary.filter(dvs_summary.CAMPAIGN_YEAR >= yield_min_year)

  rs_train_df = None
  rs_test_df = None
  if (use_remote_sensing):
    rs_train_df = train_test_dfs['REMOTE_SENSING'][0]
    rs_test_df = train_test_dfs['REMOTE_SENSING'][1]
    rs_train_df = rs_train_df.filter(rs_train_df.CAMPAIGN_YEAR >= yield_min_year)
    rs_test_df = rs_test_df.filter(rs_test_df.CAMPAIGN_YEAR >= yield_min_year)

  join_cols = ['IDREGION', 'CAMPAIGN_YEAR']
  aggr_ft_cols = {
      'WOFOST' : [wofostMaxFeatureCols(), wofostAvgFeatureCols()],
      'METEO' : [meteoMaxFeatureCols(), meteoAvgFeatureCols()],
  }

  count_ft_cols = {
      'WOFOST' : wofostCountFeatureCols(),
      'METEO' : meteoCountFeatureCols(),
  }

  train_ft_src_dfs = {
      'WOFOST' : wofost_train_df,
      'METEO' : meteo_train_df,
  }

  test_ft_src_dfs = {
      'WOFOST' : wofost_test_df,
      'METEO' : meteo_test_df,
  }

  if (use_remote_sensing):
    train_ft_src_dfs['REMOTE_SENSING'] = rs_train_df
    test_ft_src_dfs['REMOTE_SENSING'] = rs_test_df
    aggr_ft_cols['REMOTE_SENSING'] = [rsMaxFeatureCols(), rsAvgFeatureCols()]
    count_ft_cols['REMOTE_SENSING'] = {}

  crop_cal_train = dvs_summary
  crop_cal_test = dvs_summary

  train_ft_dfs = {}
  test_ft_dfs = {}
  for ft_src in train_ft_src_dfs:
    train_ft_dfs[ft_src] = cyp_featurizer.extractFeatures(train_ft_src_dfs[ft_src],
                                                          ft_src,
                                                          crop_cal_train,
                                                          aggr_ft_cols[ft_src][0],
                                                          aggr_ft_cols[ft_src][1],
                                                          count_ft_cols[ft_src],
                                                          join_cols,
                                                          True)
    test_ft_dfs[ft_src] = cyp_featurizer.extractFeatures(test_ft_src_dfs[ft_src],
                                                         ft_src,
                                                         crop_cal_test,
                                                         aggr_ft_cols[ft_src][0],
                                                         aggr_ft_cols[ft_src][1],
                                                         count_ft_cols[ft_src],
                                                         join_cols)

  pd_conversion_dict = {
      'WOFOST' : [ train_ft_dfs['WOFOST'], test_ft_dfs['WOFOST'] ],
      'METEO' : [ train_ft_dfs['METEO'], test_ft_dfs['METEO'] ],
  }

  if (use_remote_sensing):
      pd_conversion_dict['REMOTE_SENSING'] = [ train_ft_dfs['REMOTE_SENSING'], test_ft_dfs['REMOTE_SENSING'] ]

  pd_feature_dfs = {}
  for ft_src in pd_conversion_dict:
    pd_feature_dfs[ft_src] = convertFeaturesToPandas(pd_conversion_dict[ft_src], join_cols)

  # Check and drop features with all zeros (possible in early season prediction).
  for ft_src in pd_feature_dfs:
    pd_feature_dfs[ft_src] = dropZeroColumns(pd_feature_dfs[ft_src])

  if (debug_level > 1):
    join_cols = ['IDREGION', 'FYEAR']
    printFeatureData(pd_feature_dfs, join_cols)

  return pd_feature_dfs
