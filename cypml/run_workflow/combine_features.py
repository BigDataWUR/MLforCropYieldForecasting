from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import getFeatureFilename

def combineFeaturesLabels(cyp_config, sqlCtx,
                          prep_train_test_dfs, pd_feature_dfs,
                          join_cols, log_fh):
  """
  Combine wofost, meteo and soil with remote sensing. Combine centroids
  and yield trend if configured. Combine with yield data in the end.
  If configured, save features to a CSV file.
  """
  pd_soil_df = prep_train_test_dfs['SOIL'][0].toPandas()
  pd_yield_train_df = prep_train_test_dfs['YIELD'][0].toPandas()
  pd_yield_test_df = prep_train_test_dfs['YIELD'][1].toPandas()

  # Feature dataframes have already been converted to pandas
  pd_wofost_train_ft = pd_feature_dfs['WOFOST'][0]
  pd_wofost_test_ft = pd_feature_dfs['WOFOST'][1]
  pd_meteo_train_ft = pd_feature_dfs['METEO'][0]
  pd_meteo_test_ft = pd_feature_dfs['METEO'][1]

  crop = cyp_config.getCropName()
  country = cyp_config.getCountryCode()
  use_yield_trend = cyp_config.useYieldTrend()
  use_centroids = cyp_config.useCentroids()
  use_remote_sensing = cyp_config.useRemoteSensing()
  use_gaes = cyp_config.useGAES()
  save_features = cyp_config.saveFeatures()
  use_sample_weights = cyp_config.useSampleWeights()
  debug_level = cyp_config.getDebugLevel()
  
  combine_info = '\nCombine Features and Labels'
  combine_info += '\n---------------------------'
  yield_min_year = pd_yield_train_df['FYEAR'].min()
  combine_info += '\nYield min year ' + str(yield_min_year) + '\n'

  # start with static SOIL data
  pd_train_df = pd_soil_df.copy(deep=True)
  pd_test_df = pd_soil_df.copy(deep=True)
  combine_info += '\nData size after including SOIL data: '
  combine_info += '\nTrain ' + str(len(pd_train_df.index)) + ' rows.'
  combine_info += '\nTest ' + str(len(pd_test_df.index)) + ' rows.\n'

  if (use_gaes):
    pd_aez_df = prep_train_test_dfs['GAES'][0].toPandas()
    pd_aez_df = pd_aez_df.drop(columns=['AEZ_ID'])
    pd_train_df = pd_train_df.merge(pd_aez_df, on=['IDREGION'])
    pd_test_df = pd_test_df.merge(pd_aez_df, on=['IDREGION'])
    combine_info += '\nData size after including GAES data: '
    combine_info += '\nTrain ' + str(len(pd_train_df.index)) + ' rows.'
    combine_info += '\nTest ' + str(len(pd_test_df.index)) + ' rows.\n'

  if (use_centroids):
    # combine with region centroids
    pd_centroids_df = prep_train_test_dfs['CENTROIDS'][0].toPandas()
    pd_train_df = pd_train_df.merge(pd_centroids_df, on=['IDREGION'])
    pd_test_df = pd_test_df.merge(pd_centroids_df, on='IDREGION')
    combine_info += '\nData size after including CENTROIDS data: '
    combine_info += '\nTrain ' + str(len(pd_train_df.index)) + ' rows.'
    combine_info += '\nTest ' + str(len(pd_test_df.index)) + ' rows.\n'

  # combine with WOFOST features
  static_cols = list(pd_train_df.columns)
  pd_train_df = pd_train_df.merge(pd_wofost_train_ft, on=['IDREGION'])
  pd_test_df = pd_test_df.merge(pd_wofost_test_ft, on=['IDREGION'])
  wofost_cols = list(pd_wofost_train_ft.columns)
  col_order = ['IDREGION', 'FYEAR'] + static_cols[1:] + wofost_cols[2:]
  pd_train_df = pd_train_df[col_order]
  pd_test_df = pd_test_df[col_order]
  combine_info += '\nData size after including WOFOST features: '
  combine_info += '\nTrain ' + str(len(pd_train_df.index)) + ' rows.'
  combine_info += '\nTest ' + str(len(pd_test_df.index)) + ' rows.\n'

  # combine with METEO features
  pd_train_df = pd_train_df.merge(pd_meteo_train_ft, on=join_cols)
  pd_test_df = pd_test_df.merge(pd_meteo_test_ft, on=join_cols)
  combine_info += '\nData size after including METEO features: '
  combine_info += '\nTrain ' + str(len(pd_train_df.index)) + ' rows.'
  combine_info += '\nTest ' + str(len(pd_test_df.index)) + ' rows.\n'

  # combine with remote sensing features
  if (use_remote_sensing):
    pd_rs_train_ft = pd_feature_dfs['REMOTE_SENSING'][0]
    pd_rs_test_ft = pd_feature_dfs['REMOTE_SENSING'][1]

    pd_train_df = pd_train_df.merge(pd_rs_train_ft, on=join_cols)
    pd_test_df = pd_test_df.merge(pd_rs_test_ft, on=join_cols)
    combine_info += '\nData size after including REMOTE_SENSING features: '
    combine_info += '\nTrain ' + str(len(pd_train_df.index)) + ' rows.'
    combine_info += '\nTest ' + str(len(pd_test_df.index)) + ' rows.\n'

  if (use_gaes):
    # combine with crop area
    pd_area_train_df = prep_train_test_dfs['CROP_AREA'][0].toPandas()
    pd_area_test_df = prep_train_test_dfs['CROP_AREA'][1].toPandas()
    pd_train_df = pd_train_df.merge(pd_area_train_df, on=join_cols)
    pd_test_df = pd_test_df.merge(pd_area_test_df, on=join_cols)
    combine_info += '\nData size after including CROP_AREA data: '
    combine_info += '\nTrain ' + str(len(pd_train_df.index)) + ' rows.'
    combine_info += '\nTest ' + str(len(pd_test_df.index)) + ' rows.\n'

  if (use_yield_trend):
    # combine with yield trend features
    pd_yield_trend_train_ft = pd_feature_dfs['YIELD_TREND'][0]
    pd_yield_trend_test_ft = pd_feature_dfs['YIELD_TREND'][1]
    pd_train_df = pd_train_df.merge(pd_yield_trend_train_ft, on=join_cols)
    pd_test_df = pd_test_df.merge(pd_yield_trend_test_ft, on=join_cols)
    combine_info += '\nData size after including yield trend features: '
    combine_info += '\nTrain ' + str(len(pd_train_df.index)) + ' rows.'
    combine_info += '\nTest ' + str(len(pd_test_df.index)) + ' rows.\n'

  # combine with yield data
  pd_train_df = pd_train_df.merge(pd_yield_train_df, on=join_cols)
  pd_test_df = pd_test_df.merge(pd_yield_test_df, on=join_cols)
  pd_train_df = pd_train_df.sort_values(by=join_cols)
  pd_test_df = pd_test_df.sort_values(by=join_cols)
  combine_info += '\nData size after including yield (label) data: '
  combine_info += '\nTrain ' + str(len(pd_train_df.index)) + ' rows.'
  combine_info += '\nTest ' + str(len(pd_test_df.index)) + ' rows.\n'

  # sample weights
  if (use_sample_weights):
    assert use_gaes
    pd_train_df['SAMPLE_WEIGHT'] = pd_train_df['CROP_AREA']
    pd_test_df['SAMPLE_WEIGHT'] = pd_test_df['CROP_AREA']

  log_fh.write(combine_info + '\n')
  if (debug_level > 1):
    print(combine_info)
    print('\nAll Features and labels: Training')
    print(pd_train_df.head(5))
    print('\nAll Features and labels: Test')
    print(pd_test_df.head(5))

  if (save_features):
    early_season_prediction = cyp_config.earlySeasonPrediction()
    early_season_end = cyp_config.getEarlySeasonEndDekad()
    feature_file_path = cyp_config.getOutputPath()
    features_file = getFeatureFilename(crop, use_yield_trend,
                                       early_season_prediction, early_season_end,
                                       country)
    save_ft_path = feature_file_path + '/' + features_file
    save_ft_info = '\nSaving features to: ' + save_ft_path + '[train, test].csv'
    log_fh.write(save_ft_info + '\n')
    if (debug_level > 1):
      print(save_ft_info)

    pd_train_df.to_csv(save_ft_path + '_train.csv', index=False, header=True)
    pd_test_df.to_csv(save_ft_path + '_test.csv', index=False, header=True)

    # NOTE: In some environments, Spark can write, but pandas cannot.
    # In such cases, use the following code.
    # spark_train_df = sqlCtx.createDataFrame(pd_train_df)
    # spark_train_df.coalesce(1).write.option('header','true').mode('overwrite').csv(save_ft_path + '_train')
    # spark_test_df = sqlCtx.createDataFrame(pd_test_df)
    # spark_test_df.coalesce(1).write.option('header','true').mode('overwrite').csv(save_ft_path + '_test')

  return pd_train_df, pd_test_df
