def printPreprocessingInformation(df, data_source, order_cols, crop_season=None):
  """Print preprocessed data and additional debug information"""
  df_regions = [reg[0] for reg in df.select('IDREGION').distinct().collect()]
  print(data_source , 'data available for', len(df_regions), 'region(s)')
  if (crop_season is not None):
    print('Season end information')
    crop_season.orderBy(['IDREGION', 'FYEAR']).show(10)

  print(data_source, 'data')
  df.orderBy(order_cols).show(10)

def preprocessData(cyp_config, cyp_preprocessor, data_dfs):
  crop_id = cyp_config.getCropID()
  nuts_level = cyp_config.getNUTSLevel()
  season_crosses_calyear = cyp_config.seasonCrossesCalendarYear()
  clean_data = cyp_config.cleanData()
  use_centroids = cyp_config.useCentroids()
  use_remote_sensing = cyp_config.useRemoteSensing()
  use_gaes = cyp_config.useGAES()
  debug_level = cyp_config.getDebugLevel()

  order_cols = ['IDREGION', 'CAMPAIGN_YEAR', 'CAMPAIGN_DEKAD']
  # wofost data
  wofost_df = data_dfs['WOFOST']
  wofost_df = wofost_df.filter(wofost_df['CROP_ID'] == crop_id).drop('CROP_ID')
  crop_season = cyp_preprocessor.getCropSeasonInformation(wofost_df, season_crosses_calyear)
  wofost_df = cyp_preprocessor.preprocessWofost(wofost_df, crop_season, season_crosses_calyear)
  wofost_regions = [reg[0] for reg in wofost_df.select('IDREGION').distinct().collect()]
  data_dfs['WOFOST'] = wofost_df
  if (debug_level > 1):
    printPreprocessingInformation(wofost_df, 'WOFOST', order_cols, crop_season)

  # meteo data
  meteo_df = data_dfs['METEO']
  meteo_df = cyp_preprocessor.preprocessMeteo(meteo_df, crop_season, season_crosses_calyear)
  assert (meteo_df is not None)
  data_dfs['METEO'] = meteo_df
  if (debug_level > 1):
    printPreprocessingInformation(meteo_df, 'METEO', order_cols)

  # remote sensing data
  rs_df = None
  if (use_remote_sensing):
    rs_df = data_dfs['REMOTE_SENSING']
    rs_df = rs_df.drop('IDCOVER')

    # if other data is at NUTS3, convert rs_df to NUTS3 using parent region data
    if (nuts_level == 'NUTS3'):
      rs_df = cyp_preprocessor.remoteSensingNUTS2ToNUTS3(rs_df, wofost_regions)

    rs_df = cyp_preprocessor.preprocessRemoteSensing(rs_df, crop_season, season_crosses_calyear)
    assert (rs_df is not None)
    data_dfs['REMOTE_SENSING'] = rs_df
    if (debug_level > 1):
      printPreprocessingInformation(rs_df, 'REMOTE_SENSING', order_cols)

  order_cols = ['IDREGION']
  # centroids and distance to coast
  centroids_df = None
  if (use_centroids):
    centroids_df = data_dfs['CENTROIDS']
    centroids_df = cyp_preprocessor.preprocessCentroids(centroids_df)
    data_dfs['CENTROIDS'] = centroids_df
    if (debug_level > 1):
      printPreprocessingInformation(centroids_df, 'CENTROIDS', order_cols)

  # soil data
  soil_df = data_dfs['SOIL']
  soil_df = cyp_preprocessor.preprocessSoil(soil_df)
  data_dfs['SOIL'] = soil_df
  if (debug_level > 1):
    printPreprocessingInformation(soil_df, 'SOIL', order_cols)

  # agro-environmental zones
  if (use_gaes):
    aez_df = data_dfs['GAES']
    aez_df = cyp_preprocessor.preprocessGAES(aez_df, crop_id)
    data_dfs['GAES'] = aez_df
    if (debug_level > 1):
      printPreprocessingInformation(aez_df, 'GAES', order_cols)

    # crop area data
    order_cols = ['IDREGION', 'FYEAR']
    crop_area_df = data_dfs['CROP_AREA']
    crop_area_df = cyp_preprocessor.preprocessCropArea(crop_area_df, crop_id)
    data_dfs['CROP_AREA'] = crop_area_df
    if (debug_level > 1):
      printPreprocessingInformation(crop_area_df, 'CROP_AREA', order_cols)

  order_cols = ['IDREGION', 'FYEAR']
  # yield_data
  yield_df = data_dfs['YIELD']
  if (debug_level > 1):
    print('Yield before preprocessing')
    yield_df.show(10)

  yield_df = cyp_preprocessor.preprocessYield(yield_df, crop_id, clean_data)
  assert (yield_df is not None)
  data_dfs['YIELD'] = yield_df
  if (debug_level > 1):
    print('Yield after preprocessing')
    yield_df.show(10)

  return data_dfs
