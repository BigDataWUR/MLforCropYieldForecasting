from ..common import globals

if (globals.test_env == 'pkg'):
  from ..workflow.train_test_split import CYPTrainTestSplitter

def printTrainTestSplits(train_df, test_df, src, order_cols):
  """Print Training and Test Splits"""
  print('\n', src, 'training data')
  train_df.orderBy(order_cols).show(5)
  print('\n', src, 'test data')
  test_df.orderBy(order_cols).show(5)

# Training, Test Split
# --------------------
def splitTrainingTest(df, year_col, test_years):
  """Splitting given df into training and test dataframes."""
  train_df = df.filter(~df[year_col].isin(test_years))
  test_df = df.filter(df[year_col].isin(test_years))

  return [train_df, test_df]

def splitDataIntoTrainingTestSets(cyp_config, preprocessed_dfs, log_fh):
  """
  Split preprocessed data into training and test sets based on
  availability of yield data.
  """
  nuts_level = cyp_config.getNUTSLevel()
  use_centroids = cyp_config.useCentroids()
  use_remote_sensing = cyp_config.useRemoteSensing()
  use_gaes = cyp_config.useGAES()
  debug_level = cyp_config.getDebugLevel()

  yield_df = preprocessed_dfs['YIELD']
  train_test_splitter = CYPTrainTestSplitter(cyp_config)
  test_years = train_test_splitter.trainTestSplit(yield_df)
  test_years_info = '\nTest years: ' + ', '.join([str(y) for y in sorted(test_years)]) + '\n'
  log_fh.write(test_years_info + '\n')
  print(test_years_info)

  # Times series data used for feature design.
  ts_data_sources = {
      'WOFOST' : preprocessed_dfs['WOFOST'],
      'METEO' : preprocessed_dfs['METEO'],
  }

  if (use_remote_sensing):
    ts_data_sources['REMOTE_SENSING'] = preprocessed_dfs['REMOTE_SENSING']

  train_test_dfs = {}
  for ts_src in ts_data_sources:
    train_test_dfs[ts_src] = splitTrainingTest(ts_data_sources[ts_src], 'CAMPAIGN_YEAR', test_years)

  # SOIL, GAES and CENTROIDS data are static.
  train_test_dfs['SOIL'] = [preprocessed_dfs['SOIL'], preprocessed_dfs['SOIL']]
  if (use_gaes):
    train_test_dfs['GAES'] = [preprocessed_dfs['GAES'], preprocessed_dfs['GAES']]

  if (use_centroids):
    train_test_dfs['CENTROIDS'] = [preprocessed_dfs['CENTROIDS'],
                                   preprocessed_dfs['CENTROIDS']]

  # crop area
  if (use_gaes):
    crop_area_df = preprocessed_dfs['CROP_AREA']
    train_test_dfs['CROP_AREA'] = splitTrainingTest(crop_area_df, 'FYEAR', test_years)

  # yield data
  train_test_dfs['YIELD'] = splitTrainingTest(yield_df, 'FYEAR', test_years)

  if (debug_level > 2):
    for src in train_test_dfs:
      if (src in ts_data_sources):
        order_cols = ['IDREGION', 'CAMPAIGN_YEAR', 'CAMPAIGN_DEKAD']
      elif ((src == 'YIELD') or (src == 'CROP_AREA')):
        order_cols = ['IDREGION', 'FYEAR']
      else:
        order_cols = ['IDREGION']

      train_df = train_test_dfs[src][0]
      test_df = train_test_dfs[src][1]
      printTrainTestSplits(train_df, test_df, src, order_cols)

  return train_test_dfs, test_years
