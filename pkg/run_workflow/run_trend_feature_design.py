def createYieldTrendFeatures(cyp_config, cyp_trend_est,
                             yield_train_df, yield_test_df, test_years):
  """Create yield trend features"""
  join_cols = ['IDREGION', 'FYEAR']
  find_optimal = cyp_config.findOptimalTrendWindow()
  trend_window = cyp_config.getTrendWindows()[0]
  debug_level = cyp_config.getDebugLevel()
  yield_df = yield_train_df.union(yield_test_df.select(yield_train_df.columns))

  if (find_optimal):
    pd_train_features = cyp_trend_est.getOptimalWindowTrendFeatures(yield_train_df)
    pd_test_features = cyp_trend_est.getOptimalWindowTrendFeatures(yield_df)
  else:
    pd_train_features = cyp_trend_est.getFixedWindowTrendFeatures(yield_train_df)
    pd_test_features = cyp_trend_est.getFixedWindowTrendFeatures(yield_df)

  pd_test_features = pd_test_features[pd_test_features['FYEAR'].isin(test_years)]
  prev_year_cols = ['YEAR-' + str(i) for i in range(1, trend_window + 1)]
  pd_train_features = pd_train_features.drop(columns=prev_year_cols)
  pd_test_features = pd_test_features.drop(columns=prev_year_cols)

  if (debug_level > 1):
    print('\nYield Trend Features: Train')
    join_cols = ['IDREGION', 'FYEAR']
    print(pd_train_features.sort_values(by=join_cols).head(5))
    print('Total', len(pd_train_features.index), 'rows')
    print('\nYield Trend Features: Test')
    print(pd_test_features.sort_values(by=join_cols).head(5))
    print('Total', len(pd_test_features.index), 'rows')

  return pd_train_features, pd_test_features
