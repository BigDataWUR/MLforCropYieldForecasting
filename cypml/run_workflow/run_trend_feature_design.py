from ..common import globals

if (globals.test_env == 'pkg'):
  from ..run_workflow.run_feature_design import getTrendFeatureCols

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

def addFeaturesFromPreviousYears(cyp_config, pd_feature_dfs,
                                 trend_window, test_years, join_cols):
  """Add features from previous years as trend features"""
  debug_level = cyp_config.getDebugLevel()

  for ft_src in pd_feature_dfs:
    trend_cols = getTrendFeatureCols(ft_src)
    if (not trend_cols):
      continue

    pd_ft_train_df = pd_feature_dfs[ft_src][0]
    pd_ft_test_df = pd_feature_dfs[ft_src][1]
    pd_all_df = pd_ft_train_df.append(pd_ft_test_df).sort_values(by=join_cols)
    all_cols = pd_ft_train_df.columns
    common_cols = [c for c in trend_cols if c in all_cols]
    if (not common_cols):
      continue

    for tc in trend_cols:
      if (tc not in all_cols):
        continue

      for yr in range(1, trend_window + 1):
        pd_all_df[tc + '-' + str(yr)] = pd_all_df.groupby(['IDREGION'])[tc].shift(yr)

    pd_ft_train_df = pd_all_df[~pd_all_df['FYEAR'].isin(test_years)]
    pd_ft_train_df = pd_ft_train_df.dropna(axis=0)
    pd_ft_test_df = pd_all_df[pd_all_df['FYEAR'].isin(test_years)]
    pd_ft_test_df = pd_ft_test_df.dropna(axis=0)

    sel_cols = ['IDREGION', 'FYEAR']
    all_cols = pd_ft_train_df.columns
    for tc in trend_cols:
      tc_cols = [c for c in all_cols if tc in c]
      sel_cols += tc_cols

    if ((debug_level > 1) and (len(sel_cols) > 2)):
      print('\n' + ft_src + ' Trend Features: Train')
      print(pd_ft_train_df[sel_cols].sort_values(by=join_cols).head(5).to_string(index=False))
      print('\n' + ft_src + ' Trend Features: Test')
      print(pd_ft_test_df[sel_cols].sort_values(by=join_cols).head(5).to_string(index=False))

    pd_feature_dfs[ft_src] = [pd_ft_train_df, pd_ft_test_df]

  return pd_feature_dfs
