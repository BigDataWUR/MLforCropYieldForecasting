from pyspark.sql import Window
import functools

import globals

if (globals.test_env == 'pkg'):
  SpartT = globals.SparkT
  SparkF = globals.SparkF

# Training, Test Split
def splitTrainingTest(df, src, test_years):
  """Splitting given df into training and test dataframes."""
  train_df = df.filter(~df.CAMPAIGN_YEAR.isin(test_years))
  test_df = df.filter(df.CAMPAIGN_YEAR.isin(test_years))

  return [train_df, test_df]

# Feature design

def getIndicatorStats(df, ind):
  """Return min, max, avg and stddev of indicator values for given period"""
  ind_min = df.agg(SparkF.bround(SparkF.min(ind), 2)).collect()[0][0]
  ind_max = df.agg(SparkF.bround(SparkF.max(ind), 2)).collect()[0][0]
  ind_avg = df.agg(SparkF.bround(SparkF.avg(ind), 2)).collect()[0][0]
  ind_std = df.agg(SparkF.bround(SparkF.stddev(ind), 2)).collect()[0][0]

  ind_stats =  {
      'MIN' : ind_min,
      'MAX' : ind_max,
      'AVG' : ind_avg,
      'STD' : ind_std,
  }

  return ind_stats

def getPeriodIndicatorStats(df, extreme_cols, start_dekad, end_dekad):
    """Calculate AVG, STDDEV for given indicators for given period"""
    period_filter = (df['CAMPAIGN_DEKAD'] >= start_dekad) & (df['CAMPAIGN_DEKAD'] <= end_dekad)

    ind_stats = dict(map(lambda ind: (ind, getIndicatorStats(df.filter(period_filter),
                                                             ind)),
                         extreme_cols))
    return ind_stats

def getAggrFeature(df, ft_name, ft_def, join_cols):
  """Individual aggregate feature based on ft_def"""
  aggr_name = ft_def[0]
  col_name = ft_def[1]
  start_dekad = ft_def[2]
  end_dekad = ft_def[3]
  sel_cols = join_cols + [col_name]
  period_filter = (df.CAMPAIGN_DEKAD >= start_dekad) & (df.CAMPAIGN_DEKAD <= end_dekad)
  df = df.select(sel_cols).filter(period_filter)

  ft_df = df.select(join_cols).distinct()
  if (aggr_name == 'AVG'):
    ft_df = df.groupBy(join_cols).agg(SparkF.bround(SparkF.avg(col_name), 2).alias(ft_name))
  elif (aggr_name == 'MAX'):
    ft_df = df.groupBy(join_cols).agg(SparkF.bround(SparkF.max(col_name), 2).alias(ft_name))
  
  return ft_df

def getPeriodAggrFeatures(df, period_name, start_dekad, end_dekad,
                          max_cols, avg_cols, join_cols):
  """Aggregate features for given period"""
  aggrs = []
  period_filter = (df.CAMPAIGN_DEKAD >= start_dekad) & (df.CAMPAIGN_DEKAD <= end_dekad)
  if (end_dekad > start_dekad):
    aggrs += [ SparkF.bround(SparkF.max(x), 2).alias('max' + x + period_name) for x in max_cols ]
    aggrs += [ SparkF.bround(SparkF.avg(x), 2).alias('avg' + x + period_name) for x in avg_cols ]

  if (len(aggrs) > 0):
    return df.filter(period_filter).groupBy(join_cols).agg(*aggrs)
  else:
    return df.select(join_cols).distinct()

def getExtremeFeature(df, ft_name, ft_def, join_cols):
  """Count days or dekads with values above or below given threshold for given period"""
  col_name = ft_def[0]
  comp_sign = ft_def[1]
  thresh = ft_def[2]
  start_dekad = ft_def[3]
  end_dekad = ft_def[4]
  sel_cols = join_cols + [col_name]

  ft_filter = (df.CAMPAIGN_DEKAD >= start_dekad) & (df.CAMPAIGN_DEKAD <= end_dekad)
  if (comp_sign == '>'):
    ft_filter = ft_filter & (df[col_name] > thresh)
  else:
    ft_filter = ft_filter & (df[col_name] < thresh)

  df = df.select(sel_cols).filter(ft_filter)
  ft_df = df.groupBy(join_cols).agg(SparkF.count(col_name).alias(ft_name))
  return ft_df

def getColExtremeFeatures(df, period_name, period_stats, col_name, join_cols):
  """Features for extreme condtions based on given column"""
  aggrs = []
  for i in range(1, 3):
    gt_thresh = period_stats[col_name]['AVG'] + i * period_stats[col_name]['STD']
    lt_thresh = period_stats[col_name]['AVG'] - i * period_stats[col_name]['STD']

    # uppper threshold
    if (gt_thresh <= period_stats[col_name]['MAX']):
      thcross_col = col_name + 'gt' + str(i) + 'STD'
      df = df.withColumn(thcross_col,
                         SparkF.when(df[col_name] > gt_thresh, 1).otherwise(0))
      ft_name = col_name + period_name + 'gt' + str(i) + 'STD'
      aggrs.append(SparkF.bround(SparkF.sum(thcross_col), 2).alias(ft_name))

    # lower threshold
    if (lt_thresh >= period_stats[col_name]['MIN']):
      thcross_col = col_name + 'lt' + str(i) + 'STD'
      df = df.withColumn(thcross_col,
                         SparkF.when(df[col_name] < lt_thresh, 1).otherwise(0))
      ft_name = col_name + period_name + 'lt' + str(i) + 'STD'
      aggrs.append(SparkF.bround(SparkF.sum(thcross_col), 2).alias(ft_name))

  ft_df = df.groupBy(join_cols).agg(*aggrs)
  return ft_df

def getPeriodExtremeFeatures(df, period_name, start_dekad, end_dekad,
                             period_stats, extreme_cols, join_cols):
  """Features for extreme condtions for given period"""
  period_filter = (df.CAMPAIGN_DEKAD >= start_dekad) & (df.CAMPAIGN_DEKAD <= end_dekad)
  sel_cols = join_cols + extreme_cols
  df = df.select(sel_cols).filter(period_filter)

  if ((len(extreme_cols) == 0) or (df.count() == 0)):
    return df.select(join_cols).distinct()

  ft_dfs = list(map(lambda c: getColExtremeFeatures(df,
                                                    period_name,
                                                    period_stats,
                                                    c,
                                                    join_cols),
                      extreme_cols))
  extreme_fts = functools.reduce(combineFeatureDataFrames, ft_dfs)

  return extreme_fts

def combineFeatureDataFrames(df1, df2):
  """Join two feature data frames"""
  join_cols = ['IDREGION', 'CAMPAIGN_YEAR']
  df1 = df1.join(df2, join_cols, 'full')
  df1 = df1.na.fill(0)
  return df1

def combinePandasYieldTrendDFs(df1, df2):
  """Join two pandas yield trend data frames"""
  join_cols = ['IDREGION', 'FYEAR']
  df1 = df1.merge(df2, on=join_cols)
  return df1
