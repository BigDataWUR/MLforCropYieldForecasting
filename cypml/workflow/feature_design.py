from pyspark.sql import Window
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import functools

from ..common import globals

if (globals.test_env == 'pkg'):
  SparkT = globals.SparkT
  SparkF = globals.SparkF

  from .crop_calendar import getCountryCropCalendar
  from .crop_calendar import getCropCalendarPeriods
  from .crop_calendar import getSeasonStartFilter

class CYPFeaturizer:
  def __init__(self, cyp_config):
    self.use_per_year_cc = cyp_config.usePerYearCropCalendar()
    self.use_features_v2 = cyp_config.useFeaturesV2()
    self.use_gaes = cyp_config.useGAES()
    self.verbose = cyp_config.getDebugLevel()
    self.lt_stats = {}

  def extractFeatures(self, df, data_source, crop_cal,
                      max_cols, avg_cols, cum_avg_cols, extreme_cols,
                      join_cols, fit=False):
    """
    Extract aggregate and extreme features.
    If fit=True, compute and save long-term stats.
    """
    df = df.withColumn('COUNTRY', SparkF.substring('IDREGION', 1, 2))
    if (not self.use_per_year_cc):
      if (self.use_gaes):
        df = df.join(crop_cal.select(['IDREGION', 'AEZ_ID']), 'IDREGION')

      crop_cal = getCountryCropCalendar(crop_cal)
      df = df.join(SparkF.broadcast(crop_cal), 'COUNTRY')
    else:
      df = df.join(crop_cal, join_cols)

    # Calculate cumulative sums
    if (self.use_features_v2 and cum_avg_cols):
      w = Window.partitionBy(join_cols).orderBy('CAMPAIGN_DEKAD')\
                .rangeBetween(Window.unboundedPreceding, 0)
      after_season_start = getSeasonStartFilter(df)
      for c in cum_avg_cols:
        df = df.withColumn(c, SparkF.sum(SparkF.when(after_season_start, df[c])\
                                         .otherwise(0.0)).over(w))

    cc_periods = getCropCalendarPeriods(df)
    aggrs = []
    # max aggregation
    for p in max_cols:
      if (max_cols[p]):
        aggrs += [SparkF.bround(SparkF.max(SparkF.when(cc_periods[p], df[x])), 2)\
                  .alias('max' + x + p) for x in max_cols[p] ]

    # avg aggregation
    for p in avg_cols:
      if (avg_cols[p]):
        aggrs += [SparkF.bround(SparkF.avg(SparkF.when(cc_periods[p], df[x])), 2)\
                  .alias('avg' + x + p) for x in avg_cols[p] ]

    # if not computing extreme features, we can return
    if (not extreme_cols):
      ft_df = df.groupBy(join_cols).agg(*aggrs)
      return ft_df

    # compute long-term stats and save them
    if (fit):
      stat_aggrs = []
      for p in extreme_cols:
        if (extreme_cols[p]):
          stat_aggrs += [ SparkF.bround(SparkF.avg(SparkF.when(cc_periods[p], df[x])), 2)\
                         .alias('avg' + x + p) for x in extreme_cols[p] ]
          stat_aggrs += [ SparkF.bround(SparkF.stddev(SparkF.when(cc_periods[p], df[x])), 2)\
                         .alias('std' + x + p) for x in extreme_cols[p] ]

      if (stat_aggrs):
        if (self.use_gaes):
          lt_stats = df.groupBy('AEZ_ID').agg(*stat_aggrs)
        else:
          lt_stats = df.groupBy('COUNTRY').agg(*stat_aggrs)

        self.lt_stats[data_source] = lt_stats

    if (self.use_gaes):
      df = df.join(SparkF.broadcast(self.lt_stats[data_source]), 'AEZ_ID')
    else:
      df = df.join(SparkF.broadcast(self.lt_stats[data_source]), 'COUNTRY')

    # features for extreme conditions
    for p in extreme_cols:
      if (extreme_cols[p]):
        if (self.use_features_v2):
          # sum zscore for values < long-term average
          aggrs += [ SparkF.bround(SparkF.sum(SparkF.when(((df[x] - df['avg' + x + p]) < 0) & cc_periods[p],
                                                          (df['avg' + x + p] - df[x]) / df['std' + x + p])), 2)\
                     .alias('Z-' + x + p) for x in extreme_cols[p] ]
          # sum zscore for values > long-term average
          aggrs += [ SparkF.bround(SparkF.sum(SparkF.when(((df[x] - df['avg' + x + p]) > 0) & cc_periods[p],
                                                          (df[x] - df['avg' + x + p]) / df['std' + x + p])), 2)\
                     .alias('Z+' + x + p) for x in extreme_cols[p] ]

        else:
          # Count of days or dekads with values crossing threshold
          for i in range(1, 3):
            aggrs += [ SparkF.sum(SparkF.when((df[x] > (df['avg' + x + p] + i * df['std' + x + p])) &
                                              cc_periods[p], 1))\
                      .alias(x + p + 'gt' + str(i) + 'STD') for x in extreme_cols[p] ]
            aggrs += [ SparkF.sum(SparkF.when((df[x] < (df['avg' + x + p] - i * df['std' + x + p])) &
                                              cc_periods[p], 1))\
                      .alias(x + p + 'lt' + str(i) + 'STD') for x in extreme_cols[p] ]

    ft_df = df.groupBy(join_cols).agg(*aggrs)
    ft_df = ft_df.na.fill(0.0)
    return ft_df
