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

class CYPFeaturizer:
  def __init__(self, cyp_config):
    self.verbose = cyp_config.getDebugLevel()
    self.lt_stats = {}

  def extractFeatures(self, df, data_source, crop_cal,
                      max_cols, avg_cols, extreme_cols,
                      join_cols, fit=False):
    """
    Extract aggregate and extreme features.
    If fit=True, compute and save long-term stats.
    """
    df = df.withColumn('COUNTRY', SparkF.substring('IDREGION', 1, 2))
    crop_cal = getCountryCropCalendar(crop_cal)
    df = df.join(SparkF.broadcast(crop_cal), 'COUNTRY')

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
        lt_stats = df.groupBy('COUNTRY').agg(*stat_aggrs)
        self.lt_stats[data_source] = lt_stats

    df = df.join(SparkF.broadcast(self.lt_stats[data_source]), 'COUNTRY')

    # features for extreme conditions
    for p in extreme_cols:
      if (extreme_cols[p]):
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
