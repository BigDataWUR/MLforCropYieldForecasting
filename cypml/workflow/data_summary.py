from pyspark.sql import Window
import functools

from ..common import globals

if (globals.test_env == 'pkg'):
  SparkT = globals.SparkT
  SparkF = globals.SparkF
  crop_name_dict = globals.crop_name_dict
  crop_id_dict = globals.crop_id_dict
 
  from ..common.util import cropIDToName, cropNameToID

class CYPDataSummarizer:
  def __init__(self, cyp_config):
    self.verbose = cyp_config.getDebugLevel()

  def wofostDVSSummary(self, wofost_df, early_season_end=None):
    """Summary of crop calendar based on DVS"""
    join_cols = ['IDREGION', 'CAMPAIGN_YEAR']
    dvs_summary = wofost_df.select(join_cols).distinct()

    # We find the start and end dekads for DVS ranges
    my_window = Window.partitionBy(join_cols).orderBy('CAMPAIGN_DEKAD')

    wofost_df = wofost_df.withColumn('VALUE', wofost_df['DVS'])
    wofost_df = wofost_df.withColumn('PREV', SparkF.lag(wofost_df['VALUE']).over(my_window))
    wofost_df = wofost_df.withColumn('DIFF', SparkF.when(SparkF.isnull(wofost_df['PREV']), 0)\
                                 .otherwise(wofost_df['VALUE'] - wofost_df['PREV']))
    del_cols = ['VALUE', 'PREV', 'DIFF', 'EARLY_SEASON_END']
    if (early_season_end is None):
      wofost_df = wofost_df.withColumn('EARLY_SEASON_END', SparkF.lit(36))
    else:
      wofost_df = wofost_df.withColumn('EARLY_SEASON_END',
                                       wofost_df['CAMPAIGN_DEKAD'] - wofost_df['DEKAD'] + early_season_end)

    dvs_summary = dvs_summary.join(wofost_df.filter(wofost_df['VALUE'] > 0.0).groupBy(join_cols)\
                                   .agg(SparkF.min('CAMPAIGN_DEKAD').alias('START_DVS')), join_cols)
    dvs_summary = dvs_summary.join(wofost_df.filter(wofost_df.DVS >= 100).groupBy(join_cols)\
                                   .agg(SparkF.min('CAMPAIGN_DEKAD').alias('START_DVS1')), join_cols)
    dvs_summary = dvs_summary.join(wofost_df.filter(wofost_df.DVS >= 200).groupBy(join_cols)\
                                   .agg(SparkF.min('CAMPAIGN_DEKAD').alias('START_DVS2')), join_cols)
    dvs_summary = dvs_summary.join(wofost_df.groupBy(join_cols)\
                                   .agg(SparkF.max('EARLY_SEASON_END').alias('EARLY_SEASON_END')), join_cols)

    wofost_df = wofost_df.drop(*del_cols)
    return dvs_summary

  def indicatorsSummary(self, df, min_cols, max_cols, avg_cols):
    """long term min, max and avg values of selected indicators by region"""
    avgs = []
    if (avg_cols[1:]):
      avgs = [SparkF.bround(SparkF.avg(x), 2).alias('avg(' + x + ')') for x in avg_cols[1:]]
    
    if (min_cols[:1]):
      summary = df.select(min_cols).groupBy('IDREGION').min()
    else:
      summary = df.select(min_cols).groupBy('IDREGION')

    if (max_cols[1:]):
      summary = summary.join(df.select(max_cols).groupBy('IDREGION').max(), 'IDREGION')

    if (avgs):
      summary = summary.join(df.select(avg_cols).groupBy('IDREGION').agg(*avgs), 'IDREGION')
    return summary

  def yieldSummary(self, yield_df):
    """long term min, max and avg values of yield by region"""
    select_cols = ['IDREGION', 'YIELD']
    yield_summary = yield_df.select(select_cols).groupBy('IDREGION').min('YIELD')
    yield_summary = yield_summary.join(yield_df.select(select_cols).groupBy('IDREGION')\
                                       .agg(SparkF.max('YIELD')), 'IDREGION')
    yield_summary = yield_summary.join(yield_df.select(select_cols).groupBy('IDREGION')\
                                       .agg(SparkF.bround(SparkF.avg('YIELD'), 2)\
                                            .alias('avg(YIELD)')), 'IDREGION')
    return yield_summary
