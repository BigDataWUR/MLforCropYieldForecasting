from pyspark.sql import Window

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import cropNameToID, cropIDToName
  from ..common.util import getYear, getDekad

  SparkT = globals.SparkT
  SparkF = globals.SparkF
  crop_id_dict = globals.crop_id_dict
  crop_name_dict = globals.crop_name_dict

class CYPDataPreprocessor:
  def __init__(self, spark, cyp_config):
    self.spark = spark
    self.verbose = cyp_config.getDebugLevel()

  def extractYearDekad(self, df):
    """Extract year and dekad from date_col in yyyyMMdd format."""
    # Conversion to string type is required to make getYear(), getMonth() etc. work correctly.
    # They use to_date() function to verify valid dates and to_date() expects the date column to be string.
    df = df.withColumn('DATE', df['DATE'].cast("string"))
    df = df.select('*',
                   getYear('DATE').alias('FYEAR'),
                   getDekad('DATE').alias('DEKAD'))

    # Bring FYEAR, DEKAD to the front
    col_order = df.columns[:2] + df.columns[-2:] + df.columns[2:-2]
    df = df.select(col_order).drop('DATE')
    return df

  def getCropSeasonInformation(self, wofost_df, season_crosses_calyear):
    """Crop season information based on WOFOST DVS"""
    join_cols = ['IDREGION', 'FYEAR']
    if (('DATE' in wofost_df.columns) and ('FYEAR' not in wofost_df.columns)):
      wofost_df = self.extractYearDekad(wofost_df)

    crop_season = wofost_df.select(join_cols).distinct()
    diff_window = Window.partitionBy(join_cols).orderBy('DEKAD')
    cs_window = Window.partitionBy('IDREGION').orderBy('FYEAR')

    wofost_df = wofost_df.withColumn('VALUE', wofost_df['DVS'])
    wofost_df = wofost_df.withColumn('PREV', SparkF.lag(wofost_df['VALUE']).over(diff_window))
    wofost_df = wofost_df.withColumn('DIFF', SparkF.when(SparkF.isnull(wofost_df['PREV']), 0)\
                                     .otherwise(wofost_df['VALUE'] - wofost_df['PREV']))
    # calculate end of season dekad
    dvs_nochange_filter = ((wofost_df['VALUE'] >= 200) & (wofost_df['DIFF'] == 0.0))
    year_end_filter = (wofost_df['DEKAD'] == 36)
    if (season_crosses_calyear):
      value_zero_filter =  (wofost_df['VALUE'] == 0)
    else:
      value_zero_filter =  ((wofost_df['PREV'] >= 200) & (wofost_df['VALUE'] == 0))

    end_season_filter = (dvs_nochange_filter | value_zero_filter | year_end_filter)
    crop_season = crop_season.join(wofost_df.filter(end_season_filter).groupBy(join_cols)\
                                   .agg(SparkF.min('DEKAD').alias('SEASON_END_DEKAD')), join_cols)
    wofost_df = wofost_df.drop('VALUE', 'PREV', 'DIFF')

    # We take the max of SEASON_END_DEKAD for current campaign and next campaign
    # to determine which dekads go to next campaign year.
    max_year = crop_season.agg(SparkF.max('FYEAR')).collect()[0][0]
    min_year = crop_season.agg(SparkF.min('FYEAR')).collect()[0][0]
    crop_season = crop_season.withColumn('NEXT_SEASON_END', SparkF.when(crop_season['FYEAR'] == max_year,
                                                                        crop_season['SEASON_END_DEKAD'])\
                                         .otherwise(SparkF.lead(crop_season['SEASON_END_DEKAD']).over(cs_window)))
    crop_season = crop_season.withColumn('SEASON_END',
                                         SparkF.when(crop_season['SEASON_END_DEKAD'] > crop_season['NEXT_SEASON_END'],
                                                     crop_season['SEASON_END_DEKAD'])\
                                         .otherwise(crop_season['NEXT_SEASON_END']))
    crop_season = crop_season.withColumn('PREV_SEASON_END', SparkF.when(crop_season['FYEAR'] == min_year, 0)\
                                         .otherwise(SparkF.lag(crop_season['SEASON_END']).over(cs_window)))
    crop_season = crop_season.select(join_cols + ['PREV_SEASON_END', 'SEASON_END'])

    return crop_season

  def alignDataToCropSeason(self, df, crop_season, season_crosses_calyear):
    """Calculate CAMPAIGN_YEAR, CAMPAIGN_DEKAD based on crop_season"""
    join_cols = ['IDREGION', 'FYEAR']
    max_year = crop_season.agg(SparkF.max('FYEAR')).collect()[0][0]
    min_year = crop_season.agg(SparkF.min('FYEAR')).collect()[0][0]
    df = df.join(crop_season, join_cols)

    # Dekads > SEASON_END belong to next campaign year
    df = df.withColumn('CAMPAIGN_YEAR',
                       SparkF.when(df['DEKAD'] > df['SEASON_END'], df['FYEAR'] + 1)\
                       .otherwise(df['FYEAR']))
    # min_year has no previous season information. We align CAMPAIGN_DEKAD to end in 36.
    # For other years, dekads < SEASON_END are adjusted based on PREV_SEASON_END.
    # Dekads > SEASON_END get renumbered from 1 (for next campaign).
    df = df.withColumn('CAMPAIGN_DEKAD',
                       SparkF.when(df['CAMPAIGN_YEAR'] == min_year, df['DEKAD'] + 36 - df['SEASON_END'])\
                       .otherwise(SparkF.when(df['DEKAD'] > df['SEASON_END'], df['DEKAD'] - df['SEASON_END'])\
                                  .otherwise(df['DEKAD'] + 36 - df['PREV_SEASON_END'])))

    # Columns should be IDREGION, FYEAR, DEKAD, ..., CAMPAIGN_YEAR, CAMPAIGN_DEKAD.
    # Bring CAMPAIGN_YEAR and CAMPAIGN_DEKAD to the front.
    col_order = df.columns[:3] + df.columns[-2:] + df.columns[3:-2]
    df = df.select(col_order)
    if (season_crosses_calyear):
      # For crop with two seasons, remove the first year. Data from the first year
      # only contributes to the second year and we have already moved useful data
      # to the second year (or first campaign year).
      df = df.filter(df['CAMPAIGN_YEAR'] > min_year)

    # In both cases, remove extra rows beyond max campaign year
    df = df.filter(df['CAMPAIGN_YEAR'] <= max_year)
    return df

  def preprocessWofost(self, wofost_df, crop_season, season_crosses_calyear):
    """
    Extract year and dekad from date. Use crop_season to compute
    CAMPAIGN_YEAR and CAMPAIGN_DEKAD.
    """
    drop_cols = crop_season.columns[2:]
    if (('DATE' in wofost_df.columns) and ('FYEAR' not in wofost_df.columns)):
      wofost_df = self.extractYearDekad(wofost_df)

    join_cols = ['IDREGION', 'FYEAR']
    wofost_df = self.alignDataToCropSeason(wofost_df, crop_season, season_crosses_calyear)

    # WOFOST indicators come after IDREGION, FYEAR, DEKAD, CAMPAIGN_YEAR, CAMPAIGN_DEKAD.
    wofost_inds = wofost_df.columns[5:]
    # set indicators values for dekads after end of season to zero.
    # TODO - Dilli: Find a way to avoid the for loop.
    for ind in wofost_inds:
      wofost_df = wofost_df.withColumn(ind,
                                       SparkF.when(wofost_df['DEKAD'] < wofost_df['SEASON_END'],
                                                   wofost_df[ind])\
                                       .otherwise(0))

    wofost_df = wofost_df.drop(*drop_cols)
    return wofost_df

  def preprocessMeteo(self, meteo_df, crop_season, season_crosses_calyear):
    """
    Extract year and dekad from date, calculate CWB.
    Use crop_season to compute CAMPAIGN_YEAR and CAMPAIGN_DEKAD.
    """
    join_cols = ['IDREGION', 'FYEAR']
    drop_cols = crop_season.columns[2:]
    meteo_df = meteo_df.drop('IDCOVER')
    meteo_df = meteo_df.withColumn('CWB',
                                   SparkF.bround(meteo_df['PREC'] - meteo_df['ET0'], 2))
    if (('DATE' in meteo_df.columns) and ('FYEAR' not in meteo_df.columns)):
      meteo_df = self.extractYearDekad(meteo_df)

    meteo_df = self.alignDataToCropSeason(meteo_df, crop_season, season_crosses_calyear)
    meteo_df = meteo_df.drop(*drop_cols)
    return meteo_df

  def preprocessMeteoDaily(self, meteo_df):
    """
    Convert daily meteo data to dekadal. Takes avg for all indicators
    except TMAX (take max), TMIN (take min), PREC (take sum), ET0 (take sum), CWB (take sum).
    """
    self.spark.catalog.dropTempView('meteo_daily')
    meteo_df.createOrReplaceTempView('meteo_daily')
    join_cols = ['IDREGION', 'CAMPAIGN_YEAR', 'CAMPAIGN_DEKAD']
    join_df = meteo_df.select(join_cols + ['FYEAR', 'DEKAD']).distinct()

    # We are ignoring VPRES, WSPD and RELH at the moment
    # avg(VPRES) as VPRES1, avg(WSPD) as WSPD1, avg(RELH) as RELH1,
    # TMAX| TMIN| TAVG| VPRES| WSPD| PREC| ET0| RAD| RELH| CWB
    #
    # It seems keeping same name after aggregation is fine. We are using a
    # different name just to be sure nothing untoward happens.
    query = 'select IDREGION, CAMPAIGN_YEAR, CAMPAIGN_DEKAD, '
    query = query + ' max(TMAX) as TMAX1, min(TMIN) as TMIN1, '
    query = query + ' bround(avg(TAVG), 2) as TAVG1, bround(sum(PREC), 2) as PREC1, '
    query = query + ' bround(sum(ET0), 2) as ET01, bround(avg(RAD), 2) as RAD1, '
    query = query + ' bround(sum(CWB), 2) as CWB1 '
    query = query + ' from meteo_daily group by IDREGION, CAMPAIGN_YEAR, CAMPAIGN_DEKAD '
    query = query + ' order by IDREGION, CAMPAIGN_YEAR, CAMPAIGN_DEKAD'
    meteo_df = self.spark.sql(query).cache()

    # rename the columns
    selected_cols = ['TMAX', 'TMIN', 'TAVG', 'PREC', 'ET0', 'RAD', 'CWB']
    for scol in selected_cols:
      meteo_df = meteo_df.withColumnRenamed(scol + '1', scol)

    meteo_df = meteo_df.join(join_df, join_cols)
    # Bring FYEAR, DEKAD to the front
    col_order = meteo_df.columns[:1] + meteo_df.columns[-2:] + meteo_df.columns[1:-2]
    meteo_df = meteo_df.select(col_order)

    return meteo_df

  def remoteSensingNUTS2ToNUTS3(self, rs_df, nuts3_regions):
    """
    Convert NUTS2 remote sensing data to NUTS3.
    Remote sensing values for NUTS3 regions are inherited from parent regions.
    NOTE this function is called before preprocessRemoteSensing.
    preprocessRemoteSensing expects crop_season and rs_df to be at the same NUTS level.
    """
    NUTS3_dict = {}

    for nuts3 in nuts3_regions:
      nuts2 = nuts3[:4]
      try:
        existing = NUTS3_dict[nuts2]
      except KeyError as e:
        existing = []

      NUTS3_dict[nuts2] = existing + [nuts3]

    rs_NUTS3 = rs_df.rdd.map(lambda r: (NUTS3_dict[r[0]] if r[0] in NUTS3_dict else [], r[1], r[2]))
    rs_NUTS3_df = rs_NUTS3.toDF(['NUTS3_REG', 'DATE', 'FAPAR'])
    rs_NUTS3_df = rs_NUTS3_df.withColumn('IDREGION', SparkF.explode('NUTS3_REG')).drop('NUTS3_REG')
    rs_NUTS3_df = rs_NUTS3_df.select('IDREGION', 'DATE', 'FAPAR')

    return rs_NUTS3_df

  def preprocessRemoteSensing(self, rs_df, crop_season, season_crosses_calyear):
    """
    Extract year and dekad from date.
    Use crop_season to compute CAMPAIGN_YEAR and CAMPAIGN_DEKAD.
    NOTE crop_season and rs_df must be at the same NUTS level.
    """
    join_cols = ['IDREGION', 'FYEAR']
    drop_cols = crop_season.columns[2:]
    if (('DATE' in rs_df.columns) and ('FYEAR' not in rs_df.columns)):
      rs_df = self.extractYearDekad(rs_df)

    rs_df = self.alignDataToCropSeason(rs_df, crop_season, season_crosses_calyear)
    rs_df = rs_df.drop(*drop_cols)
    return rs_df

  def preprocessCentroids(self, centroids_df):
    df_cols = centroids_df.columns
    centroids_df = centroids_df.withColumn('CENTROID_X', SparkF.bround('CENTROID_X', 2))
    centroids_df = centroids_df.withColumn('CENTROID_Y', SparkF.bround('CENTROID_Y', 2))

    return centroids_df

  def preprocessSoil(self, soil_df):
    # SM_WC = water holding capacity
    soil_df = soil_df.withColumn('SM_WHC', SparkF.bround(soil_df.SM_FC - soil_df.SM_WP, 2))
    soil_df = soil_df.select(['IDREGION', 'SM_WHC'])

    return soil_df

  def preprocessAreaFractions(self, af_df, crop_id):
    """Filter area fractions data by crop id"""
    af_df = af_df.withColumn("FYEAR", af_df["FYEAR"].cast(SparkT.IntegerType()))
    af_df = af_df.filter(af_df["CROP_ID"] == crop_id).drop('CROP_ID')

    return af_df

  def preprocessCropArea(self, area_df, crop_id):
    """Filter area fractions data by crop id"""
    area_df = area_df.withColumn("FYEAR", area_df["FYEAR"].cast(SparkT.IntegerType()))
    area_df = area_df.filter(area_df["CROP_ID"] == crop_id).drop('CROP_ID')
    area_df = area_df.filter(area_df["CROP_AREA"].isNotNull())
    area_df = area_df.drop('FRACTION')

    return area_df

  def preprocessGAES(self, gaes_df, crop_id):
    """Select irrigated crop area by crop id"""
    sel_cols = [ c for c in gaes_df.columns if 'IRRIG' not in c]
    sel_cols += ['IRRIG_AREA_ALL', 'IRRIG_AREA' + str(crop_id)]

    return gaes_df.select(sel_cols)

  def removeDuplicateYieldData(self, yield_df, short_seq_len=2, long_seq_len=5,
                               max_short_seqs=1, max_long_seqs=0):
    """
    Find and remove duplicate sequences of yield values.
    Missing values are replaced with 0.0. So missing values are also handled.
    Using some ideas from
    https://stackoverflow.com/questions/51291226/finding-length-of-continuous-ones-in-list-in-a-pyspark-column
    """
    w = Window.partitionBy('IDREGION').orderBy('FYEAR')
    # check if value changes from one year to next
    yield_df = yield_df.select('*',
                               (yield_df['YIELD'] != SparkF.lag(yield_df['YIELD'], default=0)\
                                .over(w)).cast("int").alias("YIELD_CHANGE"),
                               (yield_df['FYEAR'] - SparkF.lag(yield_df['FYEAR'], default=0)\
                                .over(w)).cast("int").alias("FYEAR_CHANGE"))
    # group set of years with the same value
    yield_df = yield_df.select('*',
                               SparkF.sum(SparkF.col("YIELD_CHANGE"))\
                               .over(w.rangeBetween(Window.unboundedPreceding, 0)).alias("YIELD_GROUP"))

    w2 = Window.partitionBy(['IDREGION', 'YIELD_GROUP'])
    # compute the start, end and length of duplicate sequence
    yield_df = yield_df.select('*',
                               SparkF.min("FYEAR").over(w2).alias("SEQ_START"),
                               SparkF.max("FYEAR").over(w2).alias("SEQ_END"),
                               SparkF.count("*").over(w2).alias("SEQ_LEN"))

    w3 = Window.partitionBy('IDREGION')
    # compute max year
    yield_df = yield_df.select('*',
                               SparkF.min("FYEAR").over(w3).alias("MIN_FYEAR"),
                               SparkF.max("FYEAR").over(w3).alias("MAX_FYEAR"))
    # For sequences ending in max year, we remove data points only.
    # So such sequences are not counted here.
    yield_df = yield_df.select('*',
                               # count number of short sequences except those ending at max(FYEAR)
                               SparkF.sum(SparkF.when((yield_df['SEQ_LEN'] > short_seq_len) &
                                                      # (yield_df['SEQ_END'] != yield_df['MAX_FYEAR']) &
                                                      (yield_df['FYEAR'] == yield_df['SEQ_END']), 1)\
                                          .otherwise(0)).over(w3).alias('COUNT_SHORT_SEQ'),
                               # count number of long sequences except those ending at max(FYEAR)
                               SparkF.sum(SparkF.when((yield_df['SEQ_LEN'] > long_seq_len) &
                                                      (yield_df['SEQ_END'] != yield_df['MAX_FYEAR']) &
                                                      (yield_df['FYEAR'] == yield_df['SEQ_END']), 1)\
                                          .otherwise(0)).over(w3).alias('COUNT_LONG_SEQ'),
                               # count missing years
                               SparkF.sum(SparkF.when((yield_df['FYEAR'] != yield_df['MIN_FYEAR']) &
                                                       (yield_df['FYEAR_CHANGE'] > short_seq_len), 1)\
                                          .otherwise(0)).over(w3).alias('COUNT_SHORT_GAPS'),
                               SparkF.sum(SparkF.when((yield_df['FYEAR'] != yield_df['MIN_FYEAR']) &
                                                        (yield_df['FYEAR_CHANGE'] > long_seq_len), 1)\
                                          .otherwise(0)).over(w3).alias('COUNT_LONG_GAPS'))

    if (self.verbose > 2):
      print('Data with duplicate sequences')
      yield_df.filter(yield_df['COUNT_SHORT_SEQ'] > max_short_seqs).show()
      yield_df.filter(yield_df['COUNT_LONG_SEQ'] > max_long_seqs).show()

    # remove regions with many short sequences
    yield_df = yield_df.filter(yield_df['COUNT_SHORT_SEQ'] <= max_short_seqs)
    yield_df = yield_df.filter(yield_df['COUNT_SHORT_GAPS'] <= max_short_seqs)

    # remove regions with long sequences
    yield_df = yield_df.filter(yield_df['COUNT_LONG_SEQ'] <= max_long_seqs)
    yield_df = yield_df.filter(yield_df['COUNT_LONG_GAPS'] <= max_long_seqs)

    # remove data points, except SEQ_START, for remaining sequences
    yield_df = yield_df.filter((yield_df['FYEAR'] == yield_df['SEQ_START']) |
                               (yield_df['SEQ_LEN'] <= short_seq_len))

    return yield_df.select(['IDREGION', 'FYEAR', 'YIELD'])

  def preprocessYield(self, yield_df, crop_id, clean_data=False):
    """
    Yield preprocessing depends on the data format.
    Here we cover preprocessing for France (NUTS3), Germany (NUTS3) and the Netherlands (NUTS2).
    """
    # Delete trailing empty columns
    empty_cols = [ c for c in yield_df.columns if c.startswith('_c') ]
    for c in empty_cols:
      yield_df = yield_df.drop(c)

    # Special case for Netherlands and Germany: convert yield columns into rows
    years = [int(c) for c in yield_df.columns if c[0].isdigit()]
    if (len(years) > 0):
      yield_by_year = yield_df.rdd.map(lambda x: (x[0], cropNameToID(crop_id_dict, x[0]), x[1],
                                                  [(years[i], x[i+2]) for i in range(len(years))]))

      yield_df = yield_by_year.toDF(['CROP', 'CROP_ID', 'IDREGION', 'YIELD'])
      yield_df = yield_df.withColumn('YR_YIELD', SparkF.explode('YIELD')).drop('YIELD')
      yield_by_year = yield_df.rdd.map(lambda x: (x[0], x[1], x[2], x[3][0], x[3][1]))
      yield_df = yield_by_year.toDF(['CROP', 'CROP_ID', 'IDREGION', 'FYEAR', 'YIELD'])
    else:
      yield_by_year = yield_df.rdd.map(lambda x: (x[0], cropNameToID(crop_id_dict, x[0]), x[1], x[2], x[3]))
      yield_df = yield_by_year.toDF(['CROP', 'CROP_ID', 'IDREGION', 'FYEAR', 'YIELD'])

    yield_df = yield_df.filter(yield_df.CROP_ID == crop_id).drop('CROP', 'CROP_ID')
    if (yield_df.count() == 0):
      return None

    yield_df = yield_df.filter(yield_df.YIELD.isNotNull())
    yield_df = yield_df.withColumn("YIELD", yield_df["YIELD"].cast(SparkT.FloatType()))
    if (clean_data):
      yield_df = self.removeDuplicateYieldData(yield_df)

    yield_df = yield_df.filter(yield_df['YIELD'] > 0.0)

    return yield_df

  def preprocessYieldMCYFS(self, mcyfs_df, crop_id):
    """Preprocess MCYFS NUTS0 level yield predictions"""
    # the input columns are IDREGION, CROP, PREDICTION_DATE, FYEAR, YIELD_PRED
    mcyfs_df = mcyfs_df.withColumn('PRED_DEKAD', getDekad('PREDICTION_DATE'))
    # the columns should now be IDREGION, CROP, PREDICTION_DATE, FYEAR, YIELD_PRED, PRED_DEKAD
    yield_by_year = mcyfs_df.rdd.map(lambda x: (x[1], cropNameToID(crop_id_dict, x[1]),
                                                x[0], x[3], x[2], x[4], x[5]))
    mcyfs_df = yield_by_year.toDF(['CROP', 'CROP_ID', 'IDREGION', 'FYEAR',
                                         'PRED_DATE', 'YIELD_PRED', 'PRED_DEKAD'])
    mcyfs_df = mcyfs_df.filter(mcyfs_df.CROP_ID == crop_id).drop('CROP', 'CROP_ID')
    if (mcyfs_df.count() == 0):
      return None

    mcyfs_df = mcyfs_df.withColumn("YIELD_PRED", mcyfs_df["YIELD_PRED"].cast(SparkT.FloatType()))

    return mcyfs_df
