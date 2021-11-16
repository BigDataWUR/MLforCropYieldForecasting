import numpy as np

from ..common import globals

if (globals.test_env == 'pkg'):
  SparkT = globals.SparkT
  SparkF = globals.SparkF

def getCropCalendarPeriods(df):
  """Periods for per year crop calendar"""
  # (maximum of 4 months = 12 dekads).
  # Subtracting 11 because both ends of the period are included.
  # p0 : if CAMPAIGN_EARLY_SEASON > df.START_DVS
  #        START_DVS - 11 to START_DVS
  #      else
  #        START_DVS - 11 to CAMPAIGN_EARLY_SEASON
  p0_filter = SparkF.when(df.CAMPAIGN_EARLY_SEASON > df.START_DVS,
                          (df.CAMPAIGN_DEKAD >= (df.START_DVS - 11)) &
                          (df.CAMPAIGN_DEKAD <= df.START_DVS))\
                          .otherwise((df.CAMPAIGN_DEKAD >= (df.START_DVS - 11)) &
                                     (df.CAMPAIGN_DEKAD <= df.CAMPAIGN_EARLY_SEASON))
  # p1 : if CAMPAIGN_EARLY_SEASON > (df.START_DVS + 1)
  #        (START_DVS - 1) to (START_DVS + 1)
  #      else
  #        (START_DVS - 1) to CAMPAIGN_EARLY_SEASON
  p1_filter = SparkF.when(df.CAMPAIGN_EARLY_SEASON > (df.START_DVS + 1),
                          (df.CAMPAIGN_DEKAD >= (df.START_DVS - 1)) &
                          (df.CAMPAIGN_DEKAD <= (df.START_DVS + 1)))\
                          .otherwise((df.CAMPAIGN_DEKAD >= (df.START_DVS - 1)) &
                                     (df.CAMPAIGN_DEKAD <= df.CAMPAIGN_EARLY_SEASON))
  # p2 : if CAMPAIGN_EARLY_SEASON > df.START_DVS1
  #        START_DVS to START_DVS1
  #      else
  #        START_DVS to CAMPAIGN_EARLY_SEASON
  p2_filter = SparkF.when(df.CAMPAIGN_EARLY_SEASON > df.START_DVS1,
                          (df.CAMPAIGN_DEKAD >= df.START_DVS) &
                          (df.CAMPAIGN_DEKAD <= df.START_DVS1))\
                          .otherwise((df.CAMPAIGN_DEKAD >= df.START_DVS) &
                                     (df.CAMPAIGN_DEKAD <= df.CAMPAIGN_EARLY_SEASON))
  # p3 : if CAMPAIGN_EARLY_SEASON > (df.START_DVS1 + 1)
  #        (START_DVS1 - 1) to (START_DVS1 + 1)
  #      else
  #        (START_DVS1 - 1) to CAMPAIGN_EARLY_SEASON
  p3_filter = SparkF.when(df.CAMPAIGN_EARLY_SEASON > (df.START_DVS1 + 1),
                          (df.CAMPAIGN_DEKAD >= (df.START_DVS1 - 1)) &
                          (df.CAMPAIGN_DEKAD <= (df.START_DVS1 + 1)))\
                          .otherwise((df.CAMPAIGN_DEKAD >= (df.START_DVS1 - 1)) &
                                     (df.CAMPAIGN_DEKAD <= df.CAMPAIGN_EARLY_SEASON))
  # p4 : if CAMPAIGN_EARLY_SEASON > df.START_DVS2
  #        START_DVS1 to START_DVS2
  #      else
  #        START_DVS1 to CAMPAIGN_EARLY_SEASON
  p4_filter = SparkF.when(df.CAMPAIGN_EARLY_SEASON > df.START_DVS2,
                          (df.CAMPAIGN_DEKAD >= df.START_DVS1) &
                          (df.CAMPAIGN_DEKAD <= df.START_DVS2))\
                          .otherwise((df.CAMPAIGN_DEKAD >= df.START_DVS1) &
                                     (df.CAMPAIGN_DEKAD <= df.CAMPAIGN_EARLY_SEASON))
  # p5 : if CAMPAIGN_EARLY_SEASON > (df.START_DVS2 + 1)
  #        (START_DVS2 - 1) to (START_DVS2 + 1)
  #      else
  #        (START_DVS2 - 1) to CAMPAIGN_EARLY_SEASON
  p5_filter = SparkF.when(df.CAMPAIGN_EARLY_SEASON > (df.START_DVS2 + 1),
                          (df.CAMPAIGN_DEKAD >= (df.START_DVS2 - 1)) &
                          (df.CAMPAIGN_DEKAD <= (df.START_DVS2 + 1)))\
                          .otherwise((df.CAMPAIGN_DEKAD >= (df.START_DVS2 - 1)) &
                                     (df.CAMPAIGN_DEKAD <= df.CAMPAIGN_EARLY_SEASON))

  cc_periods = {
      'p0' : p0_filter,
      'p1' : p1_filter,
      'p2' : p2_filter,
      'p3' : p3_filter,
      'p4' : p4_filter,
      'p5' : p5_filter,
  }

  return cc_periods

def getSeasonStartFilter(df):
  """Filter for dekads after season start"""
  return df.CAMPAIGN_DEKAD >= (df.START_DVS - 11)

def getCountryCropCalendar(crop_cal):
  """Take averages to make the crop calendar per country"""
  crop_cal = crop_cal.withColumn('COUNTRY', SparkF.substring('IDREGION', 1, 2))
  aggrs = [ SparkF.bround(SparkF.avg(crop_cal['START_DVS'])).alias('START_DVS'),
            SparkF.bround(SparkF.avg(crop_cal['START_DVS1'])).alias('START_DVS1'),
            SparkF.bround(SparkF.avg(crop_cal['START_DVS2'])).alias('START_DVS2'),
            SparkF.bround(SparkF.avg(crop_cal['CAMPAIGN_EARLY_SEASON'])).alias('CAMPAIGN_EARLY_SEASON') ]

  crop_cal = crop_cal.groupBy('COUNTRY').agg(*aggrs)
  return crop_cal

def getCropCalendar(cyp_config, dvs_summary, log_fh):
  """Use DVS summary to infer the crop calendar"""
  pd_dvs_summary = dvs_summary.toPandas()
  early_season_prediction = cyp_config.earlySeasonPrediction()
  debug_level = cyp_config.getDebugLevel()

  avg_dvs_start = np.round(pd_dvs_summary['START_DVS'].mean(), 0)
  avg_dvs1_start = np.round(pd_dvs_summary['START_DVS1'].mean(), 0)
  avg_dvs2_start = np.round(pd_dvs_summary['START_DVS2'].mean(), 0)

  # We look at 6 windows
  # 0. Preplanting window (maximum of 4 months = 12 dekads).
  # Subtracting 11 because both ends of the period are included.
  p0_start = 1 if (avg_dvs_start - 11) < 1 else (avg_dvs_start - 11)
  p0_end = avg_dvs_start

  # 1. Planting window
  p1_start = avg_dvs_start - 1
  p1_end = avg_dvs_start + 1

  # 2. Vegetative phase
  p2_start = avg_dvs_start
  p2_end = avg_dvs1_start

  # 3. Flowering phase
  p3_start = avg_dvs1_start - 1
  p3_end = avg_dvs1_start + 1

  # 4. Yield formation phase
  p4_start = avg_dvs1_start
  p4_end = avg_dvs2_start

  # 5. Harvest window
  p5_start = avg_dvs2_start - 1
  p5_end = avg_dvs2_start + 1

  early_season_prediction = cyp_config.earlySeasonPrediction()
  early_season_end = 36
  if (early_season_prediction):
    early_season_end = np.round(pd_dvs_summary['CAMPAIGN_EARLY_SEASON'].mean(), 0)
    p0_end = early_season_end if (p0_end > early_season_end) else p0_end
    p1_end = early_season_end if (p1_end > early_season_end) else p1_end
    p2_end = early_season_end if (p2_end > early_season_end) else p2_end
    p3_end = early_season_end if (p3_end > early_season_end) else p3_end
    p4_end = early_season_end if (p4_end > early_season_end) else p4_end
    p5_end = early_season_end if (p5_end > early_season_end) else p5_end

  crop_cal = {}
  if (p0_end > p0_start):
    crop_cal['p0'] = { 'desc' : 'pre-planting window', 'start' : p0_start, 'end' : p0_end }
  if (p1_end > p1_start):
    crop_cal['p1'] = { 'desc' : 'planting window', 'start' : p1_start, 'end' : p1_end }
  if (p2_end > p2_start):
    crop_cal['p2'] = { 'desc' : 'vegetative phase', 'start' : p2_start, 'end' : p2_end }
  if (p3_end > p3_start):
    crop_cal['p3'] = { 'desc' : 'flowering phase', 'start' : p3_start, 'end' : p3_end }
  if (p4_end > p4_start):
    crop_cal['p4'] = { 'desc' : 'yield formation phase', 'start' : p4_start, 'end' : p4_end }
  if (p5_end > p5_start):
    crop_cal['p5'] = { 'desc' : 'harvest window', 'start' : p5_start, 'end' : p5_end }

  if (early_season_prediction):
    early_season_rel_harvest = cyp_config.getEarlySeasonEndDekad()
    early_season_info = '\nEarly Season Prediction Dekad: ' + str(early_season_rel_harvest)
    early_season_info += ', Campaign Dekad: ' + str(early_season_end)
    log_fh.write(early_season_info + '\n')
    if (debug_level > 1):
      print(early_season_info)

  crop_cal_info = '\nCrop Calendar'
  crop_cal_info += '\n-------------'
  for p in crop_cal:
    crop_cal_info += '\nPeriod ' + p + ' (' + crop_cal[p]['desc'] + '): '
    crop_cal_info += 'Campaign Dekads ' + str(crop_cal[p]['start']) + '-' + str(crop_cal[p]['end'])

  log_fh.write(crop_cal_info + '\n')
  if (debug_level > 1):
    print(crop_cal_info)

  return crop_cal
