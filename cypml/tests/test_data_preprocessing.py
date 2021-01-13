from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.config import CYPConfiguration
  from ..workflow.data_preprocessing import CYPDataPreprocessor

class TestDataPreprocessor():
  def __init__(self, spark):
    cyp_config = CYPConfiguration()
    cyp_config.setDebugLevel(2)
    self.preprocessor = CYPDataPreprocessor(spark, cyp_config)

    # create a small wofost data set
    # preprocessing currently extracts the year and dekad only
    self.wofost_df = spark.createDataFrame([(6, 'NL11', '19790110', 0.0, 0),
                                            (6, 'NL11', '19790121', 0.0, 0),
                                            (6, 'NL11', '19790331', 0.0, 50),
                                            (6, 'NL11', '19790510', 0.0, 100),
                                            (6, 'NL11', '19790821', 0.0, 150),
                                            (6, 'NL11', '19790831', 0.0, 201),
                                            (6, 'NL11', '19790910', 0.0, 201),
                                            (6, 'NL11', '19800110', 0.0, 0),
                                            (6, 'NL11', '19800121', 0.0, 0),
                                            (6, 'NL11', '19800331', 0.0, 50),
                                            (6, 'NL11', '19800610', 0.0, 100),
                                            (6, 'NL11', '19800721', 0.0, 150),
                                            (6, 'NL11', '19800831', 0.0, 200),
                                            (6, 'NL11', '19800910', 0.0, 201),
                                            (6, 'NL11', '19800921', 0.0, 201)],
                                           ['CROP_ID', 'IDREGION', 'DATE', 'POT_YB', 'DVS'])

    self.crop_season = None

    # Create a small meteo dekadal data set
    # Preprocessing currently extracts the year and dekad, and computes climate
    # water balance.
    self.meteo_dekdf = spark.createDataFrame([('NL11', '19790110', 1.2, 0.2),
                                              ('NL11', '19790121', 2.1, 0.2),
                                              ('NL11', '19790131', 0.2, 1.2),
                                              ('NL11', '19790210', 0.1, 2.0),
                                              ('NL11', '19790221', 0.0, 2.1),
                                              ('NL11', '19790228', 1.0, 0.8),
                                              ('NL11', '19790310', 1.1, 1.0),
                                              ('NL11', '19800110', 1.2, 0.2),
                                              ('NL11', '19800121', 2.1, 0.2),
                                              ('NL11', '19800131', 0.2, 1.2),
                                              ('NL11', '19800210', 0.1, 2.0),
                                              ('NL11', '19800221', 0.0, 2.1),
                                              ('NL11', '19800228', 1.0, 0.8),
                                              ('NL11', '19800310', 1.1, 1.0)],
                                             ['IDREGION', 'DATE', 'PREC', 'ET0'])

    # Create a small meteo daily data set
    # Preprocessing currently converts daily data to dekadal data by taking AVG
    # for all indicators except TMAX (MAX is used instead) and TMIN (MIN is used instead).
    query = 'select IDREGION, FYEAR, DEKAD, max(TMAX) as TMAX, min(TMIN) as TMIN, '
    query = query + ' bround(avg(TAVG), 2) as TAVG, bround(sum(PREC), 2) as PREC, '
    query = query + ' bround(sum(ET0), 2) as ET0, bround(avg(RAD), 2) as RAD, '
    query = query + ' bround(sum(CWB), 2) as CWB '
    self.meteo_daydf = spark.createDataFrame([('NL11', '19790101', 1.2, 0.2, 8.5, -1.2, 5.5, 10000.0),
                                              ('NL11', '19790102', 2.1, 0.2, 9.1, 0.3, 6.1, 12000.0),
                                              ('NL11', '19790103', 0.2, 1.2, 10.4, 1.2, 7.2, 14000.0),
                                              ('NL11', '19790104', 0.1, 2.0, 8.1, -1.5, 5.2, 10000.0),
                                              ('NL11', '19790105', 0.0, 2.1, 10.2, 1.0, 7.5, 13000.0),
                                              ('NL11', '19790106', 1.2, 0.2, 11.2, 2.5, 8.2, 16000.0),
                                              ('NL11', '19790112', 2.1, 0.5, 9.2, 0.5, 5.5, 12000.0),
                                              ('NL11', '19790113', 0.2, 1.1, 10.2, 1.4, 7.1, 14000.0),
                                              ('NL11', '19790114', 0.1, 2.0, 12.0, 3.2, 8.3, 15000.0),
                                              ('NL11', '19790115', 0.0, 1.5, 13.1, 4.5, 9.2, 17000.0),
                                              ('NL11', '19790122', 2.1, 0.5, 9.2, 0.5, 5.5, 12000.0),
                                              ('NL11', '19790123', 0.2, 1.1, 10.2, 1.4, 7.1, 14000.0),
                                              ('NL11', '19790124', 0.1, 2.0, 12.0, 3.2, 8.3, 15000.0),
                                              ('NL11', '19790125', 0.0, 1.5, 13.1, 4.5, 9.2, 17000.0),
                                              ('NL11', '19800101', 1.2, 0.2, 8.5, -1.2, 5.5, 10000.0),
                                              ('NL11', '19800102', 2.1, 0.2, 9.1, 0.3, 6.1, 12000.0),
                                              ('NL11', '19800103', 0.2, 1.2, 10.4, 1.2, 7.2, 14000.0),
                                              ('NL11', '19800104', 0.1, 2.0, 8.1, -1.5, 5.2, 10000.0),
                                              ('NL11', '19800105', 0.0, 2.1, 10.2, 1.0, 7.5, 13000.0),
                                              ('NL11', '19800106', 1.2, 0.2, 11.2, 2.5, 8.2, 16000.0),
                                              ('NL11', '19800112', 2.1, 0.5, 9.2, 0.5, 5.5, 12000.0),
                                              ('NL11', '19800113', 0.2, 1.1, 10.2, 1.4, 7.1, 14000.0),
                                              ('NL11', '19800114', 0.1, 2.0, 12.0, 3.2, 8.3, 15000.0),
                                              ('NL11', '19800115', 0.0, 1.5, 13.1, 4.5, 9.2, 17000.0),
                                              ('NL11', '19800122', 2.1, 0.5, 9.2, 0.5, 5.5, 12000.0),
                                              ('NL11', '19800123', 0.2, 1.1, 10.2, 1.4, 7.1, 14000.0),
                                              ('NL11', '19800124', 0.1, 2.0, 12.0, 3.2, 8.3, 15000.0),
                                              ('NL11', '19800125', 0.0, 1.5, 13.1, 4.5, 9.2, 17000.0)],
                                             ['IDREGION', 'DATE', 'PREC', 'ET0', 'TMAX', 'TMIN', 'TAVG', 'RAD'])

    # Create a small remote sensing data set
    # Preprocessing currently extracts the year and dekad
    self.rs_df1 = spark.createDataFrame([('NL11', '19790321', 0.47),
                                         ('NL11', '19790331', 0.49),
                                         ('NL11', '19790410', 0.55),
                                         ('NL11', '19790421', 0.49),
                                         ('NL11', '19790430', 0.64),
                                         ('NL11', '19800110', 0.42),
                                         ('NL11', '19800121', 2.43),
                                         ('NL11', '19800131', 0.41),
                                         ('NL11', '19800210', 0.42),
                                         ('NL11', '19800221', 0.44),
                                         ('NL11', '19800228', 0.45),
                                         ('NL11', '19800310', 2.43)],
                                        ['IDREGION', 'DATE', 'FAPAR'])

    self.rs_df2 = spark.createDataFrame([('FR10', '19790110', 0.42),
                                         ('FR10', '19790121', 0.43),
                                         ('FR10', '19790131', 0.41),
                                         ('FR10', '19790210', 0.42),
                                         ('FR10', '19790221', 0.44),
                                         ('FR10', '19790228', 0.45),
                                         ('FR10', '19790310', 0.47),
                                         ('FR10', '19790321', 0.49),
                                         ('FR10', '19790331', 0.55),
                                         ('FR10', '19790410', 0.62),
                                         ('FR10', '19790421', 0.66),
                                         ('FR10', '19800110', 0.42),
                                         ('FR10', '19800121', 2.43),
                                         ('FR10', '19800131', 0.41),
                                         ('FR10', '19800210', 0.42),
                                         ('FR10', '19800221', 0.44),
                                         ('FR10', '19800228', 0.45),
                                         ('FR10', '19800310', 2.43)],
                                        ['IDREGION', 'DATE', 'FAPAR'])
  
    self.crop_season_nuts3 = spark.createDataFrame([('FR101', '1979', 0, 27),
                                                    ('FR101', '1980', 27, 28),
                                                    ('FR102', '1979', 0, 27),
                                                    ('FR102', '1980', 27, 29)],
                                                   ['IDREGION', 'FYEAR', 'PREV_SEASON_END', 'SEASON_END'])

    # Create small yield data sets
    # Two formats are preprocessed: (1) year and yield are columns,
    # (2) years are columns with yield values in rows
    # Preprocessing currently converts (2) into 1
    self.yield_df1 = spark.createDataFrame([('potatoes', 'FR102', '1989', 29.75),
                                            ('potatoes', 'FR102', '1990', 25.44),
                                            ('potatoes', 'FR103', '1989', 30.2),
                                            ('potatoes', 'FR103', '1990', 29.9),
                                            ('sugarbeet', 'FR102', '1989', 66.0),
                                            ('sugarbeet', 'FR102', '1990', 55.0),
                                            ('sugarbeet', 'FR103', '1989', 69.3),
                                            ('sugarbeet', 'FR103', '1990', 59.1)],
                                           ['Crop', 'IDREGION', 'FYEAR', 'YIELD'])

    self.yield_df2 = spark.createDataFrame([('Total potatoes', 'NL11', 38.0, 40.5, 40.0),
                                            ('Total potatoes', 'NL12', 49.0, 44.0, 46.8),
                                            ('Spring barley', 'NL13', 4.6, 5.5, 6.6),
                                            ('Spring barley', 'NL12', 5.6, 6.1, 7.0)],
                                           ['Crop', 'IDREGION', '1994', '1995', '1996'])

  def testExtractYearDekad(self):
    print('WOFOST data after extracting year and dekad')
    print('-------------------------------------------')
    self.preprocessor.extractYearDekad(self.wofost_df).show(10)

  def testPreprocessWofost(self):
    print('WOFOST data after preprocessing')
    print('--------------------------------')
    self.wofost_df = self.wofost_df.filter(self.wofost_df['CROP_ID'] == 6).drop('CROP_ID')
    self.crop_season = self.preprocessor.getCropSeasonInformation(self.wofost_df,
                                                                  False)
    self.wofost_df = self.preprocessor.preprocessWofost(self.wofost_df,
                                                        self.crop_season,
                                                        False)

    self.wofost_df.show(5)
    self.crop_season.show(5)

  def testPreprocessMeteo(self):
    print('Meteo dekadal data after preprocessing')
    print('--------------------------------------')
    self.meteo_dekdf = self.preprocessor.preprocessMeteo(self.meteo_dekdf,
                                                         self.crop_season,
                                                         False)
    self.meteo_dekdf.show(5)

  def testPreprocessMeteoDaily(self):
    self.meteo_daydf = self.preprocessor.preprocessMeteo(self.meteo_daydf,
                                                         self.crop_season,
                                                         False)
    self.meteo_daydf = self.preprocessor.preprocessMeteoDaily(self.meteo_daydf)
    print('Meteo daily data after preprocessing')
    print('------------------------------------')
    self.meteo_daydf.show(5)

  def testPreprocessRemoteSensing(self):
    self.rs_df1 = self.preprocessor.preprocessRemoteSensing(self.rs_df1,
                                                            self.crop_season,
                                                            False)
    print('Remote sensing data after preprocessing')
    print('---------------------------------------')
    self.rs_df1.show(5)

  def testRemoteSensingNUTS2ToNUTS3(self):
    print('Remote sensing data before preprocessing')
    print('---------------------------------------')
    self.rs_df2.show()
    nuts3_regions = [reg[0] for reg in self.yield_df1.select('IDREGION').distinct().collect()]
    self.rs_df2 = self.preprocessor.remoteSensingNUTS2ToNUTS3(self.rs_df2, nuts3_regions)
    print('Remote sensing data at NUTS3')
    print('-----------------------------')
    self.rs_df2.show(5)

    self.rs_df2 = self.preprocessor.preprocessRemoteSensing(self.rs_df2,
                                                            self.crop_season_nuts3,
                                                            False)
    print('Remote sensing data after preprocessing')
    print('---------------------------------------')
    self.rs_df2.show(5)

  def testPreprocessYield(self):
    self.yield_df1 = self.preprocessor.preprocessYield(self.yield_df1, 7)
    print('Yield data format 1 after preprocessing')
    print('--------------------------------------')
    self.yield_df1.show(5)

    print('Yield data format 2 before preprocessing')
    print('----------------------------------------')
    self.yield_df2.show(5)

    self.yield_df2 = self.preprocessor.preprocessYield(self.yield_df2, 7)
    print('Yield data format 2 after preprocessing')
    print('----------------------------------------')
    self.yield_df2.show(5)

  def runAllTests(self):
    print('\nTest Data Preprocessor BEGIN\n')
    self.testExtractYearDekad()
    self.testPreprocessWofost()
    self.testPreprocessMeteo()
    self.testPreprocessMeteoDaily()
    self.testPreprocessRemoteSensing()
    self.testRemoteSensingNUTS2ToNUTS3()
    self.testPreprocessYield()
    print('\nTest Data Preprocessor END\n')
