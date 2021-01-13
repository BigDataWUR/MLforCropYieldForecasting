from ..common import globals

if (globals.test_env == 'pkg'):
  crop_name_dict = globals.crop_name_dict
  crop_id_dict = globals.crop_id_dict

  from ..common.util import cropNameToID, cropIDToName
  from ..common.config import CYPConfiguration
  from ..workflow.data_summary import CYPDataSummarizer

class TestDataSummarizer():
  def __init__(self, spark):
    cyp_config = CYPConfiguration()
    cyp_config.setDebugLevel(2)
    self.data_summarizer = CYPDataSummarizer(cyp_config)

    # create a small wofost data set
    self.wofost_df = spark.createDataFrame([('NL11', '1979', 14, 0.0, 0.0, 5.0),
                                            ('NL11', '1979', 15, 2.0, 1.0, 10.0),
                                            ('NL11', '1979', 16, 5.0, 4.0, 7.0),
                                            ('NL11', '1979', 17, 15.0, 12.0, 4.0),
                                            ('NL11', '1979', 18, 40.0, 35.0, 6.0),
                                            ('NL11', '1979', 19, 100.0, 80.0, 5.0),
                                            ('NL11', '1979', 20, 150.0, 120.0, 3.0),
                                            ('NL11', '1980', 13, 0.0, 0.0, 15.0),
                                            ('NL11', '1980', 14, 5.0, 2.0, 12.0),
                                            ('NL11', '1980', 15, 15.0, 12.0, 10.0),
                                            ('NL11', '1980', 16, 50.0, 40.0, 8.0),
                                            ('NL11', '1980', 17, 100.0, 80.0, 4.0),
                                            ('NL11', '1980', 18, 200.0, 140.0, 5.0),
                                            ('NL11', '1980', 19, 200.0, 150.0, 12.0),
                                            ('NL11', '1980', 20, 200.0, 150.0, 12.0)],
                                           ['IDREGION', 'FYEAR', 'DEKAD', 'POT_YB', 'WLIM_YB', 'RSM'])

    self.wofost_df2 = spark.createDataFrame([('NL11', '1979', 12, '1979', 22, 0),
                                             ('NL11', '1979', 13, '1979', 23, 1),
                                             ('NL11', '1979', 14, '1979', 24, 4),
                                             ('NL11', '1979', 15, '1979', 25, 70),
                                             ('NL11', '1979', 16, '1979', 26, 101),
                                             ('NL11', '1979', 19, '1979', 29, 150),
                                             ('NL11', '1979', 21, '1979', 31, 180),
                                             ('NL11', '1979', 23, '1979', 33, 200),
                                             ('NL11', '1979', 24, '1979', 34, 201),
                                             ('NL11', '1979', 25, '1979', 35, 201),
                                             ('NL11', '1980', 12, '1980', 22, 0),
                                             ('NL11', '1980', 13, '1980', 23, 2),
                                             ('NL11', '1980', 14, '1980', 24, 15),
                                             ('NL11', '1980', 15, '1980', 25, 80),
                                             ('NL11', '1980', 16, '1980', 26, 99),
                                             ('NL11', '1980', 19, '1980', 29, 140),
                                             ('NL11', '1980', 21, '1980', 31, 170),
                                             ('NL11', '1980', 23, '1980', 33, 195),
                                             ('NL11', '1980', 24, '1980', 34, 201),
                                             ('NL11', '1980', 25, '1980', 35, 201)],
                                           ['IDREGION', 'FYEAR', 'DEKAD', 'CAMPAIGN_YEAR', 'CAMPAIGN_DEKAD', 'DVS'])

    # Create a small meteo dekadal data set
    self.meteo_df = spark.createDataFrame([('NL11', '1979', 1, '1979', 11, 1.2, 8.5, -1.2, 5.5, 10000.0),
                                           ('NL11', '1979', 2, '1979', 12, 2.1, 9.1, 0.3, 6.1, 12000.0),
                                           ('NL11', '1979', 3, '1979', 13, 0.2, 10.4, 1.2, 7.2, 14000.0),
                                           ('NL11', '1979', 4, '1979', 14, 0.1, 8.1, -1.5, 5.2, 10000.0),
                                           ('NL11', '1979', 5, '1979', 15, 0.0, 10.2, 1.0, 7.5, 13000.0),
                                           ('NL12', '1979', 1, '1979', 12, 1.2, 11.2, 2.5, 8.2, 16000.0),
                                           ('NL12', '1979', 2, '1979', 13, 2.1, 9.2, 0.5, 5.5, 12000.0),
                                           ('NL12', '1979', 3, '1979', 14, 0.2, 10.2, 1.4, 7.1, 14000.0),
                                           ('NL12', '1979', 4, '1979', 15, 0.1, 12.0, 3.2, 8.3, 15000.0),
                                           ('NL12', '1979', 5, '1979', 16, 0.0, 13.1, 4.5, 9.2, 17000.0)],
                                          ['IDREGION', 'FYEAR', 'DEKAD', 'CAMPAIGN_YEAR', 'CAMPAIGN_DEKAD', 'PREC', 'TMAX', 'TMIN', 'TAVG', 'RAD'])

    # Create a small remote sensing data set
    # Preprocessing currently extracts the year and dekad
    self.rs_df = spark.createDataFrame([('NL11', '1979', 1, 0.42),
                                         ('NL11', '1979', 2, 0.41),
                                         ('NL11', '1979', 3, 0.42),
                                         ('NL11', '1980', 1, 0.44),
                                         ('NL11', '1980', 2, 0.45),
                                        ('NL11', '1980', 3, 0.43)],
                                        ['IDREGION', 'FYEAR', 'DEKAD', 'FAPAR'])

    # Create a small yield data set
    self.yield_df = spark.createDataFrame([(7, 'FR102', '1989', 29.75),
                                           (7, 'FR102', '1990', 25.44),
                                           (7, 'FR103', '1989', 30.2),
                                           (7, 'FR103', '1990', 29.9),
                                           (6, 'FR102', '1989', 66.0),
                                           (6, 'FR102', '1990', 55.0),
                                           (6, 'FR103', '1989', 69.3),
                                           (6, 'FR103', '1990', 59.1)],
                                          ['CROP_ID', 'IDREGION', 'FYEAR', 'YIELD'])

  def testWofostDVSSummary(self):
    print('WOFOST Crop Calendar Summary using DVS')
    print('-----------------------------')
    self.data_summarizer.wofostDVSSummary(self.wofost_df2).show()

  def testWofostIndicatorsSummary(self):
    print('WOFOST indicators summary')
    print('--------------------------')
    min_cols = ['IDREGION']
    max_cols = ['IDREGION', 'POT_YB', 'WLIM_YB']
    avg_cols = ['IDREGION', 'RSM']
    self.data_summarizer.indicatorsSummary(self.wofost_df, min_cols, max_cols, avg_cols).show()

  def testMeteoIndicatorsSummary(self):
    print('Meteo indicators summary')
    print('-------------------------')
    meteo_cols = self.meteo_df.columns[3:]
    min_cols = ['IDREGION'] + meteo_cols
    max_cols = ['IDREGION'] + meteo_cols
    avg_cols = ['IDREGION'] + meteo_cols
    self.data_summarizer.indicatorsSummary(self.meteo_df, min_cols, max_cols, avg_cols).show()

  def testRemoteSensingSummary(self):
    print('Remote sensing indicators summary')
    print('----------------------------------')
    rs_cols = ['FAPAR']
    min_cols = ['IDREGION'] + rs_cols
    max_cols = ['IDREGION'] + rs_cols
    avg_cols = ['IDREGION'] + rs_cols
    self.data_summarizer.indicatorsSummary(self.rs_df, min_cols, max_cols, avg_cols).show()

  def testYieldSummary(self):
    crop = 'potatoes'
    print('Yield summary for', crop)
    print('-----------------------------')
    crop_id = cropNameToID(crop_id_dict, crop)
    self.yield_df = self.yield_df.filter(self.yield_df.CROP_ID == crop_id)
    self.data_summarizer.yieldSummary(self.yield_df).show()

  def runAllTests(self):
    print('\nTest Data Summarizer BEGIN\n')
    self.testWofostDVSSummary()
    self.testWofostIndicatorsSummary()
    self.testMeteoIndicatorsSummary()
    self.testRemoteSensingSummary()
    self.testYieldSummary()
    print('\nTest Data Summarizer END\n')
