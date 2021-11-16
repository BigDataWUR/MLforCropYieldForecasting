import numpy as np

from ..common import globals

if (globals.test_env == 'pkg'):
  crop_name_dict = globals.crop_name_dict
  crop_id_dict = globals.crop_id_dict

  from ..common.util import getYear, getDekad, getMonth, getDay
  from ..common.util import cropIDToName, cropNameToID
  from ..common.util import printInGroups, plotTrend, plotTrueVSPredicted 

class TestUtil():
  def __init__(self, spark):
    self.good_date = spark.createDataFrame([(1, '19940102'),
                                          (2, '15831224')],
                                         ['ID', 'DATE'])
    self.bad_date = spark.createDataFrame([(1, '14341224'),
                                           (2, '12345678'),
                                          (3, '123-12-24')],
                                         ['ID', 'DATE'])

  def testDateFormat(self):
    print('\n Test Date Format')
    self.good_date = self.good_date.withColumn('FYEAR', getYear('DATE'))
    self.good_date.show()
    self.bad_date = self.bad_date.withColumn('FYEAR', getYear('DATE'))
    self.bad_date.show()
    assert (self.bad_date.filter(self.bad_date.FYEAR.isNull()).count() == 2)
    self.bad_date = self.bad_date.withColumn('MONTH', getMonth('DATE'))
    self.bad_date.show()
    assert (self.bad_date.filter(self.bad_date.MONTH.isNull()).count() == 2)
    self.bad_date = self.bad_date.withColumn('DAY', getDay('DATE'))
    # check the day here for first date, it's incorrect
    # seems to be a Spark issue
    self.bad_date.show()
    assert (self.bad_date.filter(self.bad_date.DAY.isNull()).count() == 2)
    self.bad_date = self.bad_date.withColumn('DEKAD', getDekad('DATE'))
    self.bad_date.show()
    assert (self.bad_date.filter(self.bad_date.DEKAD.isNull()).count() == 2)

  def testGetYear(self):
    print('\n Test getYear')
    self.good_date = self.good_date.withColumn('FYEAR', getYear('DATE'))
    self.good_date.show()
    year1 = self.good_date.filter(self.good_date.ID == 1).select('FYEAR').collect()[0][0]
    assert year1 == 1994
    year2 = self.good_date.filter(self.good_date.ID == 2).select('FYEAR').collect()[0][0]
    assert year2 == 1583

  def testGetMonth(self):
    print('\n Test getMonth')
    self.good_date = self.good_date.withColumn('MONTH', getMonth('DATE'))
    self.good_date.show()
    month1 = self.good_date.filter(self.good_date.ID == 1).select('MONTH').collect()[0][0]
    assert month1 == 1
    month2 = self.good_date.filter(self.good_date.ID == 2).select('MONTH').collect()[0][0]
    assert month2 == 12

  def testGetDay(self):
    print('\n Test getDay')
    self.good_date = self.good_date.withColumn('DAY', getDay('DATE'))
    self.good_date.show()
    day1 = self.good_date.filter(self.good_date.ID == 1).select('DAY').collect()[0][0]
    assert day1 == 2
    day2 = self.good_date.filter(self.good_date.ID == 2).select('DAY').collect()[0][0]
    assert day2 == 24

  def testGetDekad(self):
    print('\n Test getDekad')
    self.good_date = self.good_date.withColumn('DEKAD', getDekad('DATE'))
    self.good_date.show()
    dekad1 = self.good_date.filter(self.good_date.ID == 1).select('DEKAD').collect()[0][0]
    assert dekad1 == 1
    dekad2 = self.good_date.filter(self.good_date.ID == 2).select('DEKAD').collect()[0][0]
    assert dekad2 == 36

  def testCropIDToName(self):
    print('\n Test cropIDToName')
    crop_name = cropIDToName(crop_name_dict, 6)
    print(6, ':' + crop_name)
    assert crop_name == 'sugarbeet'
    crop_name = cropIDToName(crop_name_dict, 8)
    print(8, ':' + crop_name)
    assert crop_name == 'NA'

  def testCropNameToID(self):
    print('\n Test cropNameToID')
    crop_id = cropNameToID(crop_id_dict, 'Potatoes')
    print('Potatoes:', crop_id)
    assert crop_id == 7
    crop_id = cropNameToID(crop_id_dict, 'Soybean')
    print('Soybean:', crop_id)
    assert crop_id == 0

  def testPrintInGroups(self):
    print('\n Test printInGroups')
    features = ['feat' + str(i+1) for i in range(15)]
    num_features = len(features)
    num_half = np.cast['int64'](np.floor(num_features/2))
    indices1 = [ i for i in range(num_features)]
    indices2 = [ 2*i for i in range(num_half)]
    indices3 = [ (2*i + 1) for i in range(num_half)]

    printInGroups(features, indices1)
    printInGroups(features, indices2)
    printInGroups(features, indices3)

  def testPlotTrend(self):
    print('\n Test plotTrend')
    years = [yr for yr in range(2000, 2010)]
    trend_values = [ (i + 1) for i in range(50, 60)]
    actual_values = []
    for tval in trend_values:
      if (tval % 2) == 0:
        actual_values.append(tval + 0.5)
      else:
        actual_values.append(tval - 0.5)

    plotTrend(years, actual_values, trend_values, 'YIELD')

  def testPlotTrueVSPredicted(self):
    print('\n Test plotTrueVSPredicted')
    Y_true = [ (i + 1) for i in range(50, 60)]
    Y_predicted = []
    for tval in Y_true:
      if (tval % 2) == 0:
        Y_predicted.append(tval + 0.5)
      else:
        Y_predicted.append(tval - 0.5)

    Y_true = np.asarray(Y_true)
    Y_predicted = np.asarray(Y_predicted)

    plotTrueVSPredicted(Y_true, Y_predicted)

  def runAllTests(self):
    print('\nTest Utility Functions BEGIN\n')
    self.testDateFormat()
    self.testGetYear()
    self.testGetMonth()
    self.testGetDay()
    self.testGetDekad()
    self.testCropIDToName()
    self.testCropNameToID()
    self.testPrintInGroups()
    self.testPlotTrend()
    self.testPlotTrueVSPredicted()
    print('\nTest Utility Functions END\n')
