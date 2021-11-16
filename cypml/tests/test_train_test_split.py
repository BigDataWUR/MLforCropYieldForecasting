import numpy as np

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.config import CYPConfiguration
  from ..workflow.train_test_split import CYPTrainTestSplitter

class TestCustomTrainTestSplit:
  def __init__(self, yield_df):
    cyp_config = CYPConfiguration()
    self.verbose = 2
    cyp_config.setDebugLevel(self.verbose)
    self.yield_df = yield_df
    self.trTsSplitter = CYPTrainTestSplitter(cyp_config)

  def testCustomTrainTestSplit(self):
    print('\nTest customTrainTestSplit')
    test_fraction = 0.2
    regions = [reg[0] for reg in self.yield_df.select('IDREGION').distinct().collect()]
    num_regions = len(regions)
    test_years = self.trTsSplitter.trainTestSplit(self.yield_df, test_fraction, True)
    all_years = [yr[0] for yr in self.yield_df.select('FYEAR').distinct().collect()]
    yield_train_df = self.yield_df.filter(~self.yield_df['FYEAR'].isin(test_years))
    yield_test_df = self.yield_df.filter(self.yield_df['FYEAR'].isin(test_years))

    if(self.verbose > 1):
      print('\nCustom training, test split using yield trend')
      print('---------------------------------------------')
      print('Estimated size of test data', num_regions * np.floor(len(all_years) * test_fraction))
      print('Data Size:', yield_train_df.count(), yield_test_df.count())
      print('Test years:', test_years)

    test_years = self.trTsSplitter.trainTestSplit(self.yield_df, test_fraction, False)

    if(self.verbose > 1):
      print('\ncustom training, test split without yield trend')
      print('------------------------------------------------')
      print('Estimated size of test data', num_regions * np.floor(len(all_years) * test_fraction))
      print('Data Size:', yield_train_df.count(), yield_test_df.count())
      print('Test years:', test_years)

  def testCustomKFoldValidationSplit(self):
    print('\nTest customKFoldValidationSplit')
    test_fraction = 0.2
    num_folds = 5
    test_years = self.trTsSplitter.trainTestSplit(self.yield_df, test_fraction, 'Y')
    yield_train_df = self.yield_df.filter(~self.yield_df['FYEAR'].isin(test_years))
    yield_test_df = self.yield_df.filter(self.yield_df['FYEAR'].isin(test_years))
    yield_cols = yield_train_df.columns
    pd_yield_train_df = yield_train_df.toPandas()
    Y_train_full = pd_yield_train_df[yield_cols].values

    custom_cv, _ = self.trTsSplitter.customKFoldValidationSplit(Y_train_full, num_folds)

  def runAllTests(self):
    print('\nTest Custom Train, Test Splitter BEGIN\n')
    self.testCustomTrainTestSplit()
    self.testCustomKFoldValidationSplit()
    print('\nTest Custom Train, Test Splitter END\n')
