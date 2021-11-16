import numpy as np

class CYPTrainTestSplitter:
  def __init__(self, cyp_config):
    self.use_yield_trend = cyp_config.useYieldTrend()
    self.test_fraction = cyp_config.getTestFraction()
    self.verbose = cyp_config.getDebugLevel()

  def getTestYears(self, all_years, test_fraction=None, use_yield_trend=None):
    num_years = len(all_years)
    test_years = []
    if (test_fraction is None):
      test_fraction = self.test_fraction

    if (use_yield_trend is None):
      use_yield_trend = self.use_yield_trend

    if (use_yield_trend):
      # If test_year_start 15, years with index >= 15 are added to the test set
      test_year_start = num_years - np.floor(num_years * test_fraction).astype('int')
      test_years = all_years[test_year_start:]
    else:
      # If test_year_pos = 5, every 5th year is added to test set.
      # indices start with 0, so test_year_pos'th year has index (test_year_pos - 1)
      test_year_pos = np.floor(1/test_fraction).astype('int')
      test_years = all_years[test_year_pos - 1::test_year_pos]

    return test_years

  def trainTestSplit(self, yield_df, test_fraction=None, use_yield_trend=None):
    all_years = sorted([yr[0] for yr in yield_df.select('FYEAR').distinct().collect()])
    test_years = self.getTestYears(all_years, test_fraction, use_yield_trend)

    return test_years

  # Returns an array containings tuples (train_idxs, test_idxs) for each fold
  # NOTE Y_train should include IDREGION, FYEAR as first two columns.
  def customKFoldValidationSplit(self, Y_train_full, num_folds, log_fh=None):
    """
    Custom K-fold Validation Splits:
    When using yield trend, we cannot do k-fold cross-validation. The custom
    K-Fold validation splits data in time-ordered fashion. The test data
    always comes after the training data.
    """
    all_years = sorted(np.unique(Y_train_full[:, 1]))
    num_years = len(all_years)
    num_test_years = 1
    num_train_years = num_years - (num_test_years * num_folds)

    custom_cv = []
    custom_split_info = '\nCustom sliding validation train, test splits'
    custom_split_info += '\n----------------------------------------------'

    cv_test_years = []
    for k in range(num_folds):
      test_years_start = num_train_years + (k * num_test_years)
      train_years = all_years[:test_years_start]
      test_years = all_years[test_years_start:test_years_start + num_test_years]
      cv_test_years += test_years
      test_indexes = np.ravel(np.nonzero(np.isin(Y_train_full[:, 1], test_years)))
      train_indexes = np.ravel(np.nonzero(np.isin(Y_train_full[:, 1], train_years)))
      custom_cv.append(tuple((train_indexes, test_indexes)))

      train_years = [str(y) for y in train_years]
      test_years = [str(y) for y in test_years]
      custom_split_info += '\nValidation set ' + str(k + 1) + ' training years: ' + ', '.join(train_years)
      custom_split_info += '\nValidation set ' + str(k + 1) + ' test years: ' + ', '.join(test_years)

    custom_split_info += '\n'
    if (log_fh is not None):
      log_fh.write(custom_split_info)

    if (self.verbose > 1):
      print(custom_split_info)

    return custom_cv, cv_test_years
