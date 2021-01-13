import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import printFeatures

class CYPFeatureSelector:
  def __init__(self, cyp_config, X_train, Y_train, custom_cv, all_features):
    self.X_train = X_train
    self.Y_train = Y_train
    self.custom_cv = custom_cv
    self.all_features = all_features
    self.scaler = cyp_config.getFeatureScaler()
    self.cv_metric = cyp_config.getFeatureSelectionCVMetric()
    self.verbose = cyp_config.getDebugLevel()

  def setCustomCV(self, custom_cv):
    """Set custom K-Fold validation splits"""
    self.custom_cv = custom_cv

  def setFeatures(self, features):
    """Set the list of all features"""
    self.all_features = all_features

  # K-fold validation to find the optimal number of features
  # and optimal hyperparameters for estimator.
  def featureSelectionGridSearch(self, selector, est, param_grid):
    """Use grid search with k-fold validation to optimize the number of features"""
    X_train_copy = np.copy(self.X_train)
    pipeline = Pipeline([("scaler", self.scaler),
                         ("selector", selector),
                         ("estimator", est)])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                               scoring=self.cv_metric, cv=self.custom_cv)
    with parallel_backend('spark', n_jobs=-1):
      grid_search.fit(X_train_copy, np.ravel(self.Y_train))

    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_
    indices = []
    # feature selectors should have 'get_support' function
    if ((isinstance(selector, SelectFromModel)) or (isinstance(selector, SelectKBest)) or
        (isinstance(selector, RFE))):
      sel = grid_search.best_estimator_.named_steps['selector']
      indices = sel.get_support(indices=True)

    result = {
        'indices' : indices,
        'best_score' : best_score,
    }

    return result

  # Compare different feature selectors using cross validation score.
  # Also compare combined features with the best individual feature selector.
  def compareFeatureSelectors(self, feature_selectors,
                              est, est_param_grid):
    """Compare feature selectors based on K-fold validation scores"""
    combined_indices = []
    # set it to a large negative value
    best_score = -1000
    best_indices = []
    best_selector = ''
    fs_scores = {}
    for sel_name in feature_selectors:
      selector = feature_selectors[sel_name]['selector']
      sel_param_grid = feature_selectors[sel_name]['param_grid']
      param_grid = sel_param_grid.copy()
      param_grid.update(est_param_grid)
      result = self.featureSelectionGridSearch(selector, est, param_grid)
      param_grid.clear()

      if (self.verbose > 2):
        print('\nFeature selection using', sel_name)
        print('Best cross-validation', self.cv_metric + ':',
              np.round(result['best_score'], 3))

        print('\nSelected Features:')
        print('-------------------')
        printFeatures(self.all_features, result['indices'])

      combined_indices = list(set(combined_indices) | set(result['indices']))

      fs_scores[sel_name] = result['best_score']

      if (result['best_score'] > best_score):
        best_indices = result['indices']
        best_score = result['best_score']
        best_selector = sel_name

    result = {
        'best_selector' : best_selector,
        'best_score' : best_score,
        'best_indices' : best_indices,
        'combined_indices' : combined_indices,
        'scores' : fs_scores
    }

    return result

  # Between the optimal sets for each estimator select the set with the higher score.
  def selectOptimalFeatures(self, feature_selectors,
                            est, est_name, est_param_grid, log_fh):
    """
    Select optimal features by comparing individual feature selectors
    and combined features
    """
    X_train_selected = None
    # set it to a large negative value
    best_score = -1000
    fs_summary = {}
    row_count = 1

    est_info = '\nEstimator: ' + est_name
    est_info += '\n---------------------------'
    log_fh.write(est_info)
    print(est_info)

    result = self.compareFeatureSelectors(feature_selectors,
                                          est, est_param_grid)

    # result includes
    # - 'best_selector' : name of the best selector
    # - 'best_score' : cv score of features selected with estimator
    # - 'best_indices' : indices of features selected
    # - 'best_estimator' : estimator with settings that gave best score
    # - 'combined_indices' : indices of combined features
    # - 'scores' : dict with scores of all feature selectors

    for sel_name in result['scores']:
      est_sel_row = [est_name, sel_name, np.round(result['scores'][sel_name], 2)]
      fs_summary['row' + str(row_count)] = est_sel_row
      row_count += 1

    # calculate cross-validation score for combined features
    X_train_selected = self.X_train[:, result['combined_indices']]
    pipeline = Pipeline([("scaler", self.scaler), ("estimator", est)])
    grid_search = GridSearchCV(estimator=pipeline, param_grid=est_param_grid,
                               scoring=self.cv_metric, cv=self.custom_cv)
    X_train_selected_copy = np.copy(X_train_selected)
    with parallel_backend('spark', n_jobs=-1):
      grid_search.fit(X_train_selected_copy, np.ravel(self.Y_train))

    combined_score = grid_search.best_score_
    combo_sel_row = [est_name, 'combined', np.round(combined_score, 2)]

    fs_summary['row' + str(row_count)] = combo_sel_row
    row_count += 1

    # We check if combined features give us a better score
    # than the best feature selection method
    if (combined_score < result['best_score']):
      selector = result['best_selector']
      selected_indices = result['best_indices']
    else:
      selected_indices = result['combined_indices']

    fs_df_columns = ['estimator', 'selector', self.cv_metric]
    fs_df = pd.DataFrame.from_dict(fs_summary, orient='index', columns=fs_df_columns)

    ftsel_summary_info = '\nFeature Selection Summary'
    ftsel_summary_info += '\n---------------------------'
    ftsel_summary_info += '\n' + fs_df.to_string(index=False) + '\n'
    log_fh.write(ftsel_summary_info)
    print(ftsel_summary_info)

    return selected_indices
