import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from sklearn.utils import parallel_backend

from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.util import printInGroups
  from ..common.util import modelRefitMeanVariance

class CYPFeatureSelector:
  def __init__(self, cyp_config, X_train, Y_train, all_features,
               custom_cv, train_weights=None):
    self.X_train = X_train
    self.Y_train = Y_train
    self.train_weights = train_weights
    self.custom_cv = custom_cv
    self.all_features = all_features
    self.feature_selectors = cyp_config.getFeatureSelectors(len(all_features))
    self.scaler = cyp_config.getFeatureScaler()
    self.cv_metric = cyp_config.getFeatureSelectionCVMetric()
    self.use_sample_weights = cyp_config.useSampleWeights()
    self.verbose = cyp_config.getDebugLevel()

  def setCustomCV(self, custom_cv):
    """Set custom K-Fold validation splits"""
    self.custom_cv = custom_cv

  def setFeatures(self, features):
    """Set the list of all features"""
    self.all_features = all_features

  # K-fold validation to find the optimal number of features
  # and optimal hyperparameters for estimator.
  def featureSelectionParameterSearch(self, sel_name, selector, est, param_space, fit_params):
    """Use grid search with k-fold validation to optimize the number of features"""
    # 3 values per parameter
    # nparams_sampled = pow(3, len(param_space))
    # if (nparams_sampled < 100):
    #  nparams_sampled = 100

    nparams_sampled = 40 + pow(2, len(param_space))
    X_train_copy = np.copy(self.X_train)
    pipeline = Pipeline([("scaler", self.scaler),
                         ("selector", selector),
                         ("estimator", est)])

    bayes_search = BayesSearchCV(estimator=pipeline,
                                 search_spaces=param_space,
                                 n_iter=nparams_sampled,
                                 scoring=self.cv_metric,
                                 cv=self.custom_cv,
                                 n_points=4,
                                 refit=modelRefitMeanVariance,
                                 return_train_score=True,
                                 n_jobs=-1)

    # with parallel_backend('spark', n_jobs=-1):
    bayes_search.fit(X_train_copy, np.ravel(self.Y_train), **fit_params)

    best_params = bayes_search.best_params_
    if (self.verbose > 2):
      print('\nFeature selection using', sel_name)
      plotCVResults(bayes_search)

    best_estimator = bayes_search.best_estimator_
    with parallel_backend('spark', n_jobs=-1):
      cv_scores = cross_val_score(best_estimator, X_train_copy, self.Y_train,
                                  cv=self.custom_cv, scoring=self.cv_metric)
    indices = []
    # feature selectors should have 'get_support' function
    selector = bayes_search.best_estimator_.named_steps['selector']
    if ((isinstance(selector, SelectFromModel)) or (isinstance(selector, SelectKBest)) or
        (isinstance(selector, RFE))):
      indices = selector.get_support(indices=True)

    if (self.verbose > 2):
      print('\nSelected Features:')
      print('-------------------')
      printInGroups(self.all_features, indices)

    result = {
        'indices' : indices,
        'cv_scores' : cv_scores,
        'estimator' : best_estimator,
        'best_params' : best_params,
    }

    return result

  # Compare different feature selectors using cross validation score.
  # Also compare combined features with the best individual feature selector.
  def compareFeatureSelectors(self, est_name, est, est_param_space):
    """Compare feature selectors based on K-fold validation scores"""
    fs_results = {}
    combined_indices = []
    for sel_name in self.feature_selectors:
      selector = self.feature_selectors[sel_name]['selector']
      sel_param_space = self.feature_selectors[sel_name]['param_space']
      param_space = sel_param_space.copy()
      param_space.update(est_param_space)

      fit_params = {}
      if (self.use_sample_weights and (est_name != 'KNN')):
        fit_params['estimator__sample_weight'] = self.train_weights

      result = self.featureSelectionParameterSearch(sel_name, selector, est,
                                                    param_space, fit_params)
      param_space.clear()

      fs_results[sel_name] = {
          'indices' : result['indices'],
          'mean_score' : np.mean(result['cv_scores']),
          'std_score' : np.std(result['cv_scores']),
          'estimator' : result['estimator'],
          'best_params' : result['best_params']
      }

    # best selector has the highest score based on mean_score and std_score
    # we subtract std_score because lower variance is better.
    sel_names = list(fs_results.keys())
    mean_scores = np.array([fs_results[s]['mean_score'] for s in sel_names])
    std_scores = np.array([fs_results[s]['std_score'] for s in sel_names])
    sel_scores = mean_scores - std_scores
    best_score, best_index = max((val, idx) for (idx, val) in enumerate(sel_scores))
    best_sel_name = sel_names[best_index]
    for i in range(len(sel_names)):
      s = sel_names[i]
      fs_results[s]['sel_score'] = sel_scores[i]

    return fs_results, best_sel_name

  # Between the optimal sets for each estimator select the set with the higher score.
  def selectOptimalFeatures(self, est, est_name, est_param_space, log_fh):
    """
    Select optimal features by comparing individual feature selectors
    and combined features
    """
    X_train_selected = None
    # set it to a large negative value
    fs_summary = {}
    row_count = 1

    est_info = '\nEstimator: ' + est_name
    est_info += '\n---------------------------'
    log_fh.write(est_info)
    print(est_info)

    fs_results, best_sel = self.compareFeatureSelectors(est_name, est, est_param_space)

    # result includes
    # - 'best_selector' : name of the best selector
    # - 'best_indices' : indices of features selected
    # - 'fs_results' : dict with 'indices', 'mean_score' and 'std_score' for all selectors

    for sel_name in fs_results:
      mean_score = np.round(fs_results[sel_name]['mean_score'], 2)
      std_score = np.round(fs_results[sel_name]['std_score'], 2)
      sel_score = np.round(fs_results[sel_name]['sel_score'], 2)
      est_sel_row = [est_name, sel_name, mean_score, std_score, sel_score]
      fs_summary['row' + str(row_count)] = est_sel_row
      row_count += 1

    selected_indices = fs_results[best_sel]['indices']
    best_estimator = fs_results[best_sel]['estimator']
    fs_df_columns = ['estimator', 'selector', 'mean score', 'std score', 'selector score']
    fs_df = pd.DataFrame.from_dict(fs_summary, orient='index', columns=fs_df_columns)
    best_params = fs_results[best_sel]['best_params']
    best_params_info = '\nbest parameters:'
    for c in best_params:
      best_params_info += '\n' + c + '=' + str(best_params[c])

    log_fh.write(best_params_info)
    print(best_params_info)

    ftsel_summary_info = '\nBest selector: ' + best_sel
    ftsel_summary_info += '\nFeature Selection Summary'
    ftsel_summary_info += '\n---------------------------'
    ftsel_summary_info += '\n' + fs_df.to_string(index=False) + '\n'
    log_fh.write(ftsel_summary_info)
    print(ftsel_summary_info)

    return selected_indices, best_estimator
