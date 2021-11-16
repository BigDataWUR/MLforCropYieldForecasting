from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error

from sklearn.utils.fixes import loguniform
import scipy.stats as stats
from skopt.space import Real, Categorical, Integer

from . import globals
from .util import cropNameToID, cropIDToName
from .util import meanAbsolutePercentageError

if (globals.test_env == 'pkg'):
  crop_id_dict = globals.crop_id_dict
  crop_name_dict = globals.crop_name_dict
  countries = globals.countries
  nuts_levels = globals.nuts_levels
  debug_levels = globals.debug_levels

class CYPConfiguration:
  def __init__(self, crop_name='potatoes', country_code='NL', season_cross='N'):
    self.config = {
        'crop_name' : crop_name,
        'crop_id' : cropNameToID(crop_id_dict, crop_name),
        'season_crosses_calendar_year' : season_cross,
        'country_code' : country_code,
        'nuts_level' : 'NUTS2',
        'data_sources' : [ 'WOFOST', 'METEO_DAILY', 'SOIL', 'YIELD' ],
        'clean_data' : 'N',
        'use_yield_trend' : 'N',
        'predict_yield_residuals' : 'N',
        'find_optimal_trend_window' : 'N',
        # set it to a list with one entry for fixed window
        'trend_windows' : [5, 7, 10],
        'use_centroids' : 'N',
        'use_remote_sensing' : 'Y',
        'use_gaes' : 'N',
        'use_per_year_crop_calendar' : 'N',
        'early_season_prediction' : 'N',
        'early_season_end_dekad' : 0,
        'data_path' : '.',
        'output_path' : '.',
        'use_features_v2' : 'N',
        'save_features' : 'N',
        'use_saved_features' : 'N',
        'use_sample_weights' : 'N',
        'retrain_per_test_year' : 'N',
        'save_predictions' : 'Y',
        'use_saved_predictions' : 'N',
        'compare_with_mcyfs' : 'N',
        'debug_level' : 0,
    }

    # Description of configuration parameters
    # This should be in sync with config above
    self.config_desc = {
        'crop_name' : 'Crop name',
        'crop_id' : 'Crop ID',
        'season_crosses_calendar_year' : 'Crop growing season crosses calendar year boundary',
        'country_code' : 'Country code (e.g. NL)',
        'nuts_level' : 'NUTS level for yield prediction',
        'data_sources' : 'Input data sources',
        'clean_data' : 'Remove data or regions with duplicate or missing values',
        'use_yield_trend' : 'Estimate and use yield trend',
        'predict_yield_residuals' : 'Predict yield residuals instead of full yield',
        'find_optimal_trend_window' : 'Find optimal trend window',
        'trend_windows' : 'List of trend window lengths (number of years)',
        'use_centroids' : 'Use centroid coordinates and distance to coast',
        'use_remote_sensing' : 'Use remote sensing data (FAPAR)',
        'use_gaes' : 'Use agro-environmental zones data',
        'use_per_year_crop_calendar' : 'Use per region per year crop calendar',
        'early_season_prediction' : 'Predict yield early in the season',
        'early_season_end_dekad' : 'Early season end dekad relative to harvest',
        'data_path' : 'Path to all input data. Default is current directory.',
        'output_path' : 'Path to all output files. Default is current directory.',
        'use_features_v2' : 'Use feature design v2',
        'save_features' : 'Save features to a CSV file',
        'use_saved_features' : 'Use features from a CSV file',
        'use_sample_weights' : 'Use data sample weights based on crop area',
        'retrain_per_test_year' : 'Retrain a model for every test year',
        'save_predictions' : 'Save predictions to a CSV file',
        'use_saved_predictions' : 'Use predictions from a CSV file',
        'compare_with_mcyfs' : 'Compare predictions with MARS Crop Yield Forecasting System',
        'debug_level' : 'Debug level to control amount of debug information',
    }

    ########### Machine learning configuration ###########
    # mutual correlation threshold for features
    self.feature_correlation_threshold = 0.9

    # test fraction
    self.test_fraction = 0.3

    # scaler
    self.scaler = StandardScaler()

    # Feature selection algorithms. Initialized in getFeatureSelectors().
    self.feature_selectors = {}

    # prediction algorithms
    self.estimators = {
        # linear model
        'Ridge' : {
            'estimator' : Ridge(random_state=42, max_iter=1000, tol=1e-3,
                                normalize=False, fit_intercept=True),
            'param_space' : {
                "estimator__alpha": Real(1e-1, 1e+2, prior='log-uniform')
            }
        },
        'KNN' : {
            'estimator' : KNeighborsRegressor(n_jobs=-1),
            'param_space' : {
                "estimator__n_neighbors": Integer(7, 15),
                "estimator__weights": Categorical(['uniform', 'distance']),
                "estimator__metric" : Categorical(['minkowski', 'manhattan']),# hassanatDistance])
            }
        },
        # SVM regression
        'SVR' : {
            'estimator' : SVR(gamma='scale', epsilon=1e-1, tol=1e-3,
                              max_iter=1000, shrinking=True),
            'param_space' : {
                "estimator__C": Real(1e-2, 5e+2, prior='log-uniform'),
                "estimator__epsilon": Real(1e-2, 5e-1, prior='log-uniform'),
                "estimator__gamma": Categorical(['auto', 'scale']),
                "estimator__kernel": Categorical(['rbf', 'linear']),
            }
        },
        # gradient boosted decision trees
        'GBDT' : {
            'estimator' : GradientBoostingRegressor(loss='huber', max_features='log2',
                                                    max_depth=10, min_samples_leaf=10,
                                                    tol=1e-3, n_iter_no_change=5,
                                                    subsample=0.6, ccp_alpha=1e-2,
                                                    n_estimators=500, random_state=42),
            'param_space' : {
                "estimator__learning_rate": Real(1e-3, 1e-1, prior='log-uniform'),
                "estimator__loss" : Categorical(['huber', 'lad']),
                # "estimator__max_depth": Integer(5, 10),
                "estimator__min_samples_leaf": Integer(5, 20),
                # "estimator__n_estimators": Integer(100, 500),
            }
        }
    }

    # k-fold validation metric for feature selection
    self.fs_cv_metric = 'neg_mean_squared_error'
    # k-fold validation metric for training
    self.est_cv_metric = 'neg_mean_squared_error'

    # Performance evaluation metrics:
    # sklearn supports these metrics:
    # 'explained_variance', 'max_error', 'neg_mean_absolute_error
    # 'neg_mean_squared_error', 'neg_root_mean_squared_error'
    # 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2'
    self.eval_metrics = {
        # EXP_VAR (y_true, y_obs) = 1 - ( var(y_true - y_obs) / var (y_true) )
        # 'EXP_VAR' : explained_variance_score,
        # MAE (y_true, y_obs) = ( 1 / n ) * sum_i-n ( | y_true_i - y_obs_i | )
        # 'MAE' : mean_absolute_error,
        # MdAE (y_true, y_obs) = median ( | y_true_1 - y_obs_1 |, | y_true_2 - y_obs_2 |, ... )
        # 'MdAE' : median_absolute_error,
        # MAPE (y_true, y_obs) = ( 1 / n ) * sum_i-n ( ( y_true_i - y_obs_i ) / y_true_i )
        'MAPE' : meanAbsolutePercentageError,
        # MSE (y_true, y_obs) = ( 1 / n ) * sum_i-n ( y_true_i - y_obs_i )^2
        'RMSE' : mean_squared_error,
        # R2 (y_true, y_obs) = 1 - ( ( sum_i-n ( y_true_i - y_obs_i )^2 )
        #                           / sum_i-n ( y_true_i - mean(y_true) )^2)
        'R2' : r2_score,
    }

  ########### Setters and getters ###########
  def setCropName(self, crop_name):
    """Set the crop name"""
    crop = crop_name.lower()
    assert crop in crop_id_dict
    self.config['crop_name'] = crop
    self.config['crop_id'] = cropNameToID(crop_id_dict, crop)

  def getCropName(self):
    """Return the crop name"""
    return self.config['crop_name']

  def setCropID(self, crop_id):
    """Set the crop ID"""
    assert crop_id in crop_name_dict
    self.config['crop_id'] = crop_id
    self.config['crop_name'] = cropIDToName(crop_name_dict, crop_id)

  def getCropID(self):
    """Return the crop ID"""
    return self.config['crop_id']

  def setSeasonCrossesCalendarYear(self, season_crosses):
    """Set whether the season crosses calendar year boundary"""
    scross = season_crosses.upper()
    assert scross in ['Y', 'N']
    self.config['season_crosses_calendar_year'] = scross

  def seasonCrossesCalendarYear(self):
    """Return whether the season crosses calendar year boundary"""
    return (self.config['season_crosses_calendar_year'] == 'Y')

  def setCountryCode(self, country_code):
    """Set the country code"""
    if (country_code is None):
      self.config['country_code'] = None
    else:
      ccode = country_code.upper()
      assert len(ccode) == 2
      assert ccode in countries
      self.config['country_code'] = ccode

  def getCountryCode(self):
    """Return the country code"""
    return self.config['country_code']

  def setNUTSLevel(self, nuts_level):
    """Set the NUTS level"""
    nuts = nuts_level.upper()
    assert nuts in nuts_levels
    self.config['nuts_level'] = nuts

  def getNUTSLevel(self):
    """Return the NUTS level"""
    return self.config['nuts_level']

  def setDataSources(self, data_sources):
    """Get the data sources"""
    # TODO: some validation
    self.config['data_sources'] = data_sources

  def updateDataSources(self, data_src, include_src, nuts_level=None):
    """add or remove data_src from data sources"""
    src_nuts = self.getNUTSLevel()
    if (nuts_level is not None):
      src_nuts = nuts_level

    data_sources = self.config['data_sources']
    # no update required
    if (((include_src == 'Y') and (data_src in data_sources)) or
        ((include_src == 'N') and (data_src not in data_sources))):
      return

    if (include_src == 'Y'):
      if (isinstance(data_sources, dict)):
        data_sources[data_src] = src_nuts
      else:
        data_sources.append(data_src)
    else:
      if (isinstance(data_sources, dict)):
        del data_sources[data_src]
      else:
        data_sources.remove(data_src)

    self.config['data_sources'] = data_sources

  def getDataSources(self):
    """Return the data sources"""
    return self.config['data_sources']

  def setCleanData(self, clean_data):
    """Set whether to clean data with duplicate or missing values"""
    do_clean = clean_data.upper()
    assert do_clean in ['Y', 'N']
    self.config['clean_data'] = do_clean

  def cleanData(self):
    """Return whether to clean data with duplicate or missing values"""
    return (self.config['clean_data'] == 'Y')

  def setUseYieldTrend(self, use_trend):
    """Set whether to use yield trend"""
    use_yt = use_trend.upper()
    assert use_yt in ['Y', 'N']
    self.config['use_yield_trend'] = use_yt

  def useYieldTrend(self):
    """Return whether to use yield trend"""
    return (self.config['use_yield_trend'] == 'Y')

  def setPredictYieldResiduals(self, pred_res):
    """Set whether to use predict yield residuals"""
    pred_yres = pred_res.upper()
    assert pred_yres in ['Y', 'N']
    self.config['predict_yield_residuals'] = pred_yres

  def predictYieldResiduals(self):
    """Return whether to use predict yield residuals"""
    return (self.config['predict_yield_residuals'] == 'Y')

  def setFindOptimalTrendWindow(self, find_opt):
    """Set whether to find optimal trend window for each year"""
    find_otw = find_opt.upper()
    assert find_otw in ['Y', 'N']
    self.config['find_optimal_trend_window'] = find_otw

  def findOptimalTrendWindow(self):
    """Return whether to find optimal trend window for each year"""
    return (self.config['find_optimal_trend_window'] == 'Y')

  def setTrendWindows(self, trend_windows):
    """Set trend window lengths (years)"""
    assert isinstance(trend_windows, list)
    assert len(trend_windows) > 0

    # trend windows less than 2 years do not make sense
    for tw in trend_windows:
      assert tw > 2

    self.config['trend_windows'] = trend_windows

  def getTrendWindows(self):
    """Return trend window lengths (years)"""
    return self.config['trend_windows']

  def setUseCentroids(self, use_centroids):
    """Set whether to use centroid coordinates and distance to coast"""
    use_ct = use_centroids.upper()
    assert use_ct in ['Y', 'N']
    self.config['use_centroids'] = use_ct
    self.updateDataSources('CENTROIDS', use_ct)

  def useCentroids(self):
    """Return whether to use centroid coordinates and distance to coast"""
    return (self.config['use_centroids'] == 'Y')

  def setUseRemoteSensing(self, use_remote_sensing):
    """Set whether to use remote sensing data"""
    use_rs = use_remote_sensing.upper()
    assert use_rs in ['Y', 'N']
    self.config['use_remote_sensing'] = use_rs
    self.updateDataSources('REMOTE_SENSING', use_rs, 'NUTS2')

  def useRemoteSensing(self):
    """Return whether to use remote sensing data"""
    return (self.config['use_remote_sensing'] == 'Y')

  def setUseGAES(self, use_gaes):
    """Set whether to use GAES data"""
    use_aez = use_gaes.upper()
    assert use_aez in ['Y', 'N']
    self.config['use_gaes'] = use_aez
    self.updateDataSources('GAES', use_aez)
    self.updateDataSources('CROP_AREA', use_aez)

  def useGAES(self):
    """Return whether to use GAES data"""
    return (self.config['use_gaes'] == 'Y')

  def setUsePerYearCropCalendar(self, use_per_year_cc):
    """Set whether to use per region, per year crop calendar"""
    per_year = use_per_year_cc.upper()
    assert per_year in ['Y', 'N']
    self.config['use_per_year_crop_calendar'] = per_year

  def usePerYearCropCalendar(self):
    """Return whether to use per region, per year crop calendar"""
    return (self.config['use_per_year_crop_calendar'] == 'Y')

  def setEarlySeasonPrediction(self, early_season):
    """Set whether to do early season prediction"""
    ep = early_season.upper()
    assert ep in ['Y', 'N']
    self.config['early_season_prediction'] = ep

  def earlySeasonPrediction(self):
    """Return whether to do early season prediction"""
    return (self.config['early_season_prediction'] == 'Y')

  def setEarlySeasonEndDekad(self, end_dekad):
    """Set early season prediction dekad"""
    dekads_range = [dek for dek in range(1, 37)]
    assert end_dekad in dekads_range
    self.config['early_season_end_dekad'] = end_dekad

  def getEarlySeasonEndDekad(self):
    """Return early season prediction dekad"""
    return self.config['early_season_end_dekad']

  def setDataPath(self, data_path):
    """Set the data path"""
    # TODO: some validation
    self.config['data_path'] = data_path

  def getDataPath(self):
    """Return the data path"""
    return self.config['data_path']

  def setOutputPath(self, out_path):
    """Set the path to output files. TODO: some validation."""
    self.config['output_path'] = out_path

  def getOutputPath(self):
    """Return the path to output files."""
    return self.config['output_path']

  def setUseFeaturesV2(self, use_v2):
    """Set whether to use features v2"""
    ft_v2 = use_v2.upper()
    assert ft_v2 in ['Y', 'N']
    self.config['use_features_v2'] = ft_v2

  def useFeaturesV2(self):
    """Return whether to use features v2"""
    return (self.config['use_features_v2'] == 'Y')

  def setSaveFeatures(self, save_ft):
    """Set whether to save features in a CSV file"""
    sft = save_ft.upper()
    assert sft in ['Y', 'N']
    self.config['save_features'] = sft

  def saveFeatures(self):
    """Return whether to save features in a CSV file"""
    return (self.config['save_features'] == 'Y')

  def setUseSavedFeatures(self, use_saved):
    """Set whether to use features from CSV file"""
    saved = use_saved.upper()
    assert saved in ['Y', 'N']
    self.config['use_saved_features'] = saved

  def useSavedFeatures(self):
    """Return whether to use to use features from CSV file"""
    return (self.config['use_saved_features'] == 'Y')

  def setUseSampleWeights(self, use_weights):
    """Set whether to use data sample weights"""
    use_sw = use_weights.upper()
    assert use_sw in ['Y', 'N']
    self.config['use_sample_weights'] = use_sw

  def useSampleWeights(self):
    """Return whether to use data sample weights"""
    return (self.config['use_sample_weights'] == 'Y')

  def setRetrainPerTestYear(self, per_test_year):
    """Set whether to retrain the model for every test year"""
    pty = per_test_year.upper()
    assert pty in ['Y', 'N']
    self.config['retrain_per_test_year'] = pty

  def retrainPerTestYear(self):
    """Return whether to retrain the model for every test year"""
    return (self.config['retrain_per_test_year'] == 'Y')

  def setSavePredictions(self, save_pred):
    """Set whether to save predictions in a CSV file"""
    spd = save_pred.upper()
    assert spd in ['Y', 'N']
    self.config['save_predictions'] = spd

  def savePredictions(self):
    """Return whether to save predictions in a CSV file"""
    return (self.config['save_predictions'] == 'Y')

  def setUseSavedPredictions(self, use_saved):
    """Set whether to use predictions from CSV file"""
    saved = use_saved.upper()
    assert saved in ['Y', 'N']
    self.config['use_saved_predictions'] = saved

  def useSavedPredictions(self):
    """Return whether to use to use predictions from CSV file"""
    return (self.config['use_saved_predictions'] == 'Y')

  def setCompareWithMCYFS(self, compare_mcyfs):
    """Set whether to compare predictions with MCYFS"""
    comp_mcyfs = compare_mcyfs.upper()
    assert comp_mcyfs in ['Y', 'N']
    self.config['compare_with_mcyfs'] = comp_mcyfs

  def compareWithMCYFS(self):
    """Return whether to compare predictions with MCYFS"""
    return (self.config['compare_with_mcyfs'] == 'Y')

  def setDebugLevel(self, debug_level):
    """Set the debug level"""
    assert debug_level in debug_levels
    self.config['debug_level'] = debug_level

  def getDebugLevel(self):
    """Return the debug level"""
    return self.config['debug_level']

  def updateConfiguration(self, config_update):
    """Update configuration"""
    assert isinstance(config_update, dict)
    for k in config_update:
      assert k in self.config

      # keys that need special handling
      special_cases = {
          'crop_name' : self.setCropName,
          'crop_id' : self.setCropID,
          'use_centroids' : self.setUseCentroids,
          'use_remote_sensing' : self.setUseRemoteSensing,
          'use_gaes' : self.setUseGAES,
      }

      if (k not in special_cases):
        self.config[k] = config_update[k]
        continue

      # special case
      special_cases[k](config_update[k])

  def printConfig(self, log_fh):
    """Print current configuration and write configuration to log file."""
    config_str = '\nCurrent ML Baseline Configuration'
    config_str += '\n--------------------------------'
    for k in self.config:
      if (isinstance(self.config[k], dict)):
        conf_keys = list(self.config[k].keys())
        if (not isinstance(conf_keys[0], str)):
          conf_keys = [str(k) for k in conf_keys]

        config_str += '\n' + self.config_desc[k] + ': ' + ', '.join(conf_keys)
      elif (isinstance(self.config[k], list)):
        conf_vals = self.config[k]
        if (not isinstance(conf_vals[0], str)):
          conf_vals = [str(k) for k in conf_vals]

        config_str += '\n' + self.config_desc[k] + ': ' + ', '.join(conf_vals)
      else:
        conf_val = self.config[k]
        if (not isinstance(conf_val, str)):
          conf_val = str(conf_val)

        config_str += '\n' + self.config_desc[k] + ': ' + conf_val

    config_str += '\n'
    log_fh.write(config_str + '\n')
    print(config_str)

  # Machine learning configuration
  def getFeatureCorrelationThreshold(self):
    """Return threshold for removing mutually correlated features"""
    return self.feature_correlation_threshold

  def setFeatureCorrelationThreshold(self, corr_thresh):
    """Set threshold for removing mutually correlated features"""
    assert (corr_thresh > 0.0 and corr_thresh < 1.0)
    self.feature_correlation_threshold = corr_thresh

  def getTestFraction(self):
    """Return test set fraction (of full dataset)"""
    return self.test_fraction

  def setTestFraction(self, test_fraction):
    """Set test set fraction (of full dataset)"""
    assert (test_fraction > 0.0 and test_fraction < 1.0)
    self.test_fraction = test_fraction

  def getFeatureScaler(self):
    """Return feature scaling method"""
    return self.scaler

  def setFeatureScaler(self, scaler):
    """Set feature scaling method"""
    assert (isinstance(scaler, MinMaxScaler) or isinstance(scaler, StandardScaler))
    self.scaler = scaler

  def getFeatureSelectionCVMetric(self):
    """Return metric for feature selection using K-fold validation"""
    return self.fs_cv_metric

  def setFeatureSelectionCVMetric(self, fs_metric):
    """Return metric for feature selection using K-fold validation"""
    assert fs_metric in self.eval_metrics
    self.fs_cv_metric = fs_metric

  def getAlgorithmTrainingCVMetric(self):
    """Return metric for hyperparameter optimization using K-fold validation"""
    return self.est_cv_metric

  def setFeatureSelectionCVMetric(self, est_metric):
    """Return metric for hyperparameter optimization using K-fold validation"""
    assert est_metric in self.eval_metrics
    self.est_cv_metric = est_metric

  def getFeatureSelectors(self, num_features):
    """Feature selection methods"""
    # already defined?
    if (len(self.feature_selectors) > 0):
      return self.feature_selectors

    # Early season prediction can have less than 10 features
    min_features = 10
    if (num_features < 10):
      min_features = num_features - 1

    max_features = [i for i in range(1, num_features) if ((i % 10) == 0)]
    if (not max_features):
      max_features = [num_features]

    rf = RandomForestRegressor(bootstrap=True, max_features='log2', ccp_alpha=1e-2,
                               max_depth=10, min_samples_leaf=10, n_estimators=500,
                               random_state=42, oob_score=True)

    lasso = Lasso(copy_X=True, fit_intercept=True, normalize=False,
                  tol=1e-3, random_state=42, selection='random')

    self.feature_selectors = {
      # random forest
      'random_forest' : {
          'selector' : SelectFromModel(rf, threshold='median'),
          'param_space' : {
              "selector__estimator__min_samples_leaf" : Integer(5, 20),
              # "selector__estimator__n_estimators" : Integer(100, 500),
              # "selector__estimator__max_depth" : Integer(5, 10),
              "selector__max_features" : Integer(min_features, num_features),
          }
      },
      # recursive feature elimination using Lasso
      'RFE_Lasso' : {
          'selector' : RFE(lasso),
          'param_space' : {
              "selector__estimator__alpha" : Real(1e-2, 1e+1, prior='log-uniform'),
              "selector__n_features_to_select" : Integer(min_features, num_features),
          }
      },
      # # NOTE: Mutual info raises an error when used with spark parallel backend.
      # univariate feature selection
      # 'mutual_info' : {
      #     'selector' : SelectKBest(mutual_info_regression),
      #     'param_space' : {
      #         "selector__k" : Integer(min_features, max_features),
      #     }
      # },
    }

    return self.feature_selectors

  def setFeatureSelectors(self, ft_sel):
    """Set feature selection algorithms"""
    assert isinstance(ft_sel, dict)
    assert len(ft_sel) > 0
    for sel in ft_sel:
      assert isinstance(sel, dict)
      assert 'selector' in sel
      assert 'param_space' in sel
      # add cases if other feature selection methods are used
      assert (isinstance(sel['selector'], SelectKBest) or
              isinstance(sel['selector'], SelectFromModel) or
              isinstance(sel['selector'], RFE))
      assert isinstance(sel['param_space'], dict)

    self.feature_selectors = ft_sel

  def getEstimators(self):
    """Return machine learning algorithms for prediction"""
    return self.estimators
  
  def setEstimators(self, estimators):
    """Set machine learning algorithms for prediction"""
    assert isinstance(estimators, dict)
    assert len(estimators) > 0
    for est in estimators:
      assert isinstance(est, dict)
      assert 'estimator' in est
      assert 'param_space' in est
      assert isinstance(est['param_space'], dict)

    self.estimators = estimators
  
  def getEvaluationMetrics(self):
    """Return metrics to evaluate predictions of algorithms"""
    return self.eval_metrics
  
  def setEvaluationMetrics(self, metrics):
    assert isinstance(estimators, dict)
    self.eval_metrics = metrics
