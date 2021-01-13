# Experiment to Run
class CYPExperiments:
  def __init__(self):
    # list of crops: must be in sync with crops in wofost and yield data
    self.crop_names = ['potatoes', 'sugar beets', 'spring barley', 'sunflower']

    # early season end dekad for crops above
    self.early_season_dict = {
        'potatoes' : 18,
        'sugarbeet' : 21,
        'spring barley' : 18,
        'sunflower' : 18,
    }

    # countries and yield prediction NUTS level, must be in sync with input data
    self.country_nuts = {
        'NL' : 'NUTS2',
        'FR' : 'NUTS3',
        'DE' : 'NUTS3'
    }

    # Currently running 4 experiments:
    # 1. No yield trend, randomg train-test split
    # 2. Yield trend, custom train-test split
    # 3. Early season prediction for 1
    # 4. Early season prediction for 2
    self.experiments = {
        # default
        'rand' : {},
        'trend' : { 'use_yield_trend' : 'Y' },
        'rand-early' : { 'use_yield_trend' : 'N', 'early_season_prediction' : 'Y' },
        'trend-early' : { 'use_yield_trend' : 'Y', 'early_season_prediction' : 'Y' },
    }

  def getCropNames(self):
    return self.crop_names

  def getCountryNUTS(self):
    return self.country_nuts

  def getEarlySeasonEndDekad(self, crop_name):
    try:
      return self.early_season_dict[crop_name]
    except KeyError as e:
      return 'NA'

  def getExperiments(self):
    return self.experiments
