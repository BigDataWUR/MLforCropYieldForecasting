from ..common import globals

if (globals.test_env == 'pkg'):
  from ..common.config import CYPConfiguration
  from ..workflow.data_loading import CYPDataLoader

class TestDataLoader():
  def __init__(self, spark):
    cyp_config = CYPConfiguration()
    self.nuts_level = cyp_config.getNUTSLevel()
    data_sources = { 'SOIL' : self.nuts_level }
    cyp_config.setDataSources(data_sources)
    cyp_config.setDebugLevel(2)

    self.data_loader = CYPDataLoader(spark, cyp_config)

  def testDataLoad(self):
    print('\nTest loadData, loadAllData')
    soil_df = self.data_loader.loadData('SOIL', self.nuts_level)
    assert soil_df is not None
    soil_df.show(5)

    all_dfs = self.data_loader.loadAllData()
    soil_df = all_dfs['SOIL']
    assert soil_df is not None
    soil_df.show(5)

  def runAllTests(self):
    print('\nTest Data Loader BEGIN\n')
    self.testDataLoad()
    print('\nTest Data Loader END\n')
