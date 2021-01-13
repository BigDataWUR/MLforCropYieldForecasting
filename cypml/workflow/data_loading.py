class CYPDataLoader:
  def __init__(self, spark, cyp_config):
    self.spark = spark
    self.data_path = cyp_config.getDataPath()
    self.country_code = cyp_config.getCountryCode()
    self.nuts_level = cyp_config.getNUTSLevel()
    self.data_sources = cyp_config.getDataSources()
    self.verbose = cyp_config.getDebugLevel()
    self.data_dfs = {}

  def loadFromCSVFile(self, data_path, src, nuts, country_code):
    """
    The implied filename for each source is:
    <data_source>_<nuts_level>_<country_code>.csv
    Examples: CENTROIDS_NUTS2_NL.csv, WOFOST_NUTS2_NL.csv.
    Schema is inferred from the file. We might want to specify the schema at some point.
    """
    if (country_code is not None):
      datafile = data_path + '/' + src  + '_' + nuts + '_' + country_code + '.csv'
    else:
      datafile = data_path + '/' + src  + '_' + nuts + '.csv'

    if (self.verbose > 1):
      print('Data file name', '"' + datafile + '"')

    df = self.spark.read.csv(datafile, header = True, inferSchema = True)
    return df

  def loadData(self, src, nuts_level):
    """
    Load data for a specific data source.
    nuts_level may one level or a list of levels.
    """
    data_path = self.data_path
    country_code = self.country_code
    assert src in self.data_sources

    if (isinstance(nuts_level, list)):
      src_dfs = []
      for nuts in nuts_level:
        df = self.loadFromCSVFile(data_path, src, nuts, country_code)
        src_dfs.append(df)

    elif (isinstance(nuts_level, str)):
      src_dfs = self.loadFromCSVFile(data_path, src, nuts_level, country_code)
    else:
      src_dfs = None

    return src_dfs

  def loadAllData(self):
    """
    NOT SUPPORTED:
    1. Schema is not defined.
    2. Loading data for multiple countries.
    3. Loading data from folders.
    Ioannis: Spark has a nice way of loading several files from a folder,
    and associating the file name on each record, using the function
    input_file_name. This allows to extract medatada from the path
    into the dataframe. In your case it could be the country name, etc.
    """
    data_dfs = {}
    for src in self.data_sources:
      nuts_level = self.nuts_level
      if (isinstance(self.data_sources, dict)):
        nuts_level = self.data_sources[src]
      # REMOTE_SENSING data is at NUTS2. If nuts_level is None, leave as is.
      elif ((src == 'REMOTE_SENSING') and (nuts_level is not None)):
        nuts_level = 'NUTS2'

      if ('METEO' in src):
        data_dfs['METEO'] = self.loadData(src, nuts_level)
      else:
        data_dfs[src] = self.loadData(src, nuts_level)

    if (self.verbose > 1):
      data_sources_str = ''
      for src in data_dfs:
        data_sources_str = data_sources_str + src + ', '

      # remove the comma and space from the end
      print('Loaded data:', data_sources_str[:-2])
      print('\n')

    return data_dfs
