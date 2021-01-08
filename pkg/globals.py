# test_env = 'notebook'
# test_env = 'cluster'
test_env = 'pkg'

# change to False to skip tests
run_tests = False

# NUTS levels
nuts_levels = ['NUTS' + str(i) for i in range(4)]

# country codes
countries = ['NL', 'FR', 'DE']

# debug levels
debug_levels = [i for i in range(5)]

# Keeping these two mappings inside CYPConfiguration leads to SPARK-5063 error
# when lambda functions use them. Therefore, they are defined as globals now.

# crop name to id mapping
crop_id_dict = {
    'sugar beet' : 6,
    'sugarbeet' : 6,
    'sugarbeets' : 6,
    'sugar beets' : 6,
    'total potatoes' : 7,
    'potatoes' : 7,
    'potato' : 7,
    'winter wheat' : 90,
    'soft wheat' : 90,
    'sunflower' : 93,
    'spring barley' : 95,
}

# crop id to name mapping
crop_name_dict = {
    6 : 'sugarbeet',
    7 : 'potatoes',
    90 : 'soft wheat',
    93 : 'sunflower',
    95 : 'spring barley',
}

import pyspark

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as SparkF
from pyspark.sql import types as SparkT

SparkContext.setSystemProperty('spark.executor.memory', '12g')
SparkContext.setSystemProperty('spark.driver.memory', '6g')
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

sc = SparkContext.getOrCreate()
sqlCtx = SQLContext(sc)
