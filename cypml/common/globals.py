# test_env = 'notebook'
# test_env = 'cluster'
test_env = 'pkg'

# change to False to skip tests
run_tests = False

# NUTS levels
nuts_levels = ['NUTS' + str(i) for i in range(4)]

# country codes
countries = ['BG', 'DE', 'ES', 'FR', 'HU', 'IT', 'NL', 'PL', 'RO']

# debug levels
debug_levels = [i for i in range(5)]

# Keeping these two mappings inside CYPConfiguration leads to SPARK-5063 error
# when lambda functions use them. Therefore, they are defined as globals now.

# crop name to id mapping
crop_id_dict = {
    'grain maize': 2,
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
    2 : 'grain maize',
    6 : 'sugarbeet',
    7 : 'potatoes',
    90 : 'soft wheat',
    93 : 'sunflower',
    95 : 'spring barley',
}

import pyspark

from pyspark.sql import functions as SparkF
from pyspark.sql import types as SparkT
