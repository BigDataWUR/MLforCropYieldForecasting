# Implementation of Machine Learning Baseline for Crop Yield Prediction

## Test Environment
Google Colab environment or Microsoft Azure Databricks can be used to run
the Jupiter notebook version of the implementation. The python script has
been tested in Google Dataproc cluster. In general, running the python script
or Jupiter notebook in other environments should be possible. The data has to
be uploaded to some location and `data_path` should point to that location.
Sample data will be available [here](https://doi.org/10.5281/zenodo.4312941).

## Google Colab Notes
To run the script in Google Colab environment
1. Download the data directory and save it somewhere convenient.
2. Open the notebook using Google Colaboratory.
3. Create a copy of the notebook for yourself.
4. Click connect on the right hand side of the bar below menu items.
   When you are connected to a machine, you will see a green tick mark 
   and bars showing RAM and disk.
5. Click the folder icon on the left sidebar and click upload.
   Upload the data files you downloaded. Click *Ok* when you see a warning 
   saying the files will be deleted after the session is disconnected.
6. Use *Runtime* -> *Run before* option to run all cells before 
   **Set Configuration**.
7. Run the remaining cells except **Python Script Main**.
   The configuration subsection allows you to change configuration 
   and rerun experiments.

## Microsoft Azure Databricks Notes

To run the script in Microsoft Azure Databricks environment, download the notebook and
1. Create a Spark cluster.
2. Add necessary libraries to the cluster using the libraries tab from the cluster details page.
3. Import notebook to your workspace.
4. In `Global Variables`, remove the `if (test_env == 'notebook'):` block meant for Colab environment.
5. Use `dbfs` command line tool to upload data. 
6. Set data path to something like `dbfs:/<dir>` and output path to something like `/dbfs/<dir>`.
7. Attach your notebook to the Spark cluster and run.

## Google Dataproc Notes

To run the script in Google Dataproc environment, download the python sript and
1. Change test_env to `cluster`.
2. Remove Spark installation commands meant for Google Colab environment.
3. Create a storage bucket: the notebook uses the name *ml-spark-1*.
4. Upload data to storage bucket: create data/ and scripts/ directories inside
   the bucket. Inside data create directories for countries, e.g. NUTS2-NL,
   NUTS3-FR. Upload data for the Netherlands to data/NUTS2-NL/ and data for
   France to data/NUTS3-FR. Upload *mlbaseline.py* script to scripts/.
5. Copy *pip-install.sh*. Use the following commad in Google Cloud Shell.

`$ gsutil cp gs://dataproc-initialization-actions/python/pip-install.sh gs://ml-spark-1/scripts`

6. Create cluster with initialization actions to install packages *pandas,
   sklearn, matplotlib, joblibspark*.
7. Run python script uploaded to storage bucket:

`$ gcloud dataproc jobs submit pyspark --cluster=ml-spark-cluster1 --region=europe-west1 \`

`  gs://ml-spark-1/scripts/mlbaseline.py -- \`

`  --country NL --nuts-level NUTS2 --crop potatoes`

`  --data-path gs://ml-spark-1/data/NUTS2-NL --output-path gs://ml-spark-1/output`

Options supported by `mlbaseline.py`:

` args_dict = {

      '--crop' : { 'type' : str,
                   'default' : 'potatoes',
                   'help' : 'crop name (default: potatoes)',
                 },

      '--crosses-calendar-year' : { 'type' : str,
                                    'default' : 'N',
                                    'choices' : ['Y', 'N'],
                                    'help' : 'crop growing season crosses calendar year boundary (default: N)',
                                  },

      '--country' : { 'type' : str,
                      'default' : 'NL',
                      'choices' : ['NL', 'DE', 'FR'],
                      'help' : 'country code (default: NL)',
                    },

      '--nuts-level' : { 'type' : str,
                         'default' : 'NUTS2',
                         'choices' : ['NUTS2', 'NUTS3'],
                         'help' : 'country code (default: NL)',
                       },

      '--data-path' : { 'type' : str,
                        'default' : '.',
                        'help' : 'path to data files (default: .)',
                       },

      '--output-path' : { 'type' : str,
                          'default' : '.',
                          'help' : 'path to output files (default: .)',
                        },

      '--yield-trend' : { 'type' : str,
                          'default' : 'N',
                          'choices' : ['Y', 'N'],
                          'help' : 'estimate and use yield trend (default: N)',
                        },

      '--optimal-trend-window' : { 'type' : str,
                                   'default' : 'N',
                                   'choices' : ['Y', 'N'],
                                   'help' : 'find optimal trend window for each year (default: N)',
                                 },

      '--predict-residuals' : { 'type' : str,
                                'default' : 'N',
                                'choices' : ['Y', 'N'],
                                'help' : 'predict yield residuals instead of full yield (default: N)',
                              },

      '--early-season' : { 'type' : str,
                           'default' : 'N',
                           'choices' : ['Y', 'N'],
                           'help' : 'early season prediction (default: N)',
                         },

      '--early-season-end' : { 'type' : int,
                               'default' : 15,
                               'help' : 'early season end dekad (default: 15)',
                             },

      '--centroids' : { 'type' : str,
                        'default' : 'N',
                        'choices' : ['Y', 'N'],
                        'help' : 'use centroid coordinates and distance to coast (default: N)',
                      },

      '--remote-sensing' : { 'type' : str,
                             'default' : 'Y',
                             'choices' : ['Y', 'N'],
                             'help' : 'use remote sensing data (default: Y)',
                           },

      '--save-features' : { 'type' : str,
                            'default' : 'N',
                            'choices' : ['Y', 'N'],
                            'help' : 'save features to a CSV file (default: N)',
                          },

      '--use-saved-features' : { 'type' : str,
                                 'default' : 'N',
                                 'choices' : ['Y', 'N'],
                                 'help' : 'use features from a CSV file (default: N). Set ',
                               },

      '--save-predictions' : { 'type' : str,
                               'default' : 'Y',
                               'choices' : ['Y', 'N'],
                               'help' : 'save predictions to a CSV file (default: Y)',
                             },

      '--use-saved-predictions' : { 'type' : str,
                                    'default' : 'N',
                                    'choices' : ['Y', 'N'],
                                    'help' : 'use predictions from a CSV file (default: N)',
                                  },

      '--compare-with-mcyfs' : { 'type' : str,
                                 'default' : 'N',
                                 'choices' : ['Y', 'N'],
                                 'help' : 'compare predictions with MCYFS (default: N)',
                               },

      '--debug-level' : { 'type' : int,
                          'default' : 0,
                          'choices' : range(4),
                          'help' : 'amount of debug information to print (default: 0)',
                        },

  } `
