Run

`$ python main.py [options]`

from inside `/home/jovyan/work` directory in docker environment

Options supported by `main.py`:

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