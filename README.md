# Interpretability of deep learning models for crop yield forecasting

## Performance metrics using 10 models
The notebook `crop_yield_DL_performance.ipynb` is used to evaluate 10 models and compute performance metrics.

## Feature attributions and importance plots
The notebook `crop_yield_DL_ft_import.ipynb` is used to compute feature attributions from 10 models and 10 runs of feature attribution methods for each model. Feature attributions are plotted using bar plots and beeswarm plots. Beeswarm plot code is adapted from [SHAP](https://github.com/slundberg/shap). Feature attribution methods are from [Captum.ai](https://captum.ai).

## NUTS3 Trend Model and GBDT Model
The NUTS3 models that use NUTS3 yield data are implemented in `mloptimized_bayessearch.ipynb`.

## Test Environment
Google Colab environment or Microsoft Azure Databricks can be used to run
the Jupiter notebook version of the implementation.
