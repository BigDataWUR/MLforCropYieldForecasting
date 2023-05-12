# A weakly supervised framework for high resolution crop yield forecasts

## Linear Trend and Strongly Supervised models for Europe
The linear trend models and strongly supervised GBDT models for both High Resolution (HR) and Low Resolution (LR) are implemented in `mloptimized_combo.ipynb`. The strongly supervised LSTM models are implemented in `crop_yield_DL.ipynb`.

## Linear Trend and Strongly Supervised models for the US
The linear trend models and strongly supervised models for both High Resolution (HR) and Low Resolution (LR) are implemented in `mloptimized_bayes_US.ipynb`. The strongly supervised LSTM models are implemented in `crop_yield_DL_US.ipynb`.

## Naive disaggregation models
The naive disaggregation models (Naive Trend, Naive GBDT and Naive LSTM) are based on model forecasts from the strongly supervised models at low resolution.

## Weakly supervised models for Europe
The weakly supervised models for Europe are implemented in `dl_weak_supervision.ipynb`.

## Weakly supervised models for the US
The weakly supervised models for Europe are implemented in `dl_weak_supervision_US.ipynb`.

## Naive Trend Model for COUNTY to GRIDS Disaggregation
The notebook `dl_weak_supervision_US.ipynb` includes evaluation of the naive trend model.

## Sample data
Sample data (for the US) can be found in [Zenodo] (https://doi.org/10.5281/zenodo.7751191).
