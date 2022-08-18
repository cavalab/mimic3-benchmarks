FEAT MIMIC-III Benchmarks
=========================

This repo contains code to benchmark [FEAT](https://github.com/cavalab/feat) on the phenotyping tasks in the [MIMIC-III Critical Care Database](http://www.nature.com/articles/sdata201635). 

This code builds directly upon the curation work done by @YerevaNN [here](https://github.com/YerevaNN/mimic3-benchmarks).

## Usage

- First, follow the guide from [YerevaNN](https://github.com/YerevaNN/mimic3-benchmarks) to extract the datasets. 


### Analysis Code

- `mimic3-benchmarks/mimic3models/phenotyping/tsfresh/main.py`: extracts time series features from the MIMIC-III data using [tsfres](tsfresh.readthedocs.io/). 
- `mimic3-benchmarks/mimic3models/phenotyping/feat/main.py`: trains and evaluates FEAT models on the phenotypes. 

### Scripts

Here are some convenience scripts for batch analysis: 

- `extract_tsfresh_features.sh`: extracts time series features. 
- `lpc_feat_tsfresh_phenotyping.sh`: trains feat models.
- `lpc_lr100_tsfresh_phenotyping.sh`*: trains logistic regression models of max dimensionality 100. 
