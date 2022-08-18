FEAT MIMIC-III Benchmarks
=========================

This repo contains code to benchmark [FEAT](https://github.com/cavalab/feat) on the phenotyping tasks in the [MIMIC-III Critical Care Database](http://www.nature.com/articles/sdata201635). 

This code builds directly upon the curation work done by @YerevaNN [here](https://github.com/YerevaNN/mimic3-benchmarks).

## Usage


- `mimic3-benchmarks/mimic3models/phenotyping/tsfresh/main.py`: extracts time series features from the MIMIC-III data using [tsfres](tsfresh.readthedocs.io/). 
- `mimic3-benchmarks/mimic3models/phenotyping/feat/main.py`: trains and evaluates FEAT models on the phenotypes. 
