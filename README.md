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

# References

This repo is used to generate results in the following paper:

La Cava, W., Lee, P. C., Ajmal, I., Ding, X., Solanki, P., Cohen, J. B., Moore, J. H., & Herman, D. S. (2020).
Application of concise machine learning to construct accurate and interpretable EHR computable phenotypes.
[MedRxiv, 2020.12.12.20248005.](https://doi.org/10.1101/2020.12.12.20248005)

Additional results can be found at https://bitbucket.org/hermanlab/ehr_feat . 

## Contact

 - William La Cava: lacava@upenn.edu
 - Daniel Herman: Daniel.Herman2@pennmedicine.upenn.edu

## Acknowledgments

We would like to thank Debbie Cohen for helpful discussions about secondary hypertension.
W. La Cava was supported by NIH grant R00-LM012926.
This work was supported by Grant 2019084 from the Doris Duke Charitable Foundation and the University of Pennsylvania.
W.La Cava was supported by NIH grant K99LM012926.
J.H. Moore and W. La Cava were supported by NIH grant R01 LM010098.
J. B. Cohen was supported by NIH grants K23HL133843 and R01HL153646.
