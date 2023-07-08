# Augmentation of tabular data with continuous features for binary imbalanced classification problems

The aim of this project is to augment the observations that belong to the minority class using copula sampling and conditional GANs in order to improve the performance of the classifiers for binary imbalanced classification problems.

- For the augmentation based on copulas, my library, <a href="https://github.com/AndreaZoccatelli/GenCopula" target="_blank">GenCopula</a> has been used.
``` r
library(devtools)
install_github("AndreaZoccatelli/GenCopula")
```
- The library used for the augmentation based on cGAN is <a href="https://github.com/sdv-dev/CTGAN" target="_blank">CTGAN</a>
- To re-create the datasets used in the project run <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/Create_data.ipynb" target="_blank">Create_data.ipynb</a>

- These notebooks report the results obtained on the different dataset:
    - <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/20_30Safe.md" target="_blank">20-30% Safe</a>
