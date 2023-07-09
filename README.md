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
    - <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/BestCase.md" target="_blank">Best case</a>
    - <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/20_30Safe.md" target="_blank">20-30% Safe</a>
    - <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/less20Safe.md" target="_blank">Less 20% Safe</a>
    - <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/10perc_minority.md" target="_blank">10% Minority</a>
    - <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/5perc_minority.md" target="_blank">5% Minority</a>
    - <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/4features.md" target="_blank">4 Features</a>
    - <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/8features.md" target="_blank">8 Features</a>
    - <a href="https://github.com/AndreaZoccatelli/Tabular_data_augmentation_continuous/blob/main/Default.md" target="_blank">Default</a>
