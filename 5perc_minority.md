5perc_minority
================
Andrea Zoccatelli
2023-05-28

``` r
library(GenCopula)
library(stargazer)
```

``` r
sel_path = "D:/unibg/APP"
models = c("Logistic", "RandForest", "XGBoost")
sel_filename = "5perc_minority.csv"
```

## Augmentation share tuning

``` r
DB = import_strat_split(sel_path, sel_filename)
id = stratify(DB@train_set)
minority = extract_minority(DB@train_set)
index = best_copula(dplyr::select(DB@train_set, -c(id,y)))

Database = c()
shares = c(0.2, 0.8, 0.6, 0.8, 1)

for (k in 1:length(shares)){
  name = paste0("share_",shares[k])
  table = pipeline(train = DB@train_set, id = id, models = models, share = k, best_copula_i = index[1], outliers_r = 1, nearest = 1)
  assign(name, table)
  Database[k] = name
  print(paste0(shares[k]*100, "% done"))
}
```

    ## [1] "20% done"
    ## [1] "80% done"
    ## [1] "60% done"
    ## [1] "80% done"
    ## [1] "100% done"

``` r
augmented_results = bind_rows(
  get(Database[1])@reg_summary[which.max(get(Database[1])@reg_summary$F1_avg),],
  get(Database[1])@aug_summary[which.max(get(Database[1])@aug_summary$F1_avg),],
  get(Database[2])@aug_summary[which.max(get(Database[2])@aug_summary$F1_avg),],
  get(Database[3])@aug_summary[which.max(get(Database[3])@aug_summary$F1_avg),],
  get(Database[4])@aug_summary[which.max(get(Database[4])@aug_summary$F1_avg),],
  get(Database[5])@aug_summary[which.max(get(Database[5])@aug_summary$F1_avg),]
)

best_s = shares[which.max(augmented_results[-1,]$F1_avg)]

augmented_results = augmented_results %>%
  mutate(Tuning = c('0', '0.2', '0.8', '0.6', '0.8', '1'), Dataset = "5% minority") %>%
  rename(Model = models)

augmented_results
```

    ##      Model    F1_avg     F1_var Tuning     Dataset
    ## 1  XGBoost 0.4071429 0.09271542      0 5% minority
    ## 2 Logistic 0.6880952 0.04186823    0.2 5% minority
    ## 3 Logistic 0.4959524 0.03710573    0.8 5% minority
    ## 4 Logistic 0.5181119 0.02606193    0.6 5% minority
    ## 5 Logistic 0.4959524 0.03710573    0.8 5% minority
    ## 6 Logistic 0.4233333 0.02938889      1 5% minority

``` r
#stargazer(augmented_results, summary = F, rownames = F) for latex output
stargazer(augmented_results, summary = F, rownames = F, type = "text")
```

    ## 
    ## =========================================
    ## Model    F1_avg F1_var Tuning   Dataset  
    ## -----------------------------------------
    ## XGBoost  0.407  0.093    0    5% minority
    ## Logistic 0.688  0.042   0.2   5% minority
    ## Logistic 0.496  0.037   0.8   5% minority
    ## Logistic 0.518  0.026   0.6   5% minority
    ## Logistic 0.496  0.037   0.8   5% minority
    ## Logistic 0.423  0.029    1    5% minority
    ## -----------------------------------------

## Training set

``` r
train = DB@train_set
#train %>% write.csv('less20Safe_t.csv') export for augmentation with cGAN
```

## Evaluation on test set

### Regular

``` r
F1_reg = c()
Precision_reg = c()
Recall_reg = c()

for (k in 1:length(models)){
  model_reg = select_fit_model(train, m = models[k])
  pred_reg = predictions(m = models[k], model_reg, test_set = DB@test_set)
  F1_reg[k] = round(F1_score(DB@test_set$y, pred_reg),3)
  Precision_reg[k] = round(Precision(DB@test_set$y, pred_reg),3)
  Recall_reg[k] = round(Recall(DB@test_set$y, pred_reg),3)
}

reg_scores = tibble(Model = models, F1 = F1_reg, Precision = Precision_reg,
                    Recall = Recall_reg, Obs. = "Regular", Dataset = "5% minority")
reg_scores
```

    ## # A tibble: 3 x 6
    ##   Model           F1 Precision Recall Obs.    Dataset    
    ##   <chr>        <dbl>     <dbl>  <dbl> <chr>   <chr>      
    ## 1 Logistic   NaN           NaN    0   Regular 5% minority
    ## 2 RandForest   0.333         1    0.2 Regular 5% minority
    ## 3 XGBoost      0.333         1    0.2 Regular 5% minority

``` r
#stargazer(reg_scores, summary = F, rownames = F) for latex output
stargazer(reg_scores, summary = F, rownames = F, type = "text")
```

    ## 
    ## =====================================================
    ## Model       F1   Precision Recall  Obs.     Dataset  
    ## -----------------------------------------------------
    ## Logistic    NaN     NaN      0    Regular 5% minority
    ## RandForest 0.333     1      0.2   Regular 5% minority
    ## XGBoost    0.333     1      0.2   Regular 5% minority
    ## -----------------------------------------------------

### Copula Augmented

``` r
train_a = augment(train = DB@train_set, best_copula_i = index[1], minority = minority, share = best_s, outliers_r = 1, nearest = 1)

F1_aug = c()
Precision_aug = c()
Recall_aug = c()

for (k in 1:length(models)){
  model_aug = select_fit_model(train_a, m = models[k])
  pred_aug = predictions(m = models[k], model_aug, test_set = DB@test_set)
  F1_aug[k] = round(F1_score(DB@test_set$y, pred_aug),3)
  Precision_aug[k] = round(Precision(DB@test_set$y, pred_aug),3)
  Recall_aug[k] = round(Recall(DB@test_set$y, pred_aug),3)
}

aug_scores = tibble(Model = models, F1 = F1_aug, Precision = Precision_aug,
                    Recall = Recall_aug, Obs. = "Augmented", Dataset = "5% minority")
aug_scores
```

    ## # A tibble: 3 x 6
    ##   Model         F1 Precision Recall Obs.      Dataset    
    ##   <chr>      <dbl>     <dbl>  <dbl> <chr>     <chr>      
    ## 1 Logistic   0.333       1      0.2 Augmented 5% minority
    ## 2 RandForest 0.286       0.5    0.2 Augmented 5% minority
    ## 3 XGBoost    0.333       1      0.2 Augmented 5% minority

``` r
#stargazer(aug_scores, summary = F, rownames = F)
stargazer(aug_scores, summary = F, rownames = F, type = "text")
```

    ## 
    ## =======================================================
    ## Model       F1   Precision Recall   Obs.      Dataset  
    ## -------------------------------------------------------
    ## Logistic   0.333     1      0.2   Augmented 5% minority
    ## RandForest 0.286    0.5     0.2   Augmented 5% minority
    ## XGBoost    0.333     1      0.2   Augmented 5% minority
    ## -------------------------------------------------------

### cGAN Augmented

``` r
train_GAN = read.csv("D:\\unibg\\APP\\Train_sets_cGAN\\5perc_minority_a.csv")
minority = train_GAN %>% filter(y == 1 & type == 'positive') %>% dplyr::select(-c(y,id,type))
synthetic = train_GAN %>% filter(type == 'synthetic') %>% dplyr::select(-c(y,id,type))
synthetic = nearest(minority = minority, generated = synthetic) %>%
  outliers_rmv() %>%
  mutate(type = "synthetic", y = 1)
minority = minority %>% mutate(type = "positive", y = 1)
negatives = train_GAN %>% filter(y == 0)
new_train_GAN = bind_rows(negatives, minority, synthetic)

train_GAN = train_GAN %>% dplyr::select(-type) %>% as_tibble()

new_train_GAN = new_train_GAN %>% dplyr::select(-type)
new_train_GAN$feature.1 = new_train_GAN$feature.1 %>% as.matrix() #the columns of test set are matrices, due to scale function
new_train_GAN$feature.2 = new_train_GAN$feature.2 %>% as.matrix()
F1_GAN = c()
Precision_GAN = c()
Recall_GAN = c()
for (k in 1:length(models)){
  model_GAN = select_fit_model(new_train_GAN, m = models[k])
  pred_GAN = predictions(m = models[k], model_GAN, test_set = DB@test_set)
  F1_GAN[k] = round(F1_score(DB@test_set$y, pred_GAN),3)
  Precision_GAN[k] = round(Precision(DB@test_set$y, pred_GAN),3)
  Recall_GAN[k] = round(Recall(DB@test_set$y, pred_GAN),3)
}

GAN_scores = tibble(Model = models, F1 = F1_GAN, Precision = Precision_GAN,
                    Recall = Recall_GAN, Obs. = "augmented", Dataset = "Best case")

GAN_scores
```

    ## # A tibble: 3 x 6
    ##   Model           F1 Precision Recall Obs.      Dataset  
    ##   <chr>        <dbl>     <dbl>  <dbl> <chr>     <chr>    
    ## 1 Logistic   NaN         NaN      0   augmented Best case
    ## 2 RandForest   0.333       1      0.2 augmented Best case
    ## 3 XGBoost      0.286       0.5    0.2 augmented Best case

``` r
#stargazer(GAN_scores, summary = F, rownames = F)
stargazer(GAN_scores, summary = F, rownames = F, type = "text")
```

    ## 
    ## =====================================================
    ## Model       F1   Precision Recall   Obs.     Dataset 
    ## -----------------------------------------------------
    ## Logistic    NaN     NaN      0    augmented Best case
    ## RandForest 0.333     1      0.2   augmented Best case
    ## XGBoost    0.286    0.5     0.2   augmented Best case
    ## -----------------------------------------------------
