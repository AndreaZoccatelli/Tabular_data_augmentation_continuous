Graphs
================
Andrea Zoccatelli
2023-07-01

``` r
library(GenCopula)
```

``` r
Best = read.csv("BestCase.csv")

Best %>% ggplot()+
  geom_point(aes(feature.1, feature.2, colour = factor(y)))+
  ggtitle("40-50% Safe Observations")+
  xlab("feature 1")+
  ylab("feature 2")+
  labs(color='Class')+
  theme_minimal()
```

![](Graphs_files/figure-gfm/BestCase-1.png)<!-- -->

``` r
twth = read.csv("20_30Safe.csv")

twth %>% ggplot()+
  geom_point(aes(feature.1, feature.2, colour = factor(y)))+
  ggtitle("20-30% Safe Observations")+
  xlab("feature 1")+
  ylab("feature 2")+
  labs(color='Class')+
  theme_minimal()
```

![](Graphs_files/figure-gfm/20_30Safe-1.png)<!-- -->

``` r
ltw = read.csv("less20Safe.csv")

ltw %>% ggplot()+
  geom_point(aes(feature.1, feature.2, colour = factor(y)))+
  ggtitle("~20% or less Safe observations")+
  xlab("feature 1")+
  ylab("feature 2")+
  labs(color='Class')+
  theme_minimal()
```

![](Graphs_files/figure-gfm/less20Safe-1.png)<!-- -->

``` r
tpc = read.csv("10perc_minority.csv")

tpc %>% ggplot()+
  geom_point(aes(feature.1, feature.2, colour = factor(y)))+
  ggtitle("10% Minority Class")+
  xlab("feature 1")+
  ylab("feature 2")+
  labs(color='Class')+
  theme_minimal()
```

![](Graphs_files/figure-gfm/10perc_minority-1.png)<!-- -->

``` r
fpc = read.csv("5perc_minority.csv")

fpc %>% ggplot()+
  geom_point(aes(feature.1, feature.2, colour = factor(y)))+
  ggtitle("5% Minority Class")+
  xlab("feature 1")+
  ylab("feature 2")+
  labs(color='Class')+
  theme_minimal()
```

![](Graphs_files/figure-gfm/5perc_minority-1.png)<!-- -->

``` r
r = read.csv("default.csv")

r %>% ggplot()+
  geom_point(aes(balance, income, colour = factor(y)))+
  ggtitle("Default dataset")+
  xlab("balance")+
  ylab("income")+
  labs(color='Class')+
  theme_minimal()
```

![](Graphs_files/figure-gfm/default-1.png)<!-- -->
