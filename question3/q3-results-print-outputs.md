# Metrics

```r
print(metrics)
```

| Model   | RMSE      | MAE      |
|---------|-----------|----------|
| Model A | 154.86113 | 120.38075|
| Model B | 25.37588  | 21.05761 |

```r
print(cv_results)
```

| RMSE     | MAE      |
|----------|----------|
| 59.36029 | 57.61594 |

```r
print(aic_values)
```

| df | AIC     |
|----|---------|
| 3  | 25477.10|
| 8  | 18357.09|

```r
print(bic_values)
```

| df | BIC     |
|----|---------|
| 3  | 25493.86|
| 8  | 18401.78|

```r
print(anova_test)
```

### Analysis of Variance Table

- **Model 1**: `altcoin_price ~ price`
- **Model 2**: `altcoin_price ~ price + lag_7 + lag_14 + lag_30 + altcoin_volume + altcoin_volatility`

| Res.Df | RSS      | Df | Sum of Sq  | F      | Pr(>F)    |
|--------|----------|----|------------|--------|-----------|
| 1969   | 47268461 |    |            |        |           |
| 1964   | 1269196  | 5  | 45999264   | 14236  | < 2.2e-16 *** |

Significance codes:  
- 0 ‘***’  
- 0.001 ‘**’  
- 0.01 ‘*’  
- 0.05 ‘.’  
- 0.1 ‘ ’  

```r
print(comparison_results)
```

| Model                              | RMSE      | MAE      |
|------------------------------------|-----------|----------|
| Model A (Current Bitcoin Price)    | 154.86113 | 120.38075|
| Model B (Lagged Bitcoin Prices)    | 25.37588  | 21.05761 |

```r
print(paste("Adjusted R-squared for Model A:", adjusted_r2_a))
```

**Adjusted R-squared for Model A:** 0.804810326369797

```r
print(paste("Adjusted R-squared for Model B:", adjusted_r2_b))
```

**Adjusted R-squared for Model B:** 0.994745656374584

```r
print("AIC Values:")
```

| df | AIC     |
|----|---------|
| 3  | 25477.10|
| 8  | 18357.09|

```r
print("BIC Values:")
```

| df | BIC     |
|----|---------|
| 3  | 25493.86|
| 8  | 18401.78|

```r
print(dm_test)
```

### Diebold-Mariano Test

- **Data**: `predictions_apredictions_b`
- **DM Statistic**: -4.0235
- **Forecast Horizon**: 1
- **Loss Function Power**: 2
- **p-value**: 2.975e-05
- **Alternative Hypothesis**: less

# Conclusion: Comparing RMSE, MAE, and Hypothesis Testing Results

```r
if (rmse_b < rmse_a) {
    print("Including lagged Bitcoin prices improves prediction accuracy for altcoin prices (Model B is better).")
} else {
    print("Including lagged Bitcoin prices does not improve prediction accuracy for altcoin prices (Model A is better).")
}
```
**Conclusion:** Including lagged Bitcoin prices improves prediction accuracy for altcoin prices (Model B is better).
