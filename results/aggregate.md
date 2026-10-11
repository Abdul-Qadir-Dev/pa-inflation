### nab (30 series)

| detector | F1 | PA-F1 | F1 @1% | PA-F1 @1% | AUROC |
|---|---|---|---|---|---|
| random | 0.188 [0.158, 0.290] | 0.969 [0.953, 0.981] | 0.017 [0.013, 0.019] | 0.914 [0.640, 0.957] | 0.500 [0.495, 0.508] |
| zscore | 0.219 [0.181, 0.306] | 0.980 [0.958, 1.000] | 0.028 [0.016, 0.037] | 0.957 [0.918, 0.969] | 0.487 [0.462, 0.547] |
| iforest | 0.338 [0.199, 0.455] | 0.951 [0.781, 1.000] | 0.082 [0.019, 0.177] | 0.729 [0.548, 0.966] | 0.614 [0.513, 0.689] |
| lstm_ae | 0.376 [0.242, 0.449] | 0.958 [0.719, 1.000] | 0.097 [0.040, 0.149] | 0.891 [0.507, 0.998] | 0.654 [0.488, 0.699] |

Median across series, interquartile range in brackets. F1 and PA-F1 use the best of 200 thresholds per series; the @1% columns use the threshold that flags 1% of test points. Seeds are averaged within a series first.
