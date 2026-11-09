# Tabular-only ML edge task reports

These tables include only XGBoost, RandomForest, LogisticRegression, and TemporalCount.
They are not mixed with neural or historical graph baselines.

## tabular_ml_edge_inductive_mrr

| method             |   patent_us |   patent_in |   patent_cn |   thgl_software |   aminer_paper |   amazon_movies |   dblp |
|:-------------------|------------:|------------:|------------:|----------------:|---------------:|----------------:|-------:|
| XGBoost            |       1     |      0.9067 |      0.9056 |          0.0931 |         0.3158 |          0.834  | 0.2215 |
| RandomForest       |       1     |      0.9376 |      0.9819 |          0.0758 |         0.0856 |          0.6217 | 0.1058 |
| LogisticRegression |       1     |      0.9951 |      0.9936 |          0.1358 |         0.4149 |          0.8874 | 0.2413 |
| TemporalCount      |       0.982 |      0.0476 |      0.0476 |          0.0476 |         0.0476 |          0.0476 | 0.0476 |

## tabular_ml_edge_inductive_auc

| method             |   patent_us |   patent_in |   patent_cn |   thgl_software |   aminer_paper |   amazon_movies |   dblp |
|:-------------------|------------:|------------:|------------:|----------------:|---------------:|----------------:|-------:|
| XGBoost            |      1      |      0.9738 |      0.9793 |          0.4201 |         0.6864 |          0.9707 | 0.5789 |
| RandomForest       |      1      |      0.9743 |      0.9984 |          0.7111 |         0.6655 |          0.8906 | 0.7925 |
| LogisticRegression |      1      |      0.9987 |      0.9978 |          0.6135 |         0.7913 |          0.9571 | 0.5904 |
| TemporalCount      |      0.9939 |      0.6667 |      0.6666 |          0.6666 |         0.6619 |          0.6666 | 0.6664 |

## tabular_ml_edge_inductive_ap

| method             |   patent_us |   patent_in |   patent_cn |   thgl_software |   aminer_paper |   amazon_movies |   dblp |
|:-------------------|------------:|------------:|------------:|----------------:|---------------:|----------------:|-------:|
| XGBoost            |      1      |      0.8708 |      0.8327 |          0.0375 |         0.1659 |          0.6632 | 0.0669 |
| RandomForest       |      1      |      0.71   |      0.9314 |          0.0391 |         0.0386 |          0.4649 | 0.0468 |
| LogisticRegression |      1      |      0.9878 |      0.9847 |          0.0544 |         0.1859 |          0.7959 | 0.091  |
| TemporalCount      |      0.9825 |      0.0356 |      0.0356 |          0.0356 |         0.0355 |          0.0357 | 0.0356 |

## tabular_ml_edge_transductive_mrr

| method             |   patent_us |   patent_in |   patent_cn |   thgl_software |   aminer_paper |   amazon_movies |   dblp |
|:-------------------|------------:|------------:|------------:|----------------:|---------------:|----------------:|-------:|
| XGBoost            |      0.9927 |      0.9871 |      0.9945 |          0.1562 |         0.2255 |          0.7718 | 0.1717 |
| RandomForest       |      0.9722 |      0.7749 |      0.9151 |          0.0851 |         0.0603 |          0.57   | 0.07   |
| LogisticRegression |      0.6808 |      0.4019 |      0.5358 |          0.1509 |         0.4189 |          0.6895 | 0.6421 |
| TemporalCount      |      0.9957 |      0.9969 |      0.998  |          0.0872 |         0.0476 |          0.6167 | 0.0476 |

## tabular_ml_edge_transductive_auc

| method             |   patent_us |   patent_in |   patent_cn |   thgl_software |   aminer_paper |   amazon_movies |   dblp |
|:-------------------|------------:|------------:|------------:|----------------:|---------------:|----------------:|-------:|
| XGBoost            |      0.9967 |      0.9971 |      0.998  |          0.4896 |         0.6824 |          0.9678 | 0.5517 |
| RandomForest       |      0.9755 |      0.9329 |      0.9721 |          0.6803 |         0.671  |          0.9049 | 0.6745 |
| LogisticRegression |      0.6782 |      0.4088 |      0.5244 |          0.6794 |         0.8378 |          0.8794 | 0.9004 |
| TemporalCount      |      0.9988 |      0.9995 |      0.9997 |          0.6314 |         0.6666 |          0.8815 | 0.6666 |

## tabular_ml_edge_transductive_ap

| method             |   patent_us |   patent_in |   patent_cn |   thgl_software |   aminer_paper |   amazon_movies |   dblp |
|:-------------------|------------:|------------:|------------:|----------------:|---------------:|----------------:|-------:|
| XGBoost            |      0.9861 |      0.9732 |      0.9883 |          0.0916 |         0.0949 |          0.5581 | 0.0597 |
| RandomForest       |      0.9693 |      0.7729 |      0.9148 |          0.0742 |         0.0429 |          0.361  | 0.0473 |
| LogisticRegression |      0.6706 |      0.3409 |      0.4962 |          0.0845 |         0.1689 |          0.3715 | 0.2701 |
| TemporalCount      |      0.9946 |      0.9893 |      0.9955 |          0.0728 |         0.0356 |          0.4708 | 0.0356 |
