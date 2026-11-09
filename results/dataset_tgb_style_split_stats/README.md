# Train/validation/test split statistics

This directory reports how each formal edge-prediction dataset is split into training, validation and test sets for both inductive and transductive protocols, together with TGB2-style statistics computed separately on each split.

## Split construction

The split membership is not recomputed here; it is read directly from the processed task dictionary field `split`, whose keys are `train`, `val` and `test`. These split arrays contain edge indices in the processed temporal graph.

The transductive protocol is chronological: earlier edges form the training context, later edges form validation and test windows. The inductive protocol additionally applies task-specific unseen-node filtering or context construction, so its train/validation/test edge indices can differ from the transductive protocol. Actual edge counts can deviate from an exact 70/15/15 ratio because the preprocessing preserves task-view constraints such as timestamp windows, relation masks and dataset-specific filtering.

In the current processed files, `thgl_software` has identical split indices for inductive and transductive task views. The other five datasets do not: `dblp` and `aminer_paper_citation` keep the same total edge set but differ in split membership, while `patent_us`, `patent_in` and `patent_cn` differ in both selected edge set size and split membership.

For `thgl_software`, timestamps are read from the original TGB2.0 edgelist CSV to avoid `float32` precision loss in large Unix timestamps. Other datasets use the processed task-view timestamps.

## Dataset split summary

| Dataset | Protocol | Train edges | Val edges | Test edges | Train span | Val span | Test span |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dblp | inductive | 147,886 | 33,360 | 18,754 | 210-2022 | 2023-2025 | 1978-2026 |
| dblp | transductive | 146,099 | 36,810 | 17,091 | 210-2021 | 2022-2023 | 2024-2026 |
| aminer_paper_citation | inductive | 870,607 | 63,734 | 26,941 | 0-2010 | 2011-2012 | 2002-2012 |
| aminer_paper_citation | transductive | 857,834 | 69,911 | 33,537 | 0-2009 | 2010-2011 | 2012-2012 |
| patent_us | inductive | 863,605 | 61,642 | 131 | 1262620800-1577721600 | 1474300800-1577116800 | 1577721600-1577721600 |
| patent_us | transductive | 744,631 | 149,604 | 153,529 | 1262620800-1512403200 | 1513008000-1526918400 | 1527523200-1577721600 |
| patent_in | inductive | 935,384 | 77,977 | 61,148 | 1515081600-1730995200 | 1605196800-1730995200 | 1636646400-1730995200 |
| patent_in | transductive | 809,114 | 181,225 | 164,465 | 1515081600-1686240000 | 1686844800-1707408000 | 1708012800-1730995200 |
| patent_cn | inductive | 616,782 | 41,007 | 33,142 | 1262707200-1577750400 | 1419955200-1577750400 | 1465315200-1577750400 |
| patent_cn | transductive | 534,357 | 114,573 | 108,345 | 1262707200-1520553600 | 1520899200-1547164800 | 1547510400-1577750400 |
| thgl_software | inductive | 1,108,627 | 203,697 | 177,482 | 1704085280-1706097087 | 1706097089-1706448444 | 1706448450-1706763599 |
| thgl_software | transductive | 1,108,627 | 203,697 | 177,482 | 1704085280-1706097087 | 1706097089-1706448444 | 1706448450-1706763599 |

## CSV columns

| Column | Meaning and statistical procedure |
| --- | --- |
| `dataset` | Dataset name used in the formal experiments. |
| `protocol` | Edge-prediction protocol: `inductive` or `transductive`. |
| `split` | One of `train`, `val` or `test`. |
| `task_file` | Processed `.pt` task view from which edges and split indices are read. |
| `num_nodes` | Number of unique endpoint nodes appearing in this split. |
| `num_edges` | Number of temporal edges in this split. |
| `avg_degree_2e_over_n` | `2 * num_edges / num_nodes`, computed inside the split. |
| `num_edge_types` | Number of relation IDs appearing in this split. |
| `num_node_types` | Number of node types touched by split endpoint nodes. |
| `active_timesteps` | Number of timestamps in the split that contain at least one edge. |
| `time_unit` | Native timestep size inferred from the full dataset timestamp sequence and reused for all splits. |
| `time_source` | `original_edgelist_csv` for raw TGB2 timestamps, otherwise `task_view`. |
| `span_timesteps` | `(max(ts)-min(ts))/time_unit + 1` within the split, including empty timesteps. |
| `inductive_nodes_vs_train` | For `val` and `test`, the proportion of split nodes unseen in the training split: `|split_nodes - train_nodes| / |split_nodes|`. For `train`, this is `0`. |
| `drec` | Fraction of split edges `(s,r,o,t)` whose triple `(s,r,o)` also appears at `t - time_unit` in the full temporal graph. |
| `rec` | Fraction of split edges `(s,r,o,t)` whose triple `(s,r,o)` appears at any earlier timestamp `< t` in the full temporal graph. |
| `con` | Average maximum consecutive active run length across unique triples inside this split. |
| `mean_edges_per_ts` | `num_edges / span_timesteps` within the split. |
| `mean_nodes_per_ts` | Sum of active endpoint-node counts over observed timestamps divided by `span_timesteps`. |
| `ts_min` | Minimum timestamp in the split. |
| `ts_max` | Maximum timestamp in the split. |

## Reproduction

```bash
/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/compute_tgb_style_split_stats.py \
  --repo-root /home/xuxj/2026/dyn_relation_memory \
  --out-dir results/dataset_tgb_style_split_stats
```
