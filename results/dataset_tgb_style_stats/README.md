# TGB2-style dataset statistics

This directory contains dataset statistics for the six edge-prediction datasets used in the formal experiments:

- `dblp`
- `aminer_paper_citation`
- `patent_us`
- `patent_in`
- `patent_cn`
- `thgl_software`

The statistics are computed by `scripts/datasets/compute_tgb_style_dataset_stats.py` on the chronological/transductive task views. This avoids mixing the engineered inductive-task filtering protocol into dataset-level properties.

Definitions follow the TGB2.0 dataset table:

- `Induct. Nodes`: proportion of nodes in the test split that are unseen in the training split.
- `DRec`: fraction of test temporal triples `(s,r,o,t)` that also appear at the immediately previous native timestep.
- `Rec`: fraction of test temporal triples `(s,r,o,t)` that appeared at any earlier timestep.
- `Con`: average maximum consecutive active run length across unique triples.
- `Mean Edges/Ts.` and `Mean Nodes/Ts.`: averages over the full temporal span at the inferred native time unit, including empty timesteps.

For `thgl_software`, timestamps are read from the original TGB2.0 edgelist CSV because the processed `.pt` stores large Unix timestamps as `float32`. For the other datasets, timestamps come from the processed task views and the native time unit is inferred from positive timestamp gaps.

## CSV columns

The file `dataset_tgb_style_stats.csv` contains one row per dataset. Let each temporal edge be represented as `(src, rel, dst, ts)`, and let a temporal triple be `(src, rel, dst)` without its timestamp.

| Column | Meaning | Statistical procedure |
| --- | --- | --- |
| `dataset` | Dataset name used in the experiments. | Manually mapped to the formal edge-prediction datasets. |
| `task_file` | Processed `.pt` task view used as the statistics source. | Uses the chronological/transductive task view for each dataset, so dataset-level statistics are not affected by inductive-filter construction. |
| `num_nodes` | Number of nodes in the processed heterogeneous graph. | Read from `data["num_nodes"]`; falls back to `max(src,dst)+1` if absent. |
| `num_edges` | Number of temporal edges. | Count of entries in `edges["src"]`, equivalent to the number of `(src, rel, dst, ts)` records. |
| `avg_degree_2e_over_n` | Average graph degree under the common undirected summary convention. | Computed as `2 * num_edges / num_nodes`. This is a dataset-scale density statistic, not relation-specific degree. |
| `num_edge_types` | Number of edge or relation types. | Read from `data["num_rels"]`; falls back to the number of unique relation IDs. |
| `num_node_types` | Number of node types in the heterogeneous graph. | Read from `data["num_node_types"]`. |
| `active_timesteps` | Number of timestamps that contain at least one edge. | Count of unique `ts` values after timestamp source selection. Empty timestamps inside the time span are not counted here. |
| `time_unit` | Native timestep size used for direct recurrence and span-based averages. | Inferred from positive timestamp gaps. Normally this is the greatest common divisor of gaps; for large Unix timestamps affected by `float32` quantization, the modal positive gap is used when it better reflects the task granularity. |
| `time_source` | Source of the timestamp values used in the statistics. | `original_edgelist_csv` means timestamps were read from the raw TGB2.0 edgelist; `task_view` means timestamps came from the processed `.pt` task file. |
| `span_timesteps` | Number of native timesteps in the full temporal span. | Computed as `(max(ts) - min(ts)) / time_unit + 1`. This includes empty timesteps between the first and last observed timestamp. |
| `inductive_test_nodes` | Proportion of test nodes unseen during training, matching TGB2.0's Induct. Test Nodes idea. | Build `train_nodes` from all train split edge endpoints and `test_nodes` from all test split edge endpoints, then compute `|test_nodes - train_nodes| / |test_nodes|`. |
| `drec` | Direct Recurrency Degree. | For each test edge `(s,r,o,t)`, check whether the same triple `(s,r,o)` appears at timestamp `t - time_unit`. Report the fraction of test edges satisfying this condition. |
| `rec` | Recurrency Degree. | For each test edge `(s,r,o,t)`, check whether the same triple `(s,r,o)` appears at any timestamp `< t`. Report the fraction of test edges satisfying this condition. |
| `con` | Consecutiveness Value. | For every unique triple `(s,r,o)`, sort its active timestamps and find the maximum run length where adjacent timestamps differ by exactly `time_unit`. Report the average of this maximum run length over all unique triples. |
| `mean_edges_per_ts` | Mean number of edges per timestep over the full temporal span. | Computed as `num_edges / span_timesteps`, including empty timesteps in the denominator. |
| `mean_nodes_per_ts` | Mean number of active nodes per timestep over the full temporal span. | For each observed timestamp, count unique endpoint nodes appearing at that timestamp; sum these counts and divide by `span_timesteps`, including empty timesteps in the denominator. |
| `train_edges` | Number of train split edges in the task view. | Count of indices in `split["train"]`. |
| `test_edges` | Number of test split edges in the task view. | Count of indices in `split["test"]`. |

## Reproduction

Run the following command from the repository root:

```bash
/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/compute_tgb_style_dataset_stats.py \
  --repo-root /home/xuxj/2026/dyn_relation_memory \
  --out-dir results/dataset_tgb_style_stats
```
