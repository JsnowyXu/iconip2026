# Community Memory Case Study

This directory collects community-memory visualizations for the strongest patent and TGB2.0 software task views. Each subdirectory focuses on one trained run and exports monitor-node community assignments, community-size evolution, assignment churn, high-churn trajectories, and memory/vector plots.

## Summary

| Task | Test MRR | Test AP | Monitor nodes | Active communities | Largest share | Mean churn | Top-node changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `patent_cn_edge_org_codeIII_inductive` | 0.9956 | 0.9870 | 32 | 4 | 0.438 | 0.339 | 6.09 |
| `patent_cn_edge_org_codeIII_transductive` | 0.9955 | 0.9865 | 32 | 1 | 1.000 | 0.939 | 17.84 |
| `patent_in_edge_org_codeIII_inductive` | 0.9961 | 0.9893 | 32 | 1 | 1.000 | 0.627 | 11.91 |
| `patent_in_edge_org_codeIII_transductive` | 0.9956 | 0.9875 | 32 | 1 | 1.000 | 0.700 | 7.00 |
| `patent_us_edge_org_codeIII_inductive` | 0.9906 | 0.9816 | 32 | 1 | 1.000 | 0.039 | 0.31 |
| `patent_us_edge_org_codeIII_transductive` | 0.9963 | 0.9898 | 32 | 3 | 0.844 | 0.678 | 12.88 |
| `thgl_software_edge_inductive` | 0.8115 | 0.5872 | 96 | 6 | 0.510 | 0.836 | 15.88 |
| `thgl_software_edge_transductive` | 0.8009 | 0.5864 | 96 | 8 | 0.458 | 0.973 | 7.78 |

## Interpretation

The case studies show that high link-prediction performance can coexist with very different community dynamics. Patent CN inductive has four active monitor communities and moderate churn, suggesting a more distributed partition. Patent CN and IN transductive views collapse their monitor nodes into one final community while still exhibiting high epochwise churn, which points to unstable assignments before convergence. Patent US inductive is the most stable case: all monitor nodes remain in one final community and the average churn is close to zero. The software tasks use a larger monitor set and produce more active communities, but also higher churn, matching the short, dense, heterogeneous event stream in TGB2.0 software.

The paper uses these results to support a cautious interpretation of soft communities: they are useful as evolving mesoscopic memory, but their value should be judged together with assignment concentration and churn rather than only by final community counts.

## Paper Artifacts

- `ICONIP/tables/community_case_summary_table.tex` summarizes the quantitative case-study signals.
- `ICONIP/figures/community_case_mrr_churn.png` compares test MRR and assignment churn.
- Individual subdirectories contain the detailed plots used to select qualitative examples.
