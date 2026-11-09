# Edge Memory Ablation

This directory contains an earlier focused ablation on edge-memory and edge-type components. It is kept as development evidence and as a diagnostic companion to `results/best_config_module_ablation`, which is the ablation family used most directly in the ICONIP paper.

## Scope

- Completed metric files found: 14.
- Variants include `full`, `no_rel`, `no_edge_type`, and `no_edge_type_no_rel` depending on the task.
- Tasks include AMiner citation, DBLP, patent IN transductive, and TGB2.0 software transductive views.

## Result Index

| Run | MRR | AUC | AP |
|---|---:|---:|---:|
| `aminer_paper_citation_edge_transductive__full__s42` | 0.7369 | 0.9691 | 0.4733 |
| `aminer_paper_citation_edge_transductive__no_edge_type__s42` | 0.7670 | 0.9732 | 0.5087 |
| `aminer_paper_citation_edge_transductive__no_edge_type_no_rel__s42` | 0.7676 | 0.9731 | 0.5055 |
| `aminer_paper_citation_edge_transductive__no_rel__s42` | 0.7730 | 0.9739 | 0.5154 |
| `dblp_edge_transductive__full__s42` | 0.5447 | 0.9121 | 0.2850 |
| `dblp_edge_transductive__no_edge_type__s42` | 0.2543 | 0.5053 | 0.1230 |
| `dblp_edge_transductive__no_edge_type_no_rel__s42` | 0.4910 | 0.8818 | 0.2531 |
| `dblp_edge_transductive__no_rel__s42` | 0.4736 | 0.8773 | 0.2548 |
| `patent_in_edge_org_codeIII_transductive__full__s42` | 0.9949 | 0.9992 | 0.9823 |
| `patent_in_edge_org_codeIII_transductive__no_edge_type__s42` | 0.9911 | 0.9984 | 0.9709 |
| `patent_in_edge_org_codeIII_transductive__no_edge_type_no_rel__s42` | 0.9949 | 0.9992 | 0.9815 |
| `patent_in_edge_org_codeIII_transductive__no_rel__s42` | 0.9943 | 0.9991 | 0.9799 |
| `thgl_software_edge_transductive__full__s42` | 0.7941 | 0.9717 | 0.5953 |
| `thgl_software_edge_transductive__no_edge_type__s42` | 0.8039 | 0.9745 | 0.5935 |

## Interpretation

This ablation is useful for checking whether local edge-type features and relation-memory paths are responsible for gains on individual relation-specific tasks. It should not be read as the final module study because several runs were produced before the current best-configuration anchors were selected. The more controlled comparison is in `results/best_config_module_ablation`, where every task is rerun from the same selected anchor and all variants share the same evaluation surface.

In the paper narrative, this directory supports the development history: edge-type features and relation memory can help, but their effect depends strongly on the task protocol and on whether the relation-specific split is already close to saturation.
