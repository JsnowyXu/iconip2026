# Edge Semantics and Relation-Memory Case Study

This case-study package answers two questions: (1) whether different edge representations affect link-prediction accuracy, and (2) whether relation memory exhibits semantic evolution over training epochs.

## Evidence 1: Edge Representation Affects Accuracy

- `dblp_edge_transductive`: Full test MRR=0.5447; strongest degradation is `No edge type` with ΔMRR=-0.2904.
- `patent_us_edge_org_codeIII_transductive`: Full test MRR=0.9962; strongest degradation is `No relation memory` with ΔMRR=-0.0003.
- `thgl_software_edge_inductive`: Full test MRR=0.7971; strongest degradation is `No community` with ΔMRR=-0.0075.
- `aminer_paper_citation_edge_transductive`: Full test MRR=0.7369; no degradation in available ablations, and `No relation memory` improves ΔMRR=0.0361. Treat as a boundary condition rather than positive evidence for relation memory.
- `thgl_software_edge_transductive`: Full test MRR=0.7941; no degradation in available ablations, and `No edge type` improves ΔMRR=0.0297. Treat as a boundary condition rather than positive evidence for relation memory.

The strongest positive evidence is DBLP transductive: removing edge type drops MRR by about 0.2904 and removing relation memory drops MRR by about 0.0711. Patent-US is near saturation, so relation-memory ablation has little room to reduce MRR. AMiner and THGL include boundary cases where simpler variants can match or exceed Full.

## Evidence 2: Relation Memory Evolves Semantically

- `thgl_software_edge_inductive` target `rel_6` moves with path length=9.109, direct drift=3.044, norm change=1.094.
- `thgl_software_edge_transductive` target `rel_6` moves with path length=2.212, direct drift=0.819, norm change=-0.627.
- `aminer_paper_citation_edge_transductive` target `paper_cite_paper` moves with path length=1.806, direct drift=0.149, norm change=-0.482.
- `dblp_edge_inductive` target `paper_cite_paper` moves with path length=1.796, direct drift=0.791, norm change=-0.679.
- `aminer_paper_citation_edge_inductive` target `paper_cite_paper` moves with path length=1.078, direct drift=0.319, norm change=-0.532.
- `patent_us_edge_org_codeIII_transductive` target `org_codeIII` moves with path length=1.007, direct drift=0.611, norm change=-1.433.
- `dblp_edge_transductive` target `paper_cite_paper` moves with path length=0.979, direct drift=0.425, norm change=-0.475.
- `patent_us_edge_org_codeIII_inductive` target `org_codeIII` moves with path length=0.604, direct drift=0.498, norm change=-1.076.

For the paper case study, use `relation_type_trajectory.png` to show semantic movement in 2D, `target_relation_drift_vs_val_mrr.png` to connect movement with validation dynamics, and `edge_representation_delta_mrr.png` to quantify accuracy impact.

## Recommended Writing

A concise interpretation is: relation memory is not merely a static type embedding. It moves over epochs in the learned memory space; on DBLP, where the target paper-citation relation must be separated from author-write-paper context, disabling edge type or relation memory materially reduces MRR. On Patent-US, the org-codeIII task is almost saturated, so memory evolution is observable but ablation is not a strong accuracy discriminator. This gives both positive evidence and boundary conditions.
