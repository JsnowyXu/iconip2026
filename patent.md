# patent Dataset Note

## Main Processing Steps

### patent_us
1. Read the full raw CSV together with `clsIII-reInd.csv` and the IPC subclass table.
2. Split multi-valued `orgID` and `codeIIIId` fields, and align organization names when present.
3. Build a heterogeneous temporal graph with node types `patent`, `org`, and `codeIII`.
4. Add relation types `patent_org`, `patent_codeIII`, `org_codeIII`, and co-assignee `org_coorg` edges.
5. Initialize node text features from patent text, organization names, and IPC subclass descriptions.
6. Export unified `.pt`, task-view `.pt`, TGN-style adapters, and TGB2/STHN adapters.
7. Draw dataset analysis figures and save per-task statistics for reruns.

## Unified Graph Statistics

### patent_us

| Item | Value |
| --- | ---: |
| Nodes | 201321 |
| Edges | 1047764 |
| Node types | 3 |
| Relation types | 4 |
| Node feature dim | 128 |
| Edge feature dim | 4 |
| Labeled patents | 200192 |
| Multilabel classes | 609 |

| Node type id | Node type name | Count |
| --- | --- | ---: |
| 0 | patent | 200192 |
| 1 | org | 520 |
| 2 | codeIII | 609 |

| Relation id | Relation name | Count |
| --- | --- | ---: |
| 0 | patent_org | 202361 |
| 1 | patent_codeIII | 417982 |
| 2 | org_codeIII | 422825 |
| 3 | org_coorg | 4596 |

| Label id | Label name | Positive count |
| --- | --- | ---: |
| 0 | 0 / C25D | 525 |
| 1 | 1 / G21K | 277 |
| 2 | 2 / D21G | 4 |
| 3 | 3 / E01B | 8 |
| 4 | 4 / B09B | 22 |
| 5 | 5 / E05C | 76 |
| 6 | 6 / A44B | 43 |
| 7 | 7 / A23B | 32 |
| 8 | 8 / F41J | 10 |
| 9 | 9 / C06D | 7 |
| 10 | 10 / C10L | 453 |
| 11 | 11 / C01D | 38 |
| 12 | 12 / C12M | 393 |
| 13 | 14 / C05B | 12 |
| 14 | 15 / D21B | 8 |
| 15 | 16 / B60M | 33 |
| 16 | 17 / G01S | 1762 |
| 17 | 18 / E04H | 82 |
| 18 | 19 / F24B | 4 |
| 19 | 20 / A45C | 54 |
| 20 | 21 / B23B | 234 |
| 21 | 22 / D01G | 21 |
| 22 | 23 / E01D | 8 |
| 23 | 24 / A61Q | 1243 |
| 24 | 25 / B67D | 109 |
| 25 | 26 / G01D | 678 |
| 26 | 27 / F22G | 7 |
| 27 | 28 / D05C | 6 |
| 28 | 29 / B25G | 23 |
| 29 | 30 / A42B | 45 |
| ... | ... | 579 more labels omitted |

Task views:
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_us_edge_org_codeIII_transductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_us_edge_org_codeIII_inductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_us_node_patent_multilabel.pt`

Analysis outputs:
- `/home/xuxj/2026/dyn_relation_memory/data/reports/patent_us/degree_distribution.png`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/patent_us/temporal_density.png`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/patent_us/relation_distribution.png`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/patent_us/node_type_distribution.png`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/patent_us/community_sketch.png`

## DHC Representation Usage Check
- Edge prediction does use other edge types as temporal context.
- The patent task views keep the full heterogeneous stream and only restrict supervision with `edge_label` on `org_codeIII`.
- In `dhc/models/dhc_model.py`, link prediction consumes `rel_emb`, relation memory, edge features, and time encoding, not only node states.
- Node multilabel classification already replays the temporal edge stream before calling `predict_node_heads(...)`.
- `predict_node_heads(...)` uses `_node_repr(...)`, which fuses node memory, recent-neighbor context, relation-aware neighbor messages, and community context.
- Current conclusion: edge-type and edge-feature information is already used for both edge prediction and node classification; no immediate DHC patch is required before preprocessing and first-round experiments.
