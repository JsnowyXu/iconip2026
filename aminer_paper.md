# aminer_paper Dataset Note

## Source Files
- `datasets/Aminer/AMiner-Paper.rar`
- `datasets/Aminer/AMiner-Author.zip`
- `datasets/Aminer/AMiner-Author2Paper.zip`
- `datasets/Aminer/AMiner-Coauthor.zip`

## Main Processing Steps
1. Read paper metadata from `AMiner-Paper.rar` and keep papers in the year window `2002-2012`.
2. Apply balanced per-year citation sampling with `max_edges=200000` for the `paper_cite_paper` relation.
3. Expand the active paper set with cited papers so the citation destination nodes are preserved.
4. Join `Author2Paper` to obtain `author_write_paper`.
5. Join author metadata from `AMiner-Author.zip` to enrich author node text features.
6. Join `AMiner-Coauthor.zip` and construct dynamic `author_coauthor_author` edges by paper year.
7. Add `paper_publish_venue` edges and build three node types: `paper`, `author`, `venue`.
8. Build 128-d node features:
   - structural degree/time features,
   - text-hash features from paper/author/venue text.
9. Build paper multi-label and multi-class labels from keyword-rule topic matching on paper text.
10. Convert to unified temporal `.pt` format and export baseline adapter files.

## Unified Graph Outputs
- Unified dataset: `data/processed/aminer_paper_unified.pt`
- Baseline adapters:
  - `data/interim/baseline_adapters/aminer_paper/tgn`
  - `data/interim/baseline_adapters/aminer_paper/tgb2`
- Baseline families currently wired for this dataset:
  - DHC main experiment
  - DyGLib custom TGN-style wrapper
  - DyGLib model zoo: `JODIE`, `DyRep`, `TGAT`, `CAWN`, `TCL`, `GraphMixer`, `DyGFormer`
  - TGB2 custom: `TGN`, `TGN-edges`
  - TGB2 official: `STHN`
  - RepeatMixer: `RepeatMixer.py`

## Unified Graph Statistics

| Item | Value |
| --- | ---: |
| Nodes | 265234 |
| Edges | 961282 |
| Node types | 3 |
| Relation types | 4 |
| Node feature dim | 128 |
| Edge feature dim | 2 |
| Train edges | 678061 |
| Val edges | 144171 |
| Test edges | 139050 |
| Time min | 0.0 |
| Time max | 2012.0 |
| Avg degree | 7.2486 |
| Max degree | 1462 |

## Node Types

| Node type id | Node type name | Count |
| --- | --- | ---: |
| 0 | paper | 127349 |
| 1 | author | 137450 |
| 2 | venue | 435 |

## Edge Types

| Relation id | Relation name | Count |
| --- | --- | ---: |
| 0 | paper_cite_paper | 200000 |
| 1 | author_write_paper | 49776 |
| 2 | paper_publish_venue | 17970 |
| 3 | author_coauthor_author | 693536 |

## Label Types
- Node task type:
  - paper-node multilabel classification
  - paper-node multiclass classification
- Label source:
  - `paper_text_keyword_rules`
- Labeled paper nodes:
  - multilabel: `16077`
  - multiclass: `16077`

## Multi-Label Names

| Label id | Label name | Positive count |
| --- | --- | ---: |
| 0 | ai_ml | 1969 |
| 1 | theory_algorithms | 6689 |
| 2 | db_ir_web | 4990 |
| 3 | systems_software | 4432 |
| 4 | networks_security | 4741 |
| 5 | vision_graphics_hci | 7711 |
| 6 | bio_med | 876 |
| 7 | robotics_control | 2598 |

## Additional Metadata
- Sampling policy: `year_window_with_balanced_edge_quota`
- Year window: `2002-2012`
- Per-year citation cap: `18182`
- Papers with sampled citation edges: `17995`
- Active papers: `127349`
- Active authors: `143716`
- Venue count: `435`
- Coauthor pairs found in source zip: `606828`
- Dynamic coauthor directed edges generated: `693536`

## Top Venues
- Expert Systems with Applications: An International Journal
- Computers and Operations Research
- Pattern Recognition
- Signal Processing
- Applied Soft Computing
- Information Sciences: an International Journal
- Proceedings of the 2004 ACM symposium on Applied computing
- IEEE Transactions on Pattern Analysis and Machine Intelligence
- Computational Statistics & Data Analysis
- Journal of Computational and Applied Mathematics
- Journal of Computing Sciences in Colleges
- Proceedings of the 2002 ACM symposium on Applied computing
- CHI '02 Extended Abstracts on Human Factors in Computing Systems
- Future Generation Computer Systems
- IEEE Transactions on Computers
- Computers & Education

## Task Views
- All task views preserve the full heterogenous context:
  - `source_rel_ids = [0, 1, 2, 3]`
- The supervised edges are restricted by `target_original_rel_id`.

| Task file | Task type | Target relation | Supervised edges | Train supervised | Val supervised | Test supervised |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `data/processed/paper_tasks/aminer_paper_citation_edge_transductive.pt` | edge_transductive | paper_cite_paper (`rel=0`) | 200000 | 145456 | 36364 | 18180 |
| `data/processed/paper_tasks/aminer_paper_citation_edge_inductive.pt` | edge_inductive | paper_cite_paper (`rel=0`) | 199999 | 147123 | 32866 | 20010 |
| `data/processed/paper_tasks/aminer_paper_coauthor_edge_transductive.pt` | edge_transductive | author_coauthor_author (`rel=3`) | 693536 | 568440 | 22234 | 102862 |
| `data/processed/paper_tasks/aminer_paper_coauthor_edge_inductive.pt` | edge_inductive | author_coauthor_author (`rel=3`) | 693535 | 460838 | 101361 | 131336 |
| `data/processed/paper_tasks/aminer_paper_node_multilabel.pt` | node_multilabel | all relations | 961282 | 678061 | 144171 | 139050 |

## Analysis Outputs
- Unified graph figures:
  - `data/reports/aminer_paper/degree_distribution.png`
  - `data/reports/aminer_paper/temporal_density.png`
  - `data/reports/aminer_paper/relation_distribution.png`
  - `data/reports/aminer_paper/node_type_distribution.png`
  - `data/reports/aminer_paper/community_sketch.png`
- Task-view figure directories:
  - `data/reports/aminer_paper_citation_edge_transductive`
  - `data/reports/aminer_paper_citation_edge_inductive`
  - `data/reports/aminer_paper_coauthor_edge_transductive`
  - `data/reports/aminer_paper_coauthor_edge_inductive`
  - `data/reports/aminer_paper_node_multilabel`
- Summary CSV:
  - `data/reports/dataset_stats_summary.csv`

## Interpretation Notes
- The AMiner unified graph is dominated by coauthor edges, so if the target relation is not explicitly fixed, the default “largest relation” heuristic would bias task generation toward coauthor.
- For this run, citation and coauthor tasks were explicitly separated by relation id to avoid that ambiguity.
- Node classification is defined on paper nodes, but the node view still replays the full heterogeneous temporal history.

## DHC Representation Usage Check
- `paper-paper` edge prediction does use other edge types as temporal context.
  - task-view construction keeps `source_rel_ids = [0, 1, 2, 3]` and only marks the target relation edges through `edge_label`
  - training loss is applied only on supervised target edges, but `model.update_state(...)` is executed on the full batched stream
  - this means citation/coauthor prediction can consume `author_write_paper`, `paper_publish_venue`, and the non-target target-type edges through temporal memory and recent-neighbor context
- Node multilabel classification already includes edge-derived information.
  - evaluation first calls `replay_history(...)` on the temporal edge stream
  - then `predict_node_heads(...)` uses `_node_repr(...)`, which fuses node memory, recent-neighbor context, and community context
- Current conclusion:
  - no immediate DHC model change is required for step 4
  - the critical fix for AMiner was task-view separation by relation, not disabling or re-adding cross-type context
