# Module Locations

## relation_memory

- `dhc/train/run_train.py`: CLI switch `--no-relation-memory` is parsed and passed into `DHCModel(use_relation_memory=...)`.
- `dhc/models/memory.py`: `NodeRelationMemory` stores node memory, relation memory, and temporal update state.
- `dhc/models/dhc_model.py`: `__init__(...)` wires relation memory in, and `_rel_vec(...)`, `_recent_neighbor_context(...)`, `forward_batch(...)`, `update_state(...)` consume it.

## edge_type

- `dhc/train/run_train.py`: CLI switch `--no-edge-type-features` is parsed and passed into `DHCModel(use_edge_type_features=...)`.
- `dhc/models/dhc_model.py`: `__init__(...)` expands raw edge feature width, and `_augment_edge_feat(...)` injects one-hot edge-type features into messages and recent-neighbor cache.

## community

- `dhc/train/run_train.py`: CLI switch `--no-community` is parsed and passed into `DHCModel(use_community=...)`.
- `dhc/models/community.py`: `CommunityMemory` implements soft assignment, memory update, and contrastive objective.
- `dhc/models/dhc_model.py`: `__init__(...)` constructs `CommunityMemory`, and `_node_repr(...)`, `_encode_link_query_nodes(...)`, `forward_batch(...)`, `update_state(...)` use community context and memory.

## neighbor_context

- `dhc/train/run_train.py`: `--recent-neighbor-size` and `--neighbor-aggregation` control local temporal neighbor context.
- `dhc/models/recent_neighbors.py`: `RecentNeighborStore` keeps recent interaction context.
- `dhc/models/dhc_model.py`: `_recent_neighbor_context(...)` supports `attn` and `mean` aggregation.
