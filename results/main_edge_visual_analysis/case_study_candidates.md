# Case Study Candidates

These notes are generated from task metadata, relation counts, memory traces, ablations, and performance tables.

## aminer_paper
- Link prediction target: paper -> paper paper_cite_paper.
- Edge/relation semantics evidence: ablations did not reduce MRR in the available runs; largest gains vs full are no_rel__s42: +0.0361 MRR, no_edge_type_no_rel__s42: +0.0307 MRR, no_edge_type__s42: +0.0301 MRR.
- Performance contrast for evidence paragraph: tami=0.8297, tami=0.6026.
- Suggested narrative: citation links are predicted while author-write-paper, venue, and coauthor context act as semantic channels that move paper and relation memories over time.

## dblp
- Link prediction target: paper -> paper paper_cite_paper.
- Edge/relation semantics evidence: strongest negative effects vs full are no_edge_type__s42: -0.2904 MRR, no_rel__s42: -0.0711 MRR, no_edge_type_no_rel__s42: -0.0537 MRR.
- Performance contrast for evidence paragraph: tami=0.7952, tami=0.4785.
- Suggested narrative: paper-paper citation prediction can be framed as temporal topic diffusion under heterogeneous paper/venue context.

## patent-us
- Link prediction target: org -> codeIII org_codeIII.
- Edge/relation semantics evidence: strongest negative effects vs full are no_relation_memory__s42: -0.0003 MRR.
- Performance contrast for evidence paragraph: tami=0.6668, tami=0.5630.
- Suggested narrative: company/org memory is shaped by patent-org and org-coorg context before predicting org-codeIII links, so assignee-side collaboration or publication-like activity can be used as evidence for IPC-code affinity shifts.

## thgl_software
- Link prediction target: type_0 -> type_3 rel_6.
- Edge/relation semantics evidence: strongest negative effects vs full are no_community__s42: -0.0075 MRR, no_edge_type__s42: -0.0064 MRR, no_relation_memory__s42: -0.0004 MRR.
- Performance contrast for evidence paragraph: tami=0.8128, tami=0.8021.
- Suggested narrative: many software relation types provide a clean edge-semantics stress test; relation-memory trajectories reveal which edge slots drift toward the target link semantics.
