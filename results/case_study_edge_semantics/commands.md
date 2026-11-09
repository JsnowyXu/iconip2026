# Reproduction Commands

Run the upstream visualization first if source CSV/PNG are stale:

`/home/xuxj/.conda/envs/pytorch/bin/python scripts/analysis/analyze_main_edge_visuals.py --project-root /home/xuxj/2026/dyn_relation_memory --datasets patent_us,aminer_paper,dblp,thgl_software`

Then regenerate this case-study package:

`/home/xuxj/.conda/envs/pytorch/bin/python scripts/analysis/build_edge_semantics_case_study.py --project-root /home/xuxj/2026/dyn_relation_memory --analysis-root /home/xuxj/2026/dyn_relation_memory/results/main_edge_visual_analysis --out-dir /home/xuxj/2026/dyn_relation_memory/results/case_study_edge_semantics`
