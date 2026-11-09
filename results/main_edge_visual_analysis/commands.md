# Reproduction Commands

Run from `/home/xuxj/2026/dyn_relation_memory`.

All datasets:

`/home/xuxj/.conda/envs/pytorch/bin/python scripts/analysis/analyze_main_edge_visuals.py --project-root /home/xuxj/2026/dyn_relation_memory --datasets patent_us,aminer_paper,dblp,thgl_software`

Single dataset examples:

- `/home/xuxj/.conda/envs/pytorch/bin/python scripts/analysis/analyze_main_edge_visuals.py --project-root /home/xuxj/2026/dyn_relation_memory --datasets patent_us --out-root results/main_edge_visual_analysis_single_patent_us`
- `/home/xuxj/.conda/envs/pytorch/bin/python scripts/analysis/analyze_main_edge_visuals.py --project-root /home/xuxj/2026/dyn_relation_memory --datasets aminer_paper --out-root results/main_edge_visual_analysis_single_aminer_paper`
- `/home/xuxj/.conda/envs/pytorch/bin/python scripts/analysis/analyze_main_edge_visuals.py --project-root /home/xuxj/2026/dyn_relation_memory --datasets dblp --out-root results/main_edge_visual_analysis_single_dblp`
- `/home/xuxj/.conda/envs/pytorch/bin/python scripts/analysis/analyze_main_edge_visuals.py --project-root /home/xuxj/2026/dyn_relation_memory --datasets thgl_software --out-root results/main_edge_visual_analysis_single_thgl_software`
