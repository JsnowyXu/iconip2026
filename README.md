# ICONIP DHC: 动态异构关系与社区记忆

独立研究归档与可运行复现项目。原始来源为 `dyn_relation_memory`，整理日期 2026-11-09。
本地归档包含原代码、全部已存在的处理数据、历史模型、实验报告和原始数据副本；GitHub 版本包含源码、配置、指标与文档，大体积数据通过离线副本提供。

## 快速开始

```bash
cd iconip-master
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
python tools/reproduce.py list
python tools/reproduce.py train --config repro/configs/patent_us_edge_org_codeIII_transductive__seed42.json
```

已验证的旧环境是 Python 3.8 / PyTorch 2.2.2 / PyG 2.6.1，依赖快照见 `repro/environment.txt`。如需 CUDA，先根据本机驱动安装对应 PyTorch 2.2.2，再安装其余依赖。AMiner RAR 解析需要系统命令 `unrar`；zip/csv 不需要该命令。基线各自的可选依赖见 `vendor/baselines` 内原 README 和 requirements。

## 复现层级

1. 使用本地完整副本：数据已在 `data/processed`，无需重新处理；按保存配置训练到 `results/reproduced/`。
2. 从原始数据重建：运行 `python tools/reproduce.py preprocess --dataset patent_us`，其他选项包括 patent_cn、patent_in、aminer_paper、dblp、thgl_software、thgl_github、amazon_movies。该命令覆盖该数据集的处理产物；需要保留旧数据时，请先复制项目。
3. GitHub 克隆：须把离线副本的 `raw_datasets/`、`data/`、所需 `results/*/best_model.pt` 放回相同相对路径；核对 `repro/artifact_manifest.json` 后运行。GitHub 本身不包含这些大文件。

完整步骤和验证边界见 [REPRODUCE.md](REPRODUCE.md)，架构、公式及实验结果见 [architechture.md](architechture.md)，Agent 工具使用见 [inference/README.md](inference/README.md)。

## 项目结构

| 路径 | 用途 |
|---|---|
| `dhc/` | 数据契约、模型、训练与评估 |
| `tools/reproduce.py` | 可迁移预处理与训练入口 |
| `scripts/datasets/raw_stats/` | 各原始数据统计 |
| `repro/configs/` | 从真实运行参数导出的独立配置 |
| `repro/results.csv` | 对应配置的历史测试指标 |
| `raw_datasets/` | 原始数据本地副本，Git 忽略 |
| `data/` | 统一图、任务视图、基线转换与统计 |
| `results/` | 原始全部历史实验及模型 |
| `vendor/baselines/` | 原项目使用的三套基线实现 |
| `ICONIP/` | 现有论文与表图 |
| `inference/` | Python API 和 JSON-lines Agent 适配器 |

## 必须理解的复现边界

- 论文目前有四数据集描述，也有 CN/IN/US/software 八任务消融。此归档保留全部，不混成一张“最优”主表。主 campaign 配置与历史 best-test-MRR catalog 配置在 JSON 中标明来源。历史 catalog 按 test MRR 选配置，存在测试集选择偏差。
- 旧 checkpoint 保存器用 startswith 匹配，既漏存 `memory.*` 的节点/关系 GRU，也误删已训练的 `node_type_emb.weight`。原环境按 seed 重建也无法精确恢复已训练类型 embedding；旧 checkpoint 仅作为历史产物保留，推理入口明确拒绝。新保存器仅排除动态状态且完整保存参数，必须重新训练后使用。
- 自定义 memory GRU 在 `@torch.no_grad()` 状态更新中没有梯度；本次未改变优化语义。历史指标仅作为原记录保留，未宣称整套实验已重训验证。
- THGL inductive 可能为 temporal fallback；静态特征包含全图结构统计，存在未来信息风险；评估使用 20 个全节点均匀负例，未滤除已知正边。详见架构文档。
- Agent 工具提供已有图词表内的关系边排序，不是聊天模型。新增实体须维护 ID/特征映射并重新构建模型；不能直接接收自然语言就预测。

GitHub 代码仓库：<https://github.com/JsnowyXu/iconip2026>。仓库根目录对应本地 `iconip-master/`。

审计额外发现：归档 Patent-US inductive 的 train/test 有 129 个重叠事件。新预处理按 test→validation→train 优先级排除重叠，并写入修复元数据；旧数据与指标保留。训练包装入口默认拒绝该旧划分。详见 `repro/patent_us_split_overlap.json`。
