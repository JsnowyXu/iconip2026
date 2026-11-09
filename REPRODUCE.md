# 完整复现操作

## 1. 环境和数据完整性

使用 Python 3.8，安装 README 的依赖与本项目。`repro/environment.txt` 是原环境实际包快照，主 requirements 只保留主模型所需包。原环境额外安装 torch-scatter 2.1.2+pt22cu118、torch-sparse 0.6.18+pt22cu118，基线需要时应安装与 torch/CUDA 匹配的构建。不同 CUDA、GPU、线程和 PyTorch 版本无法保证逐位一致。运行前先执行：

```bash
python -m dhc.train.run_train --help
python tools/reproduce.py list
python tools/verify_artifacts.py
```

原始副本在 `raw_datasets`，模型数据在 `data/processed`；不能仅下载源码就宣称已经拥有数据。manifest 同时记录原始文件、处理图、实验产物和基线文件的大小与 SHA256。下载数据应遵守其原始数据许可；本归档不额外赋予数据或第三方基线的再分发权。

## 2. 原始统计

所有统计都从原始文件计算，不以任务子样本充当原始总量。保留结果在 `data/reports/raw_stats/`，可重新计算，例如：

```bash
python scripts/datasets/raw_stats/patent_us_raw_stats.py --dataset-dir raw_datasets/patent_us --json
python scripts/datasets/raw_stats/patent_cn_raw_stats.py --dataset-dir raw_datasets/patent_cn --json
python scripts/datasets/raw_stats/patent_in_raw_stats.py --dataset-dir raw_datasets/patent_in --json-output data/reports/raw_stats/patent_in_new.json
python scripts/datasets/raw_stats/dblp_raw_stats.py --input raw_datasets/DBLP-Citation-network-V18.zip --json
python scripts/datasets/raw_stats/amazon_movies_raw_stats.py --archive raw_datasets/Amazon_movies.zip --out data/reports/raw_stats/amazon_movies_new.json
python scripts/datasets/raw_stats/aminer_paper_raw_stats.py --paper-archive raw_datasets/Aminer/AMiner-Paper.rar --author-archive raw_datasets/Aminer/AMiner-Author.zip --author2paper-archive raw_datasets/Aminer/AMiner-Author2Paper.zip --coauthor-archive raw_datasets/Aminer/AMiner-Coauthor.zip --out data/reports/raw_stats/aminer_new.json
python scripts/datasets/raw_stats/thgl_software_raw_stats.py --edgelist-csv raw_datasets/tgb2/thgl-software/thgl-software_edgelist.csv --nodetype-csv raw_datasets/tgb2/thgl-software/thgl-software_nodetype.csv --edgemapping-csv raw_datasets/tgb2/thgl-software/thgl-software_edgemapping.csv --nodemapping-csv raw_datasets/tgb2/thgl-software/thgl-software_nodemapping.csv --out-json data/reports/raw_stats/software_new.json
```

大原始数据的全量统计可能很慢。DBLP 的 unique-cap 等上限影响统计精度，结果中有统计口径，不应把有上限计数当作真实唯一总数。

## 3. 处理与划分

```bash
python tools/reproduce.py preprocess --dataset patent_us
python tools/reproduce.py preprocess --dataset aminer_paper
python tools/reproduce.py preprocess --dataset dblp
python tools/reproduce.py preprocess --dataset thgl_software
```

另可对 CN/IN/Amazon/Github 运行同一入口。Patent 专用构建器生成四关系任务；不要用通用 patent parser 代替。AMiner 固定 2002–2012 与 200k 引用采样，DBLP/Amazon 固定 max-edges=200000；软件使用全量边。当前包装入口固定 AMiner/DBLP/software 目标关系 0/1/6。

每次重建后运行 `python tools/audit_data.py`，结果保存为 `repro/data_audit.json`。旧协议审计还可运行 `python scripts/datasets/audit_task_views.py --project-root .`。应同时核对目标边数量、全窗口数量、时间排序、split 索引交集与 `meta.extra.context_protocol` 中 unseen 列表。不要仅凭文件名确认 inductive 协议。原数据统计、处理统计和切分统计分别见 `data/reports` 与 `results/dataset_tgb_style*`。

## 4. 原配置训练和参数选择

```bash
python tools/reproduce.py train --config repro/configs/patent_us_edge_org_codeIII_transductive__seed42.json --gpu-id 0
```

全部训练参数来自原 run_args，覆盖维度、模块开关、优化器学习率、负采样比例、社区 temperature/topk、监控指标、早停、batch 和 seed。没有保存 run_args 的旧运行从 model_stats.args 恢复。输出写入 `results/reproduced/<config-name>`，不覆盖历史指标。`--epochs 1` 可用于链路检查，但不等于正式复现。

`repro/results.csv` 是每个导出配置的历史结果，`repro/configs/*.json` 给出原 run 路径和 best_epoch。历史 best-run catalog 部分按 test MRR 选择，不能作无偏泛化估计；正式新研究应仅基于 validation 选参，再一次性测试，并至少多 seed 统计均值和标准差。

## 5. 评估与图表

训练器自动在最佳 epoch 后 replay train+val 并计算 test。metrics.json、train_history.csv、best_model.pt 为主要产物。模型测试是在线批次协议：先预测，再用本批真实事件更新，后续批次可用更早测试事件。负采样数为 20；不是全图检索指标。改变负采样、batch 大小、划分或修正特征泄漏，都应标为新实验，不能直接复用历史表的数值。

历史比较表在 `results/edge_only_formal/reports/`，消融在 `results/best_config_module_ablation/`，敏感性在 `results/best_config_sensitivity/`。基线源码在 `vendor/baselines`。旧 campaign 的外部布局约定为 `<repo-root>/datasets` 和 `<repo-root>/baselines/src`，本项目通过 `tools/run_campaign.py` 建立隔离工作布局后调用；示例：

```bash
python tools/run_campaign.py --phase baselines --datasets patent_us,thgl_software,aminer_paper,dblp --gpu-ids 0 --max-parallel 1
```

每套基线先依其自带依赖安装；本次主模型验证不代表所有基线重新跑通。历史脚本、日志和论文内的绝对旧路径是溯源记录；实际执行使用 tools 中迁移后的入口。

## 6. 已验证与未验证

具体本次检查见 `repro/VALIDATION.md`。完整几十万节点多数据集训练不在归档步骤中自动重跑；历史记录与新 smoke 结果分开保存。模型状态 GRU 无梯度、静态结构全图计算、inductive 回退、测试集选优与采样域偏差都需在论文中披露。

## 7. GitHub 和离线大文件

`.gitignore` 忽略 raw_datasets、处理张量、缓存、环境、checkpoint、第三方大二进制，以及自动生成的大文件清单。它们仍实际存在于本地项目。源码、配置、指标、文档和离线文件 manifest 可提交。远端代码恢复完整本地项目时，按 manifest 路径补齐离线文件；本项目不编造不存在的公开数据下载地址。

`repro/best_validation.json` 另以相同任务历史 main/tuning 运行中保存的 best-epoch validation MRR 选出配置入口；这与原 best-test-MRR catalog 分开。历史搜索过程可能已经观察过测试结果，因此新增索引也不能消除历史搜索偏差。所有 main/tuning 的完整参数都导出到 configs，可自行核查而无需依赖一句“最优”。

也可用统一统计入口 `python tools/raw_statistics.py --dataset patent_us`；其余支持 aminer_paper、dblp、patent_cn、patent_in、amazon_movies、thgl_software、thgl_github。原始数据全量扫描不改变原始文件。

### 审计发现的归档异常

`data_audit.json` 中 Patent-US inductive 为失败：train/test 交集 129，旧 test 总量 131，监督目标 53。其他 40 个张量通过边界和索引互斥检查。旧数据完整保留；该失败是审计结果，不是检查工具失效。新预处理增加 eval 优先的去重并记录元数据；重新预处理后再审计。训练包装入口默认拒绝重叠划分；`--allow-historical-overlap` 仅供明确需要旧行为的诊断。互斥不等于全管线严格因果，时间窗口和静态特征问题仍需单独处理。
