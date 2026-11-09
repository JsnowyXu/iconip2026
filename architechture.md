# DHC 源码、数据与实验全流程讲解

> 本文档面向 `ICONIP/figures/2-framework.png` 的论文讲解与实验复现。所有“实际执行”描述以当前仓库源码和已保存 `.pt` 为准；论文文字只用于解释设计意图。证据采用“文件：函数/参数”的形式，便于随代码演化重新核对。

## 1. 一句话理解整个系统

DHC 把异构时序图统一成按时间排列的事件流 `(u,r,v,t,e)`。预测某条目标关系时，其他关系仍作为历史上下文进入状态更新，但不进入目标边损失。模型在看见当前事件之前，先用“关系感知的局部异构上下文分支”和“顺序事件记忆分支”表示查询节点，经门控融合后打分；随后才用真实事件更新节点、关系、近邻、顺序事件记忆和社区状态。

这对应框架图的 ①统一事件流、②节点初始化、③关系—时间消息、④动态状态、⑤DHC 局部分支、⑥事件记忆分支、⑦门控和链接预测、⑧按时间更新。核心因果约束是 **score before update**，而负样本只用于损失，不更新状态（证据：`dhc/models/dhc_model.py:DHCModel.forward_batch/update_state`，`dhc/train/trainer.py:Trainer.train`；论文：`ICONIP/texfiles/methodology.tex:Algorithm 1`）。

## 2. 代码地图与权威入口

| 阶段 | 入口 | 关键职责 |
|---|---|---|
| AMiner/DBLP 原始构建 | `scripts/datasets/prepare_dataset.py` | 解析、ID 映射、特征、排序、统一 `.pt` |
| Patent-US 原始构建 | `scripts/experiments/run_patent_preprocessing.py` | 专利专用四关系图、任务划分、保存与报告 |
| THGL-software 导入 | `scripts/datasets/import_tgb2_thg.py` | TGB2 CSV 到统一 `.pt` |
| AMiner/DBLP/THGL 任务视图 | `scripts/datasets/build_paper_task_views.py` | 目标关系、上下文流、transductive/inductive 视图 |
| Patent-US 任务视图 | `run_patent_preprocessing.py:build_task_views` | `org_codeIII` 的两种任务协议 |
| 数据加载 | `dhc/data/unified_dataset.py:load_unified_pt` | `.pt` 到 `UnifiedGraphData` |
| 时序批处理 | `dhc/data/batcher.py:TemporalBatcher/make_edge_batch` | 不打乱的批次和字段抽取 |
| 模型 | `dhc/models/dhc_model.py:DHCModel` | 双分支、打分、消息及状态提交 |
| 训练 | `dhc/train/run_train.py`、`dhc/train/trainer.py:Trainer` | 参数、负采样、损失、早停、产物 |
| 评估 | `dhc/eval/evaluator.py`、`dhc/utils/metrics.py` | 因果 replay、MRR/AUC/AP |
| 统一实验编排 | `scripts/experiments/run_edge_only_campaign.py` | 四数据集任务、DHC 与 baseline 队列 |

## 3. 统一数据契约

每个 `.pt` 是一个字典：

```text
num_nodes, num_rels, num_node_types
node_feat: [N,Dx]              node_type: [N]
edges.src/dst/ts/rel: [E]
edges.edge_feat: [E,De]        edges.edge_label: [E]
split.train/val/test: edge-index arrays
node_multiclass / node_multilabel（可选）
meta.dataset_name / meta.extra
```

`prepare_dataset.py:to_unified` 先按 `ts` 排序，再生成 `[normalized time, normalized relation id]` 两维边特征，并用 `temporal_split` 做 70/15/15；时间值重复导致分位点切分为空时，函数回退到按排序索引切分。Patent 专用管线的边特征为四维：归一化时间、关系、源节点类型、目标节点类型（`run_patent_preprocessing.py:build_unified_dataset`）。`load_unified_pt` 若缺少 `edge_label` 会默认为全 1。

任务视图中的 `edge_label` 不是二分类的原始真值：它是 **是否属于目标监督关系** 的掩码。目标边为 1，上下文边为 0；两者都会进入事件流和状态更新，只有 1 参与 BCE（`build_paper_task_views.py:build_edge_view`；`trainer.py:Trainer.train` 的 `supervised_mask`）。

当前落盘统计（直接读取四个 `data/processed/*_unified.pt`）：

| 数据集 | N | E | 关系 | 节点类型 | 节点特征 | 边特征 |
|---|---:|---:|---:|---:|---:|---:|
| Aminer-Paper | 265,234 | 961,282 | 4 | 3 | 128 | 2 |
| DBLP | 184,938 | 200,000 | 2 | 2 | 128 | 2 |
| Patent-US | 201,321 | 1,047,764 | 4 | 3 | 128 | 4 |
| THGL-software | 681,927 | 1,489,806 | 14 | 4 | 128 | 2 |

## 4. 四个数据集如何从原始数据构建

### 4.1 Aminer-Paper

原始入口是 `raw_datasets/Aminer/AMiner-Paper.rar`，并自动在同目录寻找 `AMiner-Author.zip`、`AMiner-Author2Paper.zip`、`AMiner-Coauthor.zip`（证据：`prepare_dataset.py:parse_aminer_paper`；实际命令见 `aminer_paper_process.md`）。

1. `iter_aminer_paper_records` 通过 `unrar p` 流式解析 `#index/#*/#@/#t/#c/#!/#%` 字段。
2. `parse_aminer_paper` 只保留 `--aminer-min-year 2002` 到 `--aminer-max-year 2012` 的源论文，并按年度均衡额度采样引用边；`--max-edges 200000` 限制的是引用边主体，不是最终异构图总边数。被引用论文也加入活动节点，避免目的节点丢失。
3. 全局字典将字符串键 `paper:* / author:* / venue:*` 连续映射到节点 ID。关系映射为：0 `paper_cite_paper`、1 `author_write_paper`、2 `paper_publish_venue`、3 `author_coauthor_author`；节点类型为 paper/author/venue。
4. `Author2Paper` 只保留活动论文，作者元数据补充姓名、机构和主题。共同作者边由论文作者组合按论文年份动态生成，并与 coauthor 源文件中的合法 pair 交叉校验；每个 pair/year 去重，然后生成双向边。
5. `build_degree_features(...,dim=64)` 生成入度、出度、总度、首次/末次时间、时间跨度及类型 one-hot，经标准化和补零。随后 `append_text_features` 加 64 维文本特征，默认 `hybrid_hash`，所以最终 128 维。
6. 论文标签由八组关键词规则产生，可多标签；无任何主题命中的论文不进入该标签表。这里是派生标签，不是 AMiner 官方人工标签（`parse_aminer_paper:topic_rules`）。
7. `to_unified` 全流排序并保存 `data/processed/aminer_paper_unified.pt`。

复现：

```bash
python scripts/datasets/prepare_dataset.py \
  --dataset aminer_paper \
  --archive raw_datasets/Aminer/AMiner-Paper.rar \
  --out-pt data/processed/aminer_paper_unified.pt \
  --max-edges 200000 --aminer-min-year 2002 --aminer-max-year 2012 \
  --export-baselines --baseline-out-dir data/interim/baseline_adapters/aminer_paper
```

论文主表实际用 citation 任务，必须显式 `--target-rel 0`；否则 `choose_target_relation` 会选择边数最多的关系，AMiner 会落到 coauthor：

```bash
python scripts/datasets/build_paper_task_views.py \
  --input data/processed/aminer_paper_unified.pt \
  --out-dir data/processed/paper_tasks \
  --name-prefix aminer_paper_citation --target-rel 0
```

### 4.2 DBLP

入口 `raw_datasets/DBLP-Citation-network-V18.zip` 内的 `DBLP-Citation-network-V18.jsonl`（`prepare_dataset.py:parse_dblp`）。每条 JSONL 建 paper 节点；作者建 `author_write_paper=0`，references 建 `paper_cite_paper=1`，时间均取源论文 year。ID 采用 `paper:* / author:*` 全局连续映射，节点类型为 paper=0、author=1。论文文本由 title、venue、year 合并，作者用 name；结构 64 维加默认哈希文本 64 维。`--max-edges 200000` 在逐条添加作者/引用边时截断，因此是“读到前 200k 事件”，不是分层均衡采样。缺少 `relation_map` 元数据是当前实现事实。

```bash
python scripts/datasets/prepare_dataset.py --dataset dblp \
  --archive raw_datasets/DBLP-Citation-network-V18.zip \
  --max-edges 200000 --out-pt data/processed/dblp_unified.pt \
  --export-baselines --baseline-out-dir data/interim/baseline_adapters/dblp
python scripts/datasets/build_paper_task_views.py \
  --input data/processed/dblp_unified.pt --out-dir data/processed/paper_tasks \
  --name-prefix dblp
```

未给 `--target-rel` 时选最多关系；当前已保存 inductive 视图的目标是 relation 1，即 citation。为避免数据版本改变后静默换任务，正式复现建议显式 `--target-rel 1`。

### 4.3 Patent-US

论文主实验不是通用 `prepare_dataset.py:parse_raw_patent` 的三关系版本，而是 `scripts/experiments/run_patent_preprocessing.py` 的四关系专用版本。入口目录 `raw_datasets/patent_us/`：`all-us.csv`、`IPCSubClass_US.csv`、`clsIII-reInd.csv`（`DATASET_SPECS`）。

1. 解析 patentId、title、summary、publicationDate、orgID、orgname、codeIIIId；多值字段支持 Python list 和分号文本；缺 org 或合法 code 的行跳过。
2. `clsIII-reInd.csv` 将重编号 code 与 Code-III 元数据对应；IPC 表补分类描述。
3. 三类节点为 patent=0、org=1、codeIII=2。四类边为 `patent_org=0`、`patent_codeIII=1`、`org_codeIII=2`、双向 `org_coorg=3`，同一专利中组织与技术 code 做笛卡尔积，组织 pair 生成双向合作关系。
4. 边按 `(timestamp, original order)` 稳定排序。节点特征由字符 2--4 gram TF-IDF（`feature_dim-8`）加八个结构特征组成；当前 `feature_dim=128`。专利节点的多标签是真实派生自其 Code-III 集合。
5. 目标任务是组织到 Code-III 的趋势边 `org_codeIII`，而不是 patent-to-code。

```bash
python scripts/experiments/run_patent_preprocessing.py \
  --project-root . \
  --dataset-root raw_datasets
```

具体参数请以该脚本 `main` 的 `--help` 为准。产物是 `data/processed/patent_us_unified.pt` 和 `data/processed/tasks/patent_us_edge_org_codeIII_{transductive,inductive}.pt`。Patent 构建器在统一图的 `meta.extra.task_splits` 同时保存原始 edge index 和 unseen organization ID，方便审计。

### 4.4 THGL-software

入口为 `raw_datasets/tgb2/thgl-software/thgl-software_edgelist.csv` 和 `thgl-software_nodetype.csv`。`import_tgb2_thg.py:load_thg_dataset` 要求边列 `timestamp,head,tail,relation_type`、节点列 `node_id,type`。

边先用 mergesort 按时间排序；`--max-edges -1` 表示全量。head/tail 被联合重映射为连续 ID；relation 也按排序后的唯一值映射为连续 ID。nodetype 找不到的节点记 type 0，并将数量写入元数据。特征只有 128 维结构特征，**没有文本编码**；node multiclass 直接等于 node type，multilabel 是度数/早出现时间的合成标签（`build_default_multilabel`），不应表述为真实软件语义标签。

```bash
python scripts/datasets/import_tgb2_thg.py \
  --dataset-name thgl_software \
  --edgelist-csv raw_datasets/tgb2/thgl-software/thgl-software_edgelist.csv \
  --nodetype-csv raw_datasets/tgb2/thgl-software/thgl-software_nodetype.csv \
  --out-pt data/processed/thgl_software_unified.pt --max-edges -1
python scripts/datasets/build_paper_task_views.py \
  --input data/processed/thgl_software_unified.pt \
  --out-dir data/processed/tgb2_tasks --name-prefix thgl_software
```

未显式目标关系时，任务选择事件数最多的 relation；当前日志为 `rel_6`。正式复现同样建议加 `--target-rel 6`。

## 5. Transductive 与 inductive：必须逐层区分

### 5.1 目标节点/边的可见性

| 项目 | Transductive | Generic inductive（AMiner/DBLP/THGL） | Patent-US inductive |
|---|---|---|---|
| 目标边划分 | 目标关系按时间 70/15/15 | 固定 seed=42 随机选择约覆盖 10% 目标边的 source 节点作 unseen；其目标边为 test | org 按首次出现时间排序，70/15/余下分成 train/val/test org |
| 节点可重叠 | train/val/test 可共享节点 | unseen source 不应出现在训练/验证上下文 | val/test org 均列入 unseen，训练上下文排除它们 |
| 辅助关系 | 全异构上下文保留 | 所有触及 unseen 节点的上下文边通过 `keep_mask` 排除 | 训练窗口排除 unseen；当前代码的 val/test 窗口没有同样过滤 |
| 监督 | 仅目标关系 `edge_label=1` | 同左 | 同左 |

Generic 构造证据为 `build_paper_task_views.py:inductive_partition_target/build`。它按 source 留出，而不是 source 与 destination 同时留出；`keep_mask` 才在所有关系上同时检查 src/dst，从而防止辅助边泄露 unseen source。

Patent 的划分语义略不同：validation org 也属于 unseen，因而它更接近“按实体冷启动的 train/val/test”；但 `build_task_views` 仅对 `ind_tr_old` 应用 `~involve_unseen`，val/test 是时间窗口中的所有关系。因此不能笼统声称每个 val/test 事件都只属于对应 held-out org。

### 5.2 时间窗口与上下文边

`build_edge_view` 先根据目标训练和验证的最后时间得到 cutoff，再把整个异构流分到 train/val/test 窗口；随后把显式 inductive val/test 目标边强制移动到相应集合。最终又按时间排序并重编号 edge index。于是 split 数量不是目标边数量：它包含大量 `edge_label=0` 上下文边。

训练中每个批次均执行 `forward_batch` 和 `update_state`；BCE 只索引 `supervised_mask`。所以辅助关系可以改变节点、关系、近邻、TGN、社区状态，正是图中“context edges history before t”的落地方式。

### 5.3 历史与内存状态

两种任务用的是同一模型和训练循环，没有专门的 inductive embedding 初始化器。节点 ID、静态 `node_feat`、`node_type` 和模型参数空间仍覆盖整个统一图；“未见”严格指 **训练事件历史中未见**，不代表静态特征或 ID 被从文件删除。

每轮训练开始 `model.reset_state()`，然后按 `data.train_idx`、`shuffle=False` 重放；每个 batch 先预测、反传，再 `update_state`。验证先 reset，然后只 replay train；测试先 reset，再 replay train+val。评估 batch 自身在打分后也更新状态，因此同一验证/测试流中较晚事件可利用更早的评估事件，这是标准 online temporal evaluation，而不是所有测试边彼此隔离（`evaluator.py:replay_history/eval_link_prediction`）。

### 5.4 负采样的相同与不同

代码没有按 task type 改变采样算法。训练每条事件先产生一个 negative destination；但损失只在 target mask 上计算。`Trainer._sample_negative_dst` 混合：与正目的节点同 type 的 typed pool、按训练监督目标目的节点频率构造的 popularity pool、全节点 uniform pool。默认比例 0.6/0.2/0.2，碰巧等于正目的节点时再抽一次，仍相同则 `(dst+1)%N`。

统一 campaign 会按数据集改比例：AMiner/DBLP 与 THGL-software 配置可在 `run_edge_only_campaign.py:main_job_profile` 核对。它不是 transductive/inductive 专属配置。

评估与训练 sampler 不一致：`eval_link_prediction` 对每个正目标边抽 20 个 **全节点均匀** negative destination，既不约束 node type，也不排除历史真边；负样本只打分，不更新状态，真实 batch 全部更新。这一点会让跨类型的负例偏容易，也是解释 MRR 时必须披露的限制。

### 5.5 回退机制与“伪 inductive”风险

Generic `inductive_partition_target` 在未见节点留出后监督不足时回退纯 temporal split，并清空 unseen；若 validation 太小也会如此。当前 `thgl_software_process.md` 明确记录 2026-04-22 重建后 inductive 和 transductive 均为 train 1,108,627 / val 203,697 / test 177,482；当前保存文件也应通过下面审计，而不能仅凭文件名叫它 inductive：

```bash
python scripts/datasets/audit_task_views.py --help
python - <<'PY'
import torch
p='data/processed/tgb2_tasks/thgl_software_edge_inductive.pt'
o=torch.load(p,map_location='cpu')
print(o['meta']['extra']['context_protocol'])
PY
```

若 `unseen_source_node_ids=[]`，它在实体可见性上就是 temporal/transductive fallback。论文当前实验段落无条件声称 inductive 留出 unseen source，未披露该回退，这是代码—论文不一致。

## 6. 从框架图逐框讲模型

### ① 输入

`load_unified_pt` 把全图张量加载到设备。任务视图已把 target/context 角色编码进 `edge_label`，模型本身不硬编码数据集关系名。

### ② Text-aware node initialization

`DHCModel._base_node_repr` 拼接 `node_feat`、learned node-type embedding 和当前 node memory，送入两层 MLP。注意“text-aware”只对 AMiner/DBLP/Patent 成立：THGL 的 `node_feat` 是纯结构量。

### ③ Relation-time message

`_augment_edge_feat` 可将 relation one-hot 拼入原边特征。`forward_batch` 构造反向对称的 raw message：对端 memory、本端 memory、增强边特征、相对上次更新时间的 sinusoidal encoding。`RelationTimeMessage` 用 relation vector 生成逐维 sigmoid gate 和 bias；relation vector 是 learned relation embedding 加投影后的 dynamic relation memory。

### ④ 四类状态

- `NodeRelationMemory`：节点和关系分别用 GRUCell 更新；同批次多条消息由 `latest_message` 只提交最后时间消息。
- `RecentNeighborStore`：每个节点保存最近 K 条双向交互，包括 relation、timestamp、event ID 和 edge feature。
- `SafeTGNMemory`：顺序事件记忆的源码实现类；论文中统一称为 sequential event-memory branch。
- `CommunityMemory`：软分配到 community prototype，社区 memory 带时间指数衰减后 GRU 更新；还有 contrastive loss 和可选 entropy regularization。

图中 4.4 “Sequential event memory”在源码中对应变量 `tgn_memory`，但论文表述统一使用 sequential event-memory；它不应与自定义 node memory 混为一谈。

### ⑤ DHC context branch

`_encode_link_query_nodes` 从 recent store 为查询节点取局部子图。edge attribute 拼接 dynamic relation vector、历史增强边特征和时间编码，由 PyG `TransformerConv` 编码，再与 base representation 融合；之后可再融合 community context。另一个 `_recent_neighbor_context` 是节点任务所用的 attention/mean 聚合路径。

### ⑥ Sequential event-memory branch

`self.tgn_memory(query_nodes)` 读取顺序事件记忆状态。观察事件在最终 `update_state` 中用真实 src/dst 和增强边特征提交；negative destination 永远不提交。

### ⑦ 门控与打分

`edge_gate([dhc_query_h,tgn_query_h])` 输出逐维 gate：`gate*TGN + (1-gate)*DHC`。链接 MLP 输入融合后的 src/dst、dynamic relation vector、source time gap encoding，输出 logit。

### ⑧ 状态更新

更新顺序为 custom node memory → recent-neighbor → sequential event-memory → relation memory → community memory。全部基于观测正事件，包括 `edge_label=0` 的上下文事件。

## 7. 训练、早停与评估

最小运行：

```bash
PYTHONPATH=. \
python -m dhc.train.run_train \
  --data data/processed/paper_tasks/aminer_paper_citation_edge_transductive.pt \
  --out-dir results/example_aminer_trans --task-type edge_transductive \
  --epochs 20 --batch-size 256 --lr 1e-3 \
  --monitor-metric val_mrr --early-stop-patience 5 \
  --save-memory-artifacts
```

默认维度来自 `run_train.py:parse_args`：memory 128、relation 64、time 32、message 128、community 128、64 communities、recent K=10。默认启用 recent neighbor、edge type one-hot、relation memory 和 community。训练总损失为目标边 BCE + `lambda_cl` community contrastive + 可选 entropy；edge task 不计算 node head loss。关系 loss weight 支持 `--relation-loss-weights`。

早停监控可自动选择，也可明确 `val_mrr`；原保存器意图过滤动态 buffer，但实际上还误删部分参数（见第14节）；新项目已修正 lightweight checkpoint（`Trainer._checkpoint_state_dict`），加载后评估靠 replay 重建状态。测试历史为 train+val。

指标：

- MRR：`rank=1+#(negative_score >= positive_score)`，再平均倒数；20 negatives 时随机同分基线约为 1/21。
- AUC、AP、accuracy：把每个正边和 20 个负目的边摊平成二分类样本。
- 节点任务另有 multiclass accuracy/macro-F1，以及 multilabel AUC、micro/macro-F1、mAP、P/R/NDCG@1/3/5。

注意：当前 edge task 的主训练/测试固定 `num_neg=20`，命令行没有 DHC 的 `--eval-num-negs`；campaign 的同名参数主要传给 baseline。因此改变 campaign 参数不会改变 DHC evaluator 的 20。

## 8. 批量实验与产物

```bash
python scripts/experiments/run_edge_only_campaign.py \
  --project-root . \
  --repo-root .campaign_workspace \
  --python-bin python \
  --phase main --datasets patent_us,thgl_software,aminer_paper,dblp \
  --gpu-ids 0 --max-parallel 1 --no-skip-existing
```

`edge_benchmark_common.py:build_edge_task_specs` 决定论文任务文件：AMiner citation、DBLP dominant/citation、Patent org-CodeIII、THGL dominant relation，各自两种模式。`run_edge_only_campaign.py:main_job_profile` 添加数据集专用超参并调用 `python -m dhc.train.run_train`。

每个 DHC run 目录至少包含：

- `best_model.pt`：最佳 lightweight 参数；
- `metrics.json`：最佳 epoch、test MRR/AUC/AP、关系拆分、每 epoch 历史等；
- `train_history.csv`：画曲线用；
- 启用保存后有 `memory_trace.pt`（逐 epoch 监控节点/关系/社区快照）和 `memory_final_state.pt`（train+val replay 后状态）。

campaign 总目录为 `results/edge_only_formal/{main,baselines,reports}`，并生成 manifest/汇总 CSV；baseline adapter 由 `scripts/datasets/export_baseline_formats.py` 或预处理器导出。AMiner/DBLP 通用预处理还直接输出 TGN CSV/feature NPY 和 TGB2-like edgelist/nodetype CSV（`prepare_dataset.py:export_tgn_format/export_tgb2_like_csv`）。

## 9. 代码与论文当前不一致/必须披露的边界

1. **THGL inductive 回退**：论文称 inductive 总会留出 unseen source；当前 THGL-software 已记录为 temporal fallback，两个 split 相同。应在结果表脚注披露或重构真正的冷启动划分后重跑。
2. **Inductive 仍暴露静态特征和全局 ID**：训练事件不可见，不等于节点完全未知。论文宜称 unseen-in-history / cold-start interaction，而非无任何节点信息。
3. **评估负例不保类型且不做 false-negative filtering**：论文只说 uniform 20 negatives，源码确实如此，但这可能显著抬高异构图分数，应补充限制。
4. **Patent val/test 上下文过滤不完全对称**：训练窗口排除 unseen org，validation/test 时间窗口本身未过滤所有非目标事件；论文“context edges touching unseen nodes are removed from training and validation”与 Patent 专用实现并不完全一致。
5. **“text-aware”非四数据集通用**：THGL 无文本；DBLP/AMiner 是 hash 特征，Patent 是 char TF-IDF。不要暗示使用统一预训练语言模型。
6. **DBLP 截断有顺序偏差**：200k 是 JSONL 顺序的前缀事件，不是均衡抽样；AMiner 则有按年 citation 配额，二者采样含义不同。
7. **目标关系默认选择有漂移风险**：Generic builder 默认最大关系，论文所需 AMiner citation 必须显式 relation 0；DBLP 和 THGL 也建议显式指定。
8. **图中 DHC branch 命名可能混淆**：实际模型除 relation/local/community 分支外，还显式融合 sequential event-memory；community 是源码重要模块，但图 ⑤ 没有单独突出，讲解时应补充。
9. **当前 `.pt` 与论文 split 比例不能只看数量**：离散时间、上下文窗口、显式 held-out 边移动都会令最终窗口偏离整图 70/15/15；应报告 target supervised counts 与 window counts 两套数字。

## 10. 推荐的严格复现实验检查单

1. 固定四个原始文件路径、checksum、Python 环境和 seed；不要把旧 `prepare_all_datasets.sh` 当权威入口，它的默认项目名及候选数据集与当前 CLI 已不完全同步。
2. 显式固定 target relation：AMiner 0、DBLP 当前 1、THGL 当前 6、Patent 专用 `org_codeIII=2`。
3. 每次重建后检查 `meta.extra.context_protocol`：目标关系、unseen ID 数、target supervised/window edge 数。
4. 验证 train/val/test edge index 不重叠、各自时间有序；对 inductive 检查 train 中任何关系都不触及 unseen。
5. 若 unseen 为空，停止将该文件报告为实体 inductive；调整留出策略而不是依赖文件名。
6. 记录训练/评估负采样域。若改为 type-aware evaluation，所有方法必须统一重跑，不能与当前表直接比较。
7. 用至少多个 seeds 汇报均值与方差；当前主 campaign 默认 seed 42 是单次运行配置。
8. 从 `metrics.json` 而非日志尾行取最终指标，同时保留 `best_epoch`、monitor metric 和 early-stop patience。

## 11. 一段可直接用于汇报的讲稿主线

“我们先把论文、专利和软件数据都变成统一的异构时序事件。对某一目标关系预测时，监督只落在目标边，其他关系仍作为因果历史更新状态。每个时刻，DHC 在事件发生前读取节点静态/文本属性、类型和动态节点状态，用动态关系状态调制时间消息，并从每个查询节点的最近 K 个异构事件构造局部子图；并行的 TGN 事件记忆保留紧凑的顺序历史。逐维门控根据查询自适应融合两路证据，再结合关系和时间差打分。训练使用一个混合负目的节点，评估用二十个均匀负例；真实事件在打分之后才更新全部动态状态。Transductive 只按时间切目标边；inductive 还应把 held-out source 的所有训练上下文删除。但当前 THGL-software 触发了时间回退，所以它不能被当作严格未见节点结果，这也是源码审计发现、需要在论文中透明说明或重跑的限制。”

## 12. 数学形式化：逐模块对应实际代码

本节以源码为准。记事件为 $e_i=(u_i,r_i,v_i,t_i,x_i)$，节点静态特征为 $X_u$，节点类型为 $\tau_u$；训练目标掩码 $y_i^{sup}=1$ 表示目标关系，0 表示上下文。以下状态均是当前批次更新之前的历史状态。代码按批次预测与提交，批内事件不会依次互相读取更新后的状态；因此不应把它描述为严格逐事件更新。

### 节点与关系表示

`dhc/models/dhc_model.py:_base_node_repr/_rel_vec`：

$$h_u^0=\operatorname{MLP}_x([X_u\Vert E_{type}(\tau_u)\Vert M_u]),\quad \rho_r=E_r+W_R R_r.$$

关闭关系记忆时仅保留 $E_r$。$M_u,R_r$ 分别为节点与关系 GRU 状态；node type embedding 为 16 维，默认 memory 128、relation 64。时间编码 (`time_encoding.py`) 为：

$$\phi(\Delta t)=[\sin(\Delta t/f_j)\Vert\cos(\Delta t/f_j)]_j,\quad f_j=\exp(\operatorname{linspace}(0,\log 10^4,d_t/2)).$$

$\Delta t=\max(0,t-last_u)$。边类型开关开启时 $\tilde x_i=[x_i\Vert\operatorname{onehot}(r_i)]$。

### 关系调制消息与局部状态

`relation_message.py` 的实际函数是仿射基底乘 sigmoid gate，再加关系偏置：

$$a_{u\leftarrow v}=[M_v\Vert M_u\Vert\tilde x_i\Vert\phi(t_i-last_u)],$$
$$m_{u\leftarrow v}=(W_ba_{u\leftarrow v}+b_b)\odot\sigma(W_g\rho_r+b_g)+(W_c\rho_r+b_c).$$

目的节点对称交换 u,v。`memory.py` 每批每个节点只保留时间最大的消息（同时间并列依源码排序/scatter 次序），更新为：

$$M_u^+=\operatorname{GRU}_N(W_N m_{u\leftarrow v},M_u),$$
$$R_r^+=\operatorname{GRU}_R\left(W_{rel}(m_{u\leftarrow v}+m_{v\leftarrow u})/2,R_r\right).$$

关系同样使用批内最后消息。关键实现限制：两个 update 函数都被 `@torch.no_grad()` 修饰，消息缓存也 detach，因此上述自定义 GRU updater 参数不会由此路径获得梯度；这是原代码事实，不能声称所有动态更新器均端到端学习。保留该行为以避免改变历史模型定义。

### 最近邻局部异构图

`recent_neighbors.py` 维护每个节点最近 K 个双向事件。查询取局部节点集合，历史边属性为：

$$a_{uv}^{hist}=[\rho_{r_{uv}}\Vert\tilde x_{uv}\Vert\phi(\Delta t_{uv})].$$

`TransformerConv` 在局部图上作双头注意力消息传播，可简写为：

$$\alpha_{uv}=\operatorname{softmax}_{v\in\mathcal N_K(u)}\frac{(W_Qh_u)^\top(W_Kh_v+W_Ea_{uv}^{hist})}{\sqrt d},$$
$$z_u=W_{root}h_u+\sum_v\alpha_{uv}(W_Vh_v+W_Ea_{uv}^{hist}).$$

这表示 TransformerConv 的结构形式，具体线性层、双头拼接和自环/root 规则以所锁定 PyG 实现为准。随后 `local_graph_fusion` 融合 $[h_u^0\Vert z_u]$，再按开关读取社区上下文。

### 社区分配、衰减记忆和正则项

`community.py`：可学习 prototype 为 $P_c$，维度 $d_c$。

$$q_u=W_qh_u,\quad k_c=W_pP_c,\quad \alpha_{uc}=\operatorname{softmax}_c(q_u^\top k_c/(\sqrt{d_c}T)).$$

启用 top-k 时，在 softmax 前屏蔽其余社区 logits。社区读出与融合：

$$c_u=\sum_c\alpha_{uc}C_c,\quad h_u^D=W_o[h_u\Vert c_u]+b_o.$$

写入当前批次 active nodes（由 src/dst 唯一化得到）的社区加权均值：

$$\bar m_c=\frac{\sum_u\alpha_{uc}W_{in}h_u}{\max(\sum_u\alpha_{uc},10^{-6})},\quad C_c^+=\operatorname{GRU}_C(\bar m_c,e^{-\max(0,t_{max}-last_c)/1000}C_c).$$

社区写入也处于 no_grad；该社区 GRU updater 同样未通过状态写入训练。对比正则取 $c_u^*=\arg\max_c\alpha_{uc}$，令 $\hat q,\hat C$ 为 L2 归一化向量，温度默认 0.2：

$$\mathcal L_{CL}=-\frac1{|U|}\sum_u\log\frac{\exp(\hat q_u^\top\hat C_{c_u^*}/0.2)}{\sum_c\exp(\hat q_u^\top\hat C_c/0.2)}.$$

熵项 $\mathcal H=-\operatorname{mean}_u\sum_c\alpha_{uc}\log\alpha_{uc}/\log(\max(2,C))$；总损失正号加入时鼓励低熵分配。

### 顺序记忆分支与逐维融合

`safe_tgn_memory.py` 包装 PyG TGNMemory，使用 IdentityMessage、LastAggregator 和 GRU 的顺序记忆。不同于自定义节点状态，这是一套独立状态，读出记为 $h_u^T$。其写入采用增强真实边特征，时间强转 long；连续小数时间会被截断，故新业务要统一时间粒度。

$$g_u=\sigma(W_g[h_u^D\Vert h_u^T]+b_g),\quad h_u=g_u\odot h_u^T+(1-g_u)\odot h_u^D,$$
$$s(u,r,v,t)=\operatorname{MLP}_{link}([h_u\Vert h_v\Vert\rho_r\Vert\phi(t-last_u)]).$$

概率形式为 $\sigma(s)$，仅用于训练/排序，不等于经过校准的现实发生概率。

### 训练目标与因果提交

每个目标事件抽一个 negative destination $v^-$。有监督集合 $S=\{i:y_i^{sup}>0.5\}$：

$$\mathcal L_{edge}=\frac1{|S|}\sum_{i\in S}w_{r_i}\{\operatorname{BCEWithLogits}(s_i^+,1)+\operatorname{BCEWithLogits}(s_i^-,0)\},$$
$$\mathcal L=\mathcal L_{edge}+\lambda_{CL}\mathcal L_{CL}+\lambda_H\mathcal H.$$

上式概括损失组成；具体两个 BCE 的 reduce/平均尺度见 trainer.py。只有目标关系进入 edge loss，所有真实事件都进入状态提交。训练先 forward、loss/backward/optimizer，再提交 detached cache；因此提交缓存基于该批优化前表示。每轮 reset；验证重放 train，测试重放 train+val；评估中每批先评分后观察真实事件。

### 指标的严格含义

每个真实目标边使用 $K=20$ 个均匀全节点负目的节点：

$$rank_i=1+\sum_{j=1}^{20}\mathbb1[s_{ij}^-\ge s_i^+],\quad MRR=\frac1n\sum_i1/rank_i.$$

同分采用悲观排名。无并列且随机排序的期望 MRR 为 $H_{21}/21$，约 0.1736；全部同分才为 $1/21$。前文“随机同分基线”的说法只适用于全同分情形。AUC 是随机正样本超过随机负样本的概率（并列计半）；AP 为按阈值排序后的 precision 对 recall 增量加权和；accuracy 使用分类阈值。正负比例 1:20 时全负分类器 accuracy 为 20/21≈0.9524，因此 accuracy 不能单独说明效果好。AP 的随机排序基准约为正类占比 1/21（有限样本时有偏差）。本实现不限制评估负样本节点类型，也不排除真实边或原正目的节点。

## 13. 当前项目归档与历史结果索引

项目根目录已迁移为 `iconip-master`。本文执行路径已替换为项目相对路径；进入本项目根目录执行。旧日志与论文内保留历史路径用于溯源。完整可执行命令见根目录 README、REPRODUCE 及 tools/reproduce.py。四数据集讲解不代表历史目录只有四个数据集，归档也包含 CN/IN/Amazon/Github 及全部实验产物。

历史主实验与 best-run catalog 导出的真实指标在 `repro/results.csv`；对应完整参数、best_epoch、选择来源在 `repro/configs/*.json`。后续生成表紧接本节。历史值未经本次完整重训确认。原报告可能引用过期主表或 test-MRR 选优版本，应以具体 run 的 metrics.json 为准。

## 14. 新增审计结论

1. 静态 node_feat 中结构统计使用统一图全流计算，可能含未来 degree、first/last time。即使动态分支 score-before-update，也不能据此声称整条管线完全无未来信息。严格因果复现需改成 train-only/online 统计并重跑所有方法。
2. 原 lightweight checkpoint 排除整个 memory 前缀，会漏掉 frozen node/relation GRU 权重。同时 startswith("node_type") 误删了已训练 node_type_emb.weight，无法按 seed 精确恢复完整模型。新项目修复匹配边界，仅排除动态 buffers，推理须重新训练。
3. 社区 GRU 及自定义节点/关系 updater 的状态写入不追踪梯度，不能把状态更新公式等同于 updater 已被优化。
4. 历史 best catalog 明确依据 test MRR 挑选配置，属于测试集参与选择。归档保留事实，不将其称为验证集独立选优或重新认证的论文结果。

## 15. 本次导出的历史实验结果

以下数值直接来自对应 metrics.json。main 是旧主 campaign；catalog_best_test_mrr 是历史 catalog 的 run（不意味着其 metrics 与旧 catalog 数字完全一致）；targeted_search 为调参记录。AP/AUC/MRR 均为小数。

|Run|来源|Best epoch|MRR|AP|AUC|
|---|---|---:|---:|---:|---:|
|amazon_movies_edge_inductive__seed42|main|7|0.924793|0.794603|0.991095|
|amazon_movies_edge_transductive__seed42|main|8|0.911772|0.813351|0.989520|
|aminer_paper_citation_edge_inductive__seed42|main|2|0.779022|0.511440|0.965797|
|aminer_paper_citation_edge_transductive__seed42|main|13|0.767032|0.514945|0.973249|
|dblp_edge_inductive__seed42|main|4|0.768291|0.472673|0.947293|
|dblp_edge_transductive__seed42|main|2|0.254786|0.136416|0.408493|
|patent_cn_edge_org_codeIII_inductive__seed42|catalog_best_test_mrr|12|0.995645|0.986999|0.999572|
|patent_cn_edge_org_codeIII_transductive__seed42|main|1|0.992949|0.979413|0.999197|
|patent_in_edge_org_codeIII_inductive__seed42|main|11|0.993546|0.980089|0.999202|
|patent_in_edge_org_codeIII_transductive__seed42|main|1|0.995557|0.983860|0.999371|
|patent_us_edge_org_codeIII_inductive__seed42|catalog_best_test_mrr|2|0.990566|0.981553|0.999395|
|patent_us_edge_org_codeIII_transductive__seed42|main|1|0.988519|0.967685|0.998806|
|thgl_software_edge_inductive__seed42|main|1|0.470311|0.164213|0.860341|
|thgl_software_edge_transductive__seed42|catalog_best_test_mrr|1|0.799978|0.569953|0.974024|
|aminer_paper_citation_edge_inductive__rank_norel_lr5e4__s42|targeted_search|2|0.783040|0.554610|0.976167|
|aminer_paper_citation_edge_inductive__rank_norel_topk8__s42|targeted_search|2|0.740911|0.473085|0.954333|
|aminer_paper_citation_edge_transductive__rank_norel_lr5e4__s42|targeted_search|18|0.769302|0.519774|0.973452|
|aminer_paper_citation_edge_transductive__rank_norel_topk8__s42|targeted_search|11|0.775494|0.525463|0.974729|
|dblp_edge_inductive__rank_nocomm_lr5e4__s42|targeted_search|13|0.844784|0.579579|0.970046|
|dblp_edge_inductive__rank_topk4_lr1e3__s42|targeted_search|2|0.738807|0.456792|0.964633|
|dblp_edge_inductive__rank_topk8_lr5e4__s42|targeted_search|20|0.857220|0.644225|0.983733|
|dblp_edge_transductive__rank_nocomm_lr5e4__s42|targeted_search|3|0.507243|0.270257|0.906900|
|dblp_edge_transductive__rank_topk4_lr1e3__s42|targeted_search|7|0.540808|0.284166|0.919448|
|dblp_edge_transductive__rank_topk8_lr5e4__s42|targeted_search|20|0.507501|0.247565|0.916181|
|patent_cn_edge_org_codeIII_transductive__mrr_topk4_lr7e4__s42|catalog_best_test_mrr|10|0.995491|0.986506|0.999547|
|patent_cn_edge_org_codeIII_transductive__mrr_topk8_lr5e4__s42|targeted_search|9|0.994811|0.981191|0.999392|
|patent_in_edge_org_codeIII_inductive__mrr_topk4_lr7e4__s42|catalog_best_test_mrr|17|0.996138|0.989330|0.999613|
|patent_in_edge_org_codeIII_inductive__mrr_topk8_lr5e4__s42|targeted_search|1|0.995732|0.974189|0.999170|
|patent_in_edge_org_codeIII_transductive__mrr_topk4_lr7e4__s42|targeted_search|13|0.995368|0.987802|0.999552|
|patent_in_edge_org_codeIII_transductive__mrr_topk8_lr5e4__s42|catalog_best_test_mrr|1|0.995630|0.987465|0.999529|
|patent_us_edge_org_codeIII_transductive__mrr_topk4_lr7e4__s42|targeted_search|2|0.996256|0.987724|0.999604|
|patent_us_edge_org_codeIII_transductive__mrr_topk8_lr5e4__s42|targeted_search|7|0.996307|0.984886|0.999473|
|thgl_software_edge_inductive__rank_no_edge_type__s42|targeted_search|1|0.817174|0.611132|0.976598|
|thgl_software_edge_inductive__rank_topk8_lr5e4__s42|catalog_best_test_mrr|13|0.811451|0.587178|0.974568|
|thgl_software_edge_transductive__rank_no_edge_type__s42|targeted_search|1|0.846576|0.640618|0.979167|
|thgl_software_edge_transductive__rank_topk8_lr5e4__s42|targeted_search|10|0.800826|0.518379|0.971713|
|patent_in_edge_org_codeIII_transductive__close_topk4_lr7e4_ap__s42|targeted_search|18|0.995450|0.988987|0.999584|
|patent_in_edge_org_codeIII_transductive__close_topk8_lr4e4_ap__s42|targeted_search|4|0.995412|0.985558|0.999500|
|patent_in_edge_org_codeIII_transductive__close_topk8_lr5e4_ap__s42|targeted_search|7|0.995134|0.986318|0.999504|
|patent_us_edge_org_codeIII_transductive__close_topk4_lr7e4_ap__s42|catalog_best_test_mrr|17|0.996349|0.989818|0.999621|
|patent_us_edge_org_codeIII_transductive__close_topk8_lr4e4_ap__s42|targeted_search|14|0.995393|0.986560|0.999518|
|patent_us_edge_org_codeIII_transductive__close_topk8_lr5e4_ap__s42|targeted_search|6|0.996070|0.986607|0.999541|
|thgl_software_edge_inductive__noetype_topk0_lr7e4__s42|targeted_search|1|0.797897|0.582749|0.972754|
|thgl_software_edge_inductive__noetype_topk4_lr5e4__s42|targeted_search|1|0.773989|0.524704|0.969055|
|thgl_software_edge_inductive__noetype_topk8_lr3e4__s42|targeted_search|13|0.799931|0.573593|0.971989|
|thgl_software_edge_transductive__noetype_topk0_lr7e4__s42|targeted_search|1|0.793287|0.575715|0.972863|
|thgl_software_edge_transductive__noetype_topk4_lr5e4__s42|targeted_search|1|0.784812|0.539214|0.971219|
|thgl_software_edge_transductive__noetype_topk8_lr3e4__s42|targeted_search|13|0.797488|0.530325|0.972104|

## 16. 归档数据审计与修复协议

本次实际读取 41 份统一图/任务张量。40 份通过索引互斥与边界检查；`data/processed/tasks/patent_us_edge_org_codeIII_inductive.pt` 中 train 与 test 共享 129 个事件，test 总共只有 131 个事件（53 个监督目标）。详情见 `repro/data_audit.json` 和 `repro/patent_us_split_overlap.json`。这属于事件集合泄漏，不能把对应旧指标作为严格隔离测试结果。

原 builder 依据不同实体组的末事件时间建立窗口，train cutoff 可能晚于 validation cutoff，窗口便可能重叠。新项目在 `build_task_view` 中按 test 优先、validation 其次分配事件，从 train/validation 删除重复索引，保持 held-out 事件，记录 `meta.extra.split_overlap_repair`。本地已有数据未被覆盖；重新运行 preprocess 才生成修正后的任务。该修复保证事件集合互斥，但不自动解决实体留出和全局时间先后冲突；严格因果实体划分仍需重新设计并审计跨窗口时间。

`tools/reproduce.py train` 默认拒绝有重复索引的旧数据。仅为诊断历史行为时可显式添加 `--allow-historical-overlap`；此模式不能用于声称严格测试有效。重新划分和修正 checkpoint 后的新结果应作为新实验报告，不能沿用旧表。

## 17. 原始统计与处理后规模不能混用

本次 US/CN/IN/software 直接扫描原始文件；AMiner 原始统计沿用已有完整扫描 JSON。DBLP/Amazon/Github 的统计代码与原始文件已归档，本次未重新全量扫描其原始记录。原始统计结果的字段、去重口径和边生成计数见 `data/reports/raw_stats`。

|原始数据|记录/事件数|实体口径|说明|
|---|---:|---|---|
|Patent-US|200,192 专利行|520 org、609 CodeIII|2010–2019；处理规则生成的 org–CodeIII 事件 422,825|
|Patent-CN|200,033 专利行|2,160 org、609 CodeIII|2010–2019；org–CodeIII 事件 279,864|
|Patent-IN|160,718 专利行|3,166 org、631 CodeIII|org–CodeIII 事件 486,402|
|THGL-software|1,489,806 原始事件|681,927 端点节点、14 关系|全量 CSV；导入不做 200k 截断|
|AMiner-Paper|2,092,356 已扫描论文记录|详见历史 raw_stats JSON|后续仅保留指定年份和引用额度，统一图规模见第3节|
|DBLP/Amazon/Github|本次未全量重扫|以 raw_statistics.py 输出为准|不把 200k 处理子集冒充原始规模|

专利统计表的 org 为 orgID 数，不是 applicant_name 数；一行专利会展开多种关系，不能把原始行数等同于异构图边数。原始文本特征、全流结构特征、目标关系事件及上下文窗口数量分别具有不同口径。
