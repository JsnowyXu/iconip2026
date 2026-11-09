# DHC 实验设计与模型整体说明

本文基于 `dyn_relation_memory/dhc` 目录下当前实现整理，目标是从任务视图、输入结构、模型模块、状态更新、loss 设计、评估指标到输出产物，对 DHC 做一次自上而下的整体总结。这里的 DHC 不是单一模块，而是一个把异构关系记忆、近期邻居上下文、社区记忆和 TGN 风格时序记忆组合起来的动态图学习框架。

## 1. DHC 解决什么问题

DHC 面向统一异构时序图上的三类主要实验：

- `edge_transductive`：目标关系上的时序链路预测，训练/验证/测试中节点集合可重合，但监督边按时间切分。
- `edge_inductive`：目标关系上的归纳式链路预测，测试时显式保留未在训练上下文中出现的目标侧或源侧节点，考察泛化。
- `node classification`：基于完整时序边流先构建节点动态状态，再做节点多分类或多标签预测。

当前任务构建脚本主要产出 `node_multilabel` 视图，但 DHC 训练和评估代码同时支持：

- `node_multiclass`
- `node_multilabel`

另外代码里还支持一个专利场景的 `edge_patent_ipc`，本质上仍按边任务协议执行。

## 2. 统一输入数据结构

统一数据由 `UnifiedGraphData` 封装，核心字段包括：

- 图结构与静态属性：
  - `num_nodes`
  - `num_rels`
  - `num_node_types`
  - `node_feat`
  - `node_type`
- 时序边流：
  - `src`
  - `dst`
  - `ts`
  - `rel`
  - `edge_feat`
  - `edge_label`
- 时序划分：
  - `train_idx`
  - `val_idx`
  - `test_idx`
- 节点监督：
  - `node_multiclass = {"node_ids", "labels"}`
  - `node_multilabel = {"node_ids", "labels"}`
- 元信息：
  - `meta`

其中最关键的是两点：

1. `train_idx/val_idx/test_idx` 指向的是时序边流中的边位置，而不是节点划分。
2. `edge_label` 用于区分“这条边是否参与监督”。对于边任务，很多非目标关系边会保留在图里作为上下文，但不参与链路损失。

## 3. 两类任务视图如何构造

### 3.1 Edge Transductive

边转导任务不是只保留目标关系，而是采用“完整异构上下文 + 目标边监督”的协议：

- 先在原图中选定目标关系边。
- 仅对目标关系边做时间排序，再按时间切成 train/val/test。
- 真正导出的任务视图中，保留所有落在对应时间窗内的上下文边。
- 只有目标边对应 `edge_label = 1`，其余上下文边 `edge_label = 0`。

这意味着模型训练时看到的是完整异构时序流，但 link loss 只落在目标边上。

### 3.2 Edge Inductive

边归纳任务在 transductive 的基础上，再增加“未见节点”约束。

论文类任务视图里，默认流程是：

- 从目标关系边的源节点中随机抽取 10% 作为 `unseen`。
- 所有关联这些 `unseen` 节点的目标边进入测试集合。
- 训练/验证只在其余目标边上按时间切分。
- 同时通过 `context_keep_mask` 把所有触及 `unseen` 节点的上下文边从训练上下文中移除。

因此 inductive 不是简单的时间外推，而是同时控制了目标监督边和上下文可见性，保证测试节点在训练期不被上下文泄漏。

专利任务里如果元数据提供了预定义 `task_splits` 和 `unseen_assignee_node_ids`，则优先沿用；否则也会退化成类似的时间划分协议。

## 4. 模型总览

DHCModel 的输入和结构可以概括成下面五层：

1. 静态节点输入层
2. 多粒度动态状态层
3. 近期局部结构编码层
4. 社区级中观记忆层
5. 任务预测头与状态更新层

它的核心不是一条编码路径，而是两条动态表示分支再融合：

- DHC 本地异构上下文分支
- TGN memory 分支

最终通过一个门控模块做查询级融合。

## 5. 输入表示与基础编码

### 5.1 节点静态输入

每个节点的基础输入由三部分拼接而成：

- `node_feat`
- `node_type` 的 embedding
- 当前节点记忆 `node_memory`

然后送入 `node_encoder`：

- `Linear(node_feat_dim + 16 + mem_dim -> mem_dim)`
- `ReLU`
- `Linear(mem_dim -> mem_dim)`

因此基础节点表示已经同时包含：

- 静态特征
- 异构类型信息
- 动态记忆状态

### 5.2 关系表示

每条关系的表示由两部分组成：

- 关系嵌入 `rel_emb`
- 可选的关系记忆 `rel_memory` 经过 `rel_mem_to_vec` 后的投影

若开启 `use_relation_memory`，关系向量为：

`rel_vec = rel_emb(rel) + rel_mem_to_vec(rel_memory(rel))`

这样关系不再只是离散类型，而是能随历史交互动态变化。

### 5.3 时间编码

时间差由 `TimeEncoding` 编码，使用一组指数频率上的 `sin/cos` 位置样式编码。编码输入主要是：

- 节点上次更新时间与当前事件时间的差值
- 近期邻居事件和当前查询时刻之间的差值
- 局部子图中边时间与节点最近更新时间之间的差值

时间信号在 DHC 中不是附属特征，而是消息、局部结构编码和预测头的共同输入。

## 6. 动态状态的四个核心记忆模块

### 6.1 节点-关系双记忆 `NodeRelationMemory`

这是 DHC 的第一层核心记忆，维护：

- `node_memory[num_nodes, mem_dim]`
- `rel_memory[num_rels, mem_dim]`
- 以及各自的 `last_update`

更新方式：

- 节点记忆使用 `GRUCell(mem_dim, mem_dim)`
- 关系记忆也使用 `GRUCell(mem_dim, mem_dim)`

一个很关键的实现细节是：

- 同一个 batch 内，如果某个节点或关系收到多条消息，只保留时间戳最后的一条消息用于更新。

这保证了状态更新和时序事件流的一致性，避免同 batch 内重复覆盖造成的混乱。

### 6.2 最近邻居存储 `RecentNeighborStore`

这是第二层动态结构缓存。它为每个节点维护固定长度的最近交互窗口，记录：

- 最近邻居 id
- 对应关系 id
- 时间戳
- 事件 id
- 边特征

插入时：

- 把每条边当成无向交互写入 `src` 和 `dst` 两侧。
- 只保留最近的 `size` 条事件。

它承担两个作用：

1. 为 `_recent_neighbor_context` 提供轻量级时序邻域注意力。
2. 为 `_encode_link_query_nodes` 构造近期局部子图，交给 `TransformerConv` 编码。

### 6.3 TGN 记忆 `SafeTGNMemory`

DHC 不是完全替代 TGN，而是额外保留一条 TGN 风格记忆分支：

- `raw_msg_dim = edge_feat_dim`
- `message_module = IdentityMessage`
- `aggregator_module = LastAggregator`

这个分支只直接消费原始边特征，不显式消费 DHC 里构造的关系感知消息。

`SafeTGNMemory` 对原始 PyG `TGNMemory` 做了一个安全包装：

- 在从训练切到评估时，先把 pending messages flush 到 memory。
- 再重置 message store，避免长时序流上积压消息导致一次性内存分配过大。

### 6.4 社区记忆 `CommunityMemory`

这是 DHC 的中观层设计，用于建模跨节点、跨关系共享的社区语义。

它包含：

- 可学习社区原型 `prototypes[num_communities, node_dim]`
- 社区动态记忆 `community_memory[num_communities, community_dim]`
- 节点到社区的软分配函数
- 社区更新 GRU

软分配流程：

1. 用 `node_to_comm` 把节点表示映射到社区查询空间。
2. 用 `proto_to_comm` 把社区原型映射到同一空间。
3. 做缩放点积得到 assignment logits。
4. 可选地只保留 top-k 社区，再 softmax。

得到的 `alpha` 是节点对各社区的软归属权重。

社区上下文读取：

- `ctx = alpha @ community_memory`
- 再通过 `out_proj([node_emb, ctx])` 融合成新的节点表示

社区更新：

- 先对节点表示做 `in_proj`
- 按 `alpha` 加权聚合到社区
- 用节点对社区的平均贡献作为社区输入消息
- 对旧社区记忆施加时间衰减 `exp(-dt/1000)`
- 再用 `GRUCell` 更新社区记忆

因此社区模块既能提供跨节点共享上下文，也能随时间漂移。

## 7. 局部结构编码与查询表示构造

DHC 实际上有两种“近期邻域建模”方式。

### 7.1 节点表示路径：轻量级近期邻居注意力

`_node_repr` 的逻辑是：

1. 先计算基础节点表示 `base_h`
2. 若开启最近邻居模块，则调用 `_recent_neighbor_context`
3. 把 `base_h` 与邻域上下文拼接后，经 `node_fusion` 融合
4. 若开启社区模块，再读出社区上下文

其中 `_recent_neighbor_context` 会对每个节点的最近邻居做一次时序注意力聚合：

- Value 输入：`[neighbor_memory, relation_vector, delta_t_encoding]`
- Attention 输入：`[query_h, value, relation_vector, delta_t_encoding]`

所以这个分支更像“事件级邻域摘要器”，主要服务于：

- 节点任务预测头
- 社区对活跃节点的读取和更新

### 7.2 边查询路径：近期局部子图 + TransformerConv

链路预测不会直接复用上面的轻量邻域摘要，而是单独构造局部子图：

1. 对当前待预测的 `query_nodes = unique(src, dst)`，从 `RecentNeighborStore` 抽取近期交互子图。
2. 对子图节点先计算基础表示 `base`.
3. 构造边属性：
   - 关系向量 `rel_vec`
   - 原始边特征 `edge_feat`
   - 时间编码 `time_enc(rel_dt)`
4. 用 `TransformerConv` 在局部子图上传播。
5. 再通过 `local_graph_fusion([base, msg_h])` 融合。
6. 若开启社区模块，再加社区上下文。

这条路径比 `_recent_neighbor_context` 更“图结构化”，专门面向链路查询。

## 8. DHC 与 TGN 双分支融合

在真正的 `predict_links` 中，DHC 不直接用单一路径做预测，而是融合：

- `dhc_query_h = _encode_link_query_nodes(query_nodes)`
- `tgn_query_h = tgn_memory(query_nodes)`

然后通过一个门控网络：

- `gate = sigmoid(Linear([dhc_query_h, tgn_query_h]))`
- `query_h = gate * tgn_query_h + (1 - gate) * dhc_query_h`

这一步很重要，它说明代码里的 DHC 是“增强型 DHC/TGN 混合体”：

- DHC 分支擅长关系、社区和近期结构建模
- TGN 分支擅长经典时序记忆汇总
- 门控决定当前查询更依赖哪一种状态

## 9. 消息生成与状态更新设计

### 9.1 正负边预测

训练 batch 输入为：

- 正边：`(src, dst, rel, ts, edge_feat)`
- 负边：把 `dst` 替换成负采样得到的 `neg_dst`

模型分别计算：

- `pos_logit = predict_links(src, dst, rel, ts)`
- `neg_logit = predict_links(src, neg_dst, rel, ts)`

### 9.2 关系-时间双感知消息

在预测之后，DHC 会为状态更新构造双向消息。

对源点接收的消息：

- 原始拼接为 `[dst_mem, src_mem, edge_feat, dt_src]`

对终点接收的消息：

- 原始拼接为 `[src_mem, dst_mem, edge_feat, dt_dst]`

然后都经过 `RelationTimeMessage`：

- `base = Linear(x)`
- `gate = sigmoid(Linear(rel_vec))`
- `bias = Linear(rel_vec)`
- 输出 `base * gate + bias`

因此消息不是简单拼接后线性投影，而是显式受关系向量调制。

### 9.3 消息如何写入不同记忆

得到关系感知消息后：

- `msg_to_mem` 投影到节点记忆更新空间
- `msg_to_rel` 投影到关系记忆更新空间

随后在 `update_state` 中依次更新：

1. 节点记忆
2. 最近邻居缓存
3. TGN 记忆
4. 关系记忆
5. 社区记忆

这是一个从微观事件到中观社区的层级更新顺序。

## 10. 负采样设计

训练时每条正边只采样一个负目标点，负采样混合三种来源：

- typed negative：与正样本 `dst` 节点类型相同
- popular negative：从训练集中高频目标点池按频次采样
- uniform negative：全局均匀随机

默认比例：

- `typed_negative_ratio = 0.6`
- `popular_negative_ratio = 0.2`
- 其余 0.2 为均匀采样

并且代码会显式避免 `neg_dst == dst`。

这套设计希望让训练负样本既保持语义难度，也覆盖真实分布中的热门节点。

## 11. 节点任务的监督方式

节点任务训练时，仍然按 `train_idx` 的时序边流逐 batch 推进，这意味着：

- 模型会先通过边流更新节点/关系/社区状态
- 同时在当前 batch 涉及到的活跃节点上计算节点头损失

这里有两个关键点：

1. 节点任务下，link prediction loss 被关掉。
2. 但 `forward_batch` 仍会生成消息、更新缓存，并可继续使用社区对比损失。

所以节点任务的本质是：

- 边流负责“动态表示生成”
- 节点头负责“监督信号注入”

### 11.1 Multi-class 头

若存在 `node_multiclass`，模型增加：

- `multiclass_head: Linear(mem_dim -> num_multiclass)`

训练损失使用：

- `CrossEntropyLoss`

### 11.2 Multi-label 头

若存在 `node_multilabel`，模型增加：

- `multilabel_head: Linear(mem_dim -> num_multilabel)`

训练损失使用：

- `BCEWithLogitsLoss`

可选增强：

- `pos_weight` 按标签正负样本比构造
- 再用 `ml_pos_weight_power` 做幂次压缩
- 并截断到 `ml_pos_weight_max`

这对长尾多标签场景比较关键。

## 12. 总体 loss 设计

### 12.1 边任务

边任务的主损失为监督边上的二元交叉熵：

- 正边对 `1`
- 负边对 `0`

并支持 relation-specific loss weight：

- 对每条监督边，根据其 `rel` 乘以关系权重

完整形式可以理解为：

`L = L_link + lambda_cl * L_contrastive + lambda_entropy * L_comm_entropy`

其中：

- `L_link`：监督边上的正负二分类损失
- `L_contrastive`：社区对比损失
- `L_comm_entropy`：社区分配熵正则


### 12.3 社区相关附加损失

#### 社区对比损失

对活跃节点表示：

- 先计算软分配 `alpha`
- 取 `argmax(alpha)` 作为正社区
- 用节点投影和社区记忆做 InfoNCE 风格分类

这推动节点更贴近其主社区。

#### 社区分配熵

计算每个节点社区分配的归一化熵并取均值。

由于训练里是把该项直接加到总损失上，所以：

- 当 `lambda_community_entropy > 0` 时，会倾向于更高熵、更平滑的社区分配
- 若希望鼓励更尖锐分配，理论上应给负权重，但当前 CLI 中该权重被限制为非负

这一点是当前实现的明确行为。

## 13. 训练协议

训练入口会先根据数据 `meta` 自动推断任务类型，优先读取：

- `meta.extra.task_view_name`
- `meta.dataset_name`

然后执行如下流程：

1. 加载统一任务视图 `.pt`
2. 根据标签表自动确定是否启用多分类/多标签头
3. 初始化模型
4. 逐 epoch 重置所有动态状态
5. 按时间顺序遍历 `train_idx`，不打乱边流
6. 每个 batch 先前向、再反向、再 `update_state`
7. 每轮结束后重新回放训练历史并做验证
8. 按监控指标保存最佳 checkpoint
9. 最后用 `train + val` 作为历史回放，做最终测试

两个重要实现点：

- 每个 epoch 开始都会 `reset_state()`，保证训练是严格从空状态顺序重放。
- 验证和测试不会沿用训练末尾状态，而是重新 reset 后 replay history，保证评估是因果一致的时序协议。

## 14. 评估协议

### 14.1 历史回放 `replay_history`

无论边任务还是节点任务，评估前都会先：

- `model.reset_state()`
- 按时间顺序回放历史边流
- 每个 batch 调 `forward_batch`
- 只取 cache 做 `update_state`

这里的随机负样本只是为了复用前向接口，真正目的是推进状态，不参与评估指标。

### 14.2 边任务评估

评估时对每条监督边：

- 计算正边分数
- 采样 `num_neg = 20` 个负目标点
- 得到负边分数

再汇总指标：

- `AUC`
- `AP`
- `ACC`
- `MRR`

其中：

- `ACC` 用 0.5 阈值
- `MRR` 用正样本分数相对 20 个负样本的排名计算

评估时每个 batch 打分后，还会用该 batch 的真实边继续更新状态，所以测试是在线时序评估，不是静态一次性打分。

### 14.3 节点任务评估

节点任务先回放历史边流，再对目标节点批量调用 `predict_node_heads`。

多分类指标：

- `acc`
- `macro_f1`

多标签指标：

- `auc`
- `micro_f1`
- `macro_f1`
- `map`
- `precision_at_1/3/5`
- `recall_at_1/3/5`
- `ndcg_at_1/3/5`

对于多标签任务，还支持：

- 在校准节点上搜索最优阈值
- 候选阈值默认在 `[0.01, 0.99]` 上均匀取样
- 以 `micro_f1` 最优为准

## 15. 监控指标与早停

`monitor_metric=auto` 时：

- 边任务默认监控 `val_ap`
- 多标签节点任务默认监控 `val_node_ml_micro_f1`
- 多分类节点任务默认监控 `val_node_mc_acc`

训练支持：

- `early_stop_patience`
- `early_stop_min_delta`

最佳模型保存到 `best_model.pt`。默认保存的是 lightweight checkpoint，会去掉：

- 静态 node feature/type buffer
- 各类运行期 memory buffer
- TGN 与社区的运行态缓存

因此 checkpoint 更像“参数快照”，不是训练当下状态的完整镜像。

## 16. 训练输出与实验产物

一次标准训练后，主要输出包括：

- `best_model.pt`
- `metrics.json`
- `train_history.csv`

若开启 `--save-memory-artifacts`，还会额外生成：

- `memory_trace.pt`
- `memory_final_state.pt`

`metrics.json` 中会记录：

- `best_epoch`
- `best_val`
- `test`
- `test_relation_breakdown`
- `test_task_splits`
- `node_eval`
- `multilabel_threshold_calibration`
- `history`
- `epoch_time_sec`

训练入口还会把：

- 模型参数量
- 运行设备
- CLI 参数

补写到 `metrics.json` 和 `model_stats.json`。

## 17. 默认超参与当前实现上的几个注意点

`dhc/configs/model_config.json` 给出的建议维度是：

- `memory_dim = 128`
- `relation_dim = 64`
- `time_dim = 32`
- `message_dim = 128`
- `community_dim = 128`
- `num_communities = 64`
- `node_type_emb_dim = 16`
- `edge_feat_dim = 2`

CLI 默认训练参数大体一致，但有两个值得注意的点：

1. `run_train.py` 当前默认 `epochs = 5`，而配置文件写的是 `epochs = 10`。
2. 训练循环当前每条正边只采样 1 个负样本；评估时默认使用 20 个负样本。配置文件中的 `negative_samples_per_edge = 20` 并没有直接接入训练主循环。

因此如果后续要写论文或实验说明，建议以 `run_train.py` 和 `trainer.py` 的实际行为为准，而不是只看 JSON 配置。

## 18. 一句话总结 DHC 的实现特征

当前实现里的 DHC 可以概括为：

- 以节点记忆和关系记忆为底座
- 用最近邻交互缓存提取局部时序结构
- 用社区记忆建模中观共享语义
- 再与一条 TGN 记忆分支进行门控融合
- 最终统一服务于边预测和节点分类两类任务

如果从实验设计角度看，它最核心的特点不是“只做链路预测”，而是：

- 在统一异构时序流上，区分“上下文边”和“监督边”
- 让同一套动态状态机同时支持 transductive、inductive两种协议

这也是这套实现最值得在论文或方法章节里强调的地方。
