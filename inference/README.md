# 将 DHC 用作 Agent 工具

DHC 接收时序异构图节点、关系和时间，给候选目的节点打分。Agent 负责实体解析、候选生成和结果解释，工具负责图关系排序。分数为 logit 与 sigmoid(logit)，后者不是经过校准的现实概率。一个业务图需使用其自身的数据、关系模式和训练权重。

## 加载与调用

```python
from inference import DHCTool

tool = DHCTool(
    "results/reproduced/patent_us_edge_org_codeIII_transductive__seed42",
    device="cpu", history="train",
)
result = tool.rank_links(
    source_id=10, candidate_ids=[20, 21, 22],
    relation_id=2, timestamp=2000000000.0, top_k=3,
)
```

示例 ID 仅演示接口，必须替换为同一任务 `.pt` 内部节点 ID、正确的源/目标类型、关系 ID，以及不早于历史的同单位时间戳。Patent 的关系 2 是 org→CodeIII。查询时间可以直接采用 `tool.watermark` 或其后合法时间；候选须为 CodeIII 节点。节点类型见 `tool.data.node_type`，关系与可能存在的 ID 映射见 `tool.data.meta`。部分原预处理未持久化完整业务 ID 映射，不可将任意业务编号当作模型 ID；生产接入需在数据准备时持久化业务 ID↔内部 ID 映射。

`history="train"` 只重放训练；`train_val` 重放训练+验证，用于测试/上线前状态；`none` 从空状态开始。初始化会加载全图，数十万节点需要 GB 级内存和较长 replay 时间；服务应长期持有一个实例。批次大小保持训练配置；改单事件 replay 会改变批内状态语义。`rank_links` 不提交事件，重复查询不改变记忆。

原始轻量 checkpoint 同时缺少 memory GRU 和已训练的 node-type embedding，因此不能精确恢复，入口会拒绝并列出缺失参数。请先用修正后的保存器重新训练，再加载 `results/reproduced/` 下的 checkpoint。`results/smoke` 是可直接验证接口的合成小图模型，不代表业务模型。

`memory_final_state.pt` 仅含抽样节点状态，不是完整可恢复服务快照；重启后须重新 replay 历史。

## 事件写入

确认发生的真实事件才调用：

```python
tool.observe(source_id=10, destination_id=20, relation_id=2,
             timestamp=2000000001.0, edge_features=[0.1, 0.2, 0.0, 1.0])
```

edge_features 必须复用训练时归一化统计与字段顺序，示例数值不可直接用于业务。模型自动补 relation one-hot。所有上下文关系也应按时间顺序写入。预测候选不能写入历史，否则将形成反馈泄漏。单实例锁保证查询/写入串行；多进程各自有独立状态，应由上游事件流统一广播，外层自行实现鉴权和持久化。

## JSON-lines 进程接口

```bash
python -m inference.cli \
  --run-dir results/reproduced/patent_us_edge_org_codeIII_transductive__seed42 \
  --history train
```

stdin 一行一个 JSON，stdout 一行一个结果；启动诊断在 stderr。

```json
{"action":"rank_links","source_id":10,"candidate_ids":[20,21,22],"relation_id":2,"timestamp":2000000000.0,"top_k":3}
```

响应包含 `ok`、`result.ranked`、历史水位和 logit；非法 ID、倒序时间、空候选等返回 `ok:false`。每次最多 4096 个候选。Agent 可将 JSON schema 暴露为 function tool，只允许 `rank_links`；`observe` 应由可信业务事件入口调用。

可将 `inference/tool_schema.json` 的通用函数 schema 注册到所用 Agent 框架；收到调用后用 `tool.rank_links(**arguments)` 分派。schema 本身不依赖任何特定 LLM SDK。不要把观测写入接口作为可由模型自由调用的工具。
