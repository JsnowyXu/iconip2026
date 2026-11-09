# 边语义与 Relation Memory 的 Case Study 写作材料

本 case study 面向两个问题：第一，不同边表征是否会影响链路预测准确度；第二，relation memory 是否真的学习到了随训练过程变化的语义表征，而不是静态的 relation id embedding。

## 1. 不同边表征对准确度的影响

- `dblp_edge_transductive`：Full 的 test MRR 为 `0.5447`；去掉 `No edge type` 后下降最明显，ΔMRR=`-0.2904`。
- `patent_us_edge_org_codeIII_transductive`：Full 的 test MRR 为 `0.9962`；去掉 `No relation memory` 后下降最明显，ΔMRR=`-0.0003`。
- `thgl_software_edge_inductive`：Full 的 test MRR 为 `0.7971`；去掉 `No community` 后下降最明显，ΔMRR=`-0.0075`。
- `aminer_paper_citation_edge_transductive`：Full 的 test MRR 为 `0.7369`；当前消融没有带来下降，`No relation memory` 反而提升 ΔMRR=`0.0361`，应作为边界条件讨论。
- `thgl_software_edge_transductive`：Full 的 test MRR 为 `0.7941`；当前消融没有带来下降，`No edge type` 反而提升 ΔMRR=`0.0297`，应作为边界条件讨论。

最强证据来自 DBLP transductive 任务：完整模型 test MRR 为 `0.5447`，去掉 edge type 后降至 `0.2543`，下降约 `0.2904`；去掉 relation memory 后降至 `0.4736`，下降约 `0.0711`。这说明在 paper-citation 预测中，模型需要区分 `author_write_paper` 上下文边和 `paper_cite_paper` 目标边；边类型/关系记忆不是装饰项，而会实质影响预测准确度。

Patent-US 则是一个边界案例：`org -> codeIII` 任务接近饱和，Full MRR 已接近 1.0，因此去掉 relation memory 只带来很小下降。这不否定 relation memory 的作用，而说明该任务本身可分性较强，accuracy 指标难以继续放大模块差异。

AMiner 和 THGL 中出现部分简化模型超过 Full 的情况，建议在论文中表述为：relation memory 的动态表征是存在的，但其带来的准确度收益依赖数据集的监督关系、上下文关系噪声和任务难度。

对应图：`edge_representation_delta_mrr.png` 展示各消融相对 Full 的 MRR 变化；`case_task_variant_test_mrr.png` 展示 Full 与各边表征变体的 test MRR。

## 2. Relation Memory 是否体现语义演变

- `thgl_software_edge_inductive` 的目标关系 `rel_6` 在二维 relation-memory 空间中的路径长度为 `9.109`，首尾直线漂移为 `3.044`，范数变化为 `1.094`。
- `thgl_software_edge_transductive` 的目标关系 `rel_6` 在二维 relation-memory 空间中的路径长度为 `2.212`，首尾直线漂移为 `0.819`，范数变化为 `-0.627`。
- `aminer_paper_citation_edge_transductive` 的目标关系 `paper_cite_paper` 在二维 relation-memory 空间中的路径长度为 `1.806`，首尾直线漂移为 `0.149`，范数变化为 `-0.482`。
- `dblp_edge_inductive` 的目标关系 `paper_cite_paper` 在二维 relation-memory 空间中的路径长度为 `1.796`，首尾直线漂移为 `0.791`，范数变化为 `-0.679`。
- `aminer_paper_citation_edge_inductive` 的目标关系 `paper_cite_paper` 在二维 relation-memory 空间中的路径长度为 `1.078`，首尾直线漂移为 `0.319`，范数变化为 `-0.532`。
- `patent_us_edge_org_codeIII_transductive` 的目标关系 `org_codeIII` 在二维 relation-memory 空间中的路径长度为 `1.007`，首尾直线漂移为 `0.611`，范数变化为 `-1.433`。
- `dblp_edge_transductive` 的目标关系 `paper_cite_paper` 在二维 relation-memory 空间中的路径长度为 `0.979`，首尾直线漂移为 `0.425`，范数变化为 `-0.475`。
- `patent_us_edge_org_codeIII_inductive` 的目标关系 `org_codeIII` 在二维 relation-memory 空间中的路径长度为 `0.604`，首尾直线漂移为 `0.498`，范数变化为 `-1.076`。

这些轨迹说明 relation memory 并非固定查表向量。以 DBLP 为例，`paper_cite_paper` 和 `author_write_paper` 在训练过程中沿不同方向移动，目标 citation 关系的二维位置、路径长度和范数均发生变化；同时 DBLP 的消融结果显示去掉 edge type 或 relation memory 会显著降低 MRR。因此可以把 DBLP 写成“语义演化 + 准确度因果证据”最强的 case。

THGL-software 的 `rel_6` 目标关系路径长度较大，适合作为多关系场景下 relation memory 发生显著动态漂移的可视化案例；Patent-US 虽然准确率接近饱和，但 `org_codeIII` 的 relation memory 仍有可观轨迹变化，适合说明模型内部确实在更新目标关系语义。

对应图：每个任务都有 `*_relation_type_trajectory.png` 展示 relation type 的二维轨迹，`*_target_relation_drift_vs_val_mrr.png` 展示目标 relation 的逐 epoch 漂移与验证 MRR 的对应关系，`target_relation_memory_path_length.png` 汇总目标关系的语义路径长度。

## 3. 可直接写入论文的段落草稿

为了验证边语义记忆是否真正影响预测，我们进一步比较了完整模型与去除 edge type、去除 relation memory 等变体。在 DBLP citation 任务中，完整模型达到 0.5447 test MRR；去除 edge type 后下降到 0.2543，去除 relation memory 后下降到 0.4736，分别带来 0.2904 和 0.0711 的 MRR 损失。这表明模型需要显式区分 `author_write_paper` 上下文边和 `paper_cite_paper` 目标边，边语义表征对 paper citation prediction 具有实质贡献。

进一步地，我们将每个 epoch 保存的 relation memory 投影到二维空间。结果显示目标关系并非停留在固定位置，而是在训练过程中形成连续轨迹。例如 DBLP 的 `paper_cite_paper` 关系在二维空间中产生明显漂移，同时其 relation-memory 范数也发生变化；THGL-software 中目标关系 `rel_6` 的路径长度更大，说明多关系软件交互场景下 relation memory 具有更强的动态演化。上述现象表明 relation memory 学到的是随时间更新的边语义状态，而不是静态 relation id 编码。
