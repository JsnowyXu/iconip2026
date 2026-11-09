SUBMISSION: 644
TITLE: DHC: Dynamic Heterogeneous Context Learning with Relation Memory for Temporal Link Prediction

## REVIEW 1
----------- Originality and Contribution -----------
SCORE: 1 (Weak Accept)
----------- Overall evaluation -----------
SCORE: 1 (Accept)
----- TEXT:
This paper proposes DHC, a dynamic heterogeneous context-learning framework for temporal link prediction on dynamic heterogeneous graphs. Unlike many temporal graph models that treat relation types as static edge features, DHC maintains dynamic node and relation memories with GRU cells, generates relation-conditioned temporal messages, encodes recent-neighbor temporal subgraphs, and fuses a relation-aware context branch with a sequential event-memory branch through a query-level gate. The method is evaluated on four real-world scholarly, patent, and software datasets under both transductive and inductive settings.

The paper addresses an important problem in temporal heterogeneous graph learning, and the motivation is reasonable. Modeling relation semantics as evolving states rather than fixed edge labels via an explicit relation memory is a good choice, and the relation-conditioned message gate provides a way to incorporate dynamic relation information into temporal message passing. The problem formulation is clear, with helpful definitions, notation, and an overall training algorithm. The experimental evaluation is reasonably comprehensive, covering both transductive and inductive settings, multiple competitive baselines, ablation studies, sensitivity analysis, and case studies. The discussion cases on where DHC does not outperform the strongest baseline increase confidence in the reported results.
### Question 1
My primary concern is that one of the two core components of the model is essentially undefined. The sequential event-memory branch is introduced only in Eq. 15, without describing its architecture or its relationship to the TGN-style memory. Since the final prediction depends on the gated fusion of the context branch and this event-memory branch, omitting the implementation details of the latter substantially limits readers' understanding of how the model operates and makes reproduction difficult. 
A closely related concern is the lack of a clear conceptual differentiation from STHN, which is both the most relevant prior work and the strongest competing baseline in the experiments. The related-work discussion only briefly mentions STHN without explaining what DHC fundamentally adds beyond it. Otherwise, it is difficult to determine whether the reported improvements arise primarily from the proposed relation memory or from additional architectural complexity. 

**Response EN.** Thank you; we agree that the submitted Eq. 15 was not reproducible. We now define the sequential event-memory branch explicitly through directional event messages, latest-message aggregation, a 128-dimensional GRU state update, and the relation-one-hot-augmented edge feature. For each chronological batch, the current state is read before scoring; after scoring and optimization, observed events update it, while sampled negatives never update memory. This state is separate from DHC's node and relation memories. The context branch combines fixed node/type attributes, DHC node states, recurrent relation states, and a bounded recent-neighbor subgraph; the event branch stores compact endpoint-specific chronological history. A vector gate is their only fusion point.

We also added a direct comparison with STHN and TAMI. STHN centers on compact sampling/encoding of recent heterogeneous interactions. DHC instead adds a recurrent state per relation type and uses it to gate temporal messages, construct neighbor-edge attributes, and score queries, together with a separate sequential event-memory branch. TAMI targets interaction heterogeneity through target-aware matching but does not instantiate this recurrent per-relation state. We have moderated the claim: the novelty is dynamic relation semantics throughout the pipeline, not complexity by itself. The existing relation-memory ablation is the available component-level isolation; we do not claim it alone explains every gain.

**中文回复。** 感谢指出。我们同意投稿版本的式(15)不足以复现。修订稿已通过双向事件消息、最新消息聚合、128 维 GRU 状态更新及拼接关系 one-hot 的边特征，完整定义顺序事件记忆。每个时间批次先读取状态并打分，预测和优化后才由观测事件更新；负采样边不更新记忆。该状态与 DHC 的节点记忆、关系记忆独立。上下文分支融合固定节点/类型属性、DHC 节点状态、逐关系递归状态与有界近期邻居子图；事件分支保存端点级紧凑时间历史；两者只在向量门控处融合。

修订稿也补充了与 STHN/TAMI 的正面对比。STHN 的重点是近期异质交互的紧凑采样与编码；DHC 增加逐关系递归状态，并让它参与消息门控、邻居边属性构造和查询打分，同时保留独立 TGN 记忆分支。TAMI 通过 target-aware matching 处理交互异质性，但不具有这一逐关系递归状态。我们已收敛表述：贡献在于动态关系语义贯穿完整流程，而非单纯增加复杂度。现有关系记忆消融是可用的组件级隔离证据，但我们不声称全部提升仅由它造成。

### Question 2
The empirical evaluation is generally thorough, but several aspects could be strengthened. Evaluation is performed using only 20 sampled negatives per positive edge. Given that the paper motivates its evaluation with TGB, TGB 2.0, and TGB-Seq, it would be helpful to additionally report results under more standardized negative-sampling methods with larger datasets. The ablation study is conducted on only two of the four datasets, while the sensitivity analysis is limited to a single dataset. Extending these analyses or providing justification for the current choices would strengthen the empirical evidence.

**Response EN.** We agree. Twenty negatives made it possible to rerun every model with one identical, tractable candidate protocol over all four processed views; it is not the official dataset-specific TGB/TGB-2.0 sampler. We now explicitly scope the reported MRR to controlled sampled-ranking comparisons and do not claim a TGB leaderboard/SOTA result. No completed comparable 100-negative or all-candidate artifacts exist, so we have not invented those values; they remain required follow-up experiments.

Patent-US and THGL-Software were selected for ablation because they contrast a dense, repetitive patent stream with a sparse 14-relation software stream. Patent-US was used for one-factor sensitivity because its repeated organization--technology target gives a stable sweep. These pragmatic choices do not imply that the other datasets are insensitive. We now acknowledge the coverage limitation and identify four-dataset, multi-seed extensions as necessary.

**中文回复。** 我们同意。采用 20 个负例，是为了在四个处理后任务视图上以同一且可承受的候选协议重跑所有模型；它不是 TGB/TGB-2.0 的官方数据集专用采样器。修订稿已把 MRR 的结论限定为受控采样排序比较，不再据此宣称 TGB 榜单级 SOTA。现有产物中没有完整可比的 100 负例或全候选结果，因此我们没有虚构数值；这些仍是必须补充的后续实验。

Patent-US 与 THGL-Software 分别代表稠密、高重复的专利流和稀疏、含 14 种关系的软件流，因而被用于消融对照；Patent-US 的重复组织—技术目标关系适合稳定的单因素敏感性扫描。这些是务实选择，并不表示其他数据集不敏感。我们已明确覆盖限制，并把四数据集、多随机种子扩展列为必要工作。

### Question 3
Another concern relates to efficiency. The proposed framework introduces several additional components, including relation memory, recent-neighbor storage, and a dual-branch architecture, yet it provides no analysis of computational complexity, runtime, parameter count, or memory overhead. Since temporal graph learning is often applied to large-scale dynamic graphs, reporting efficiency alongside predictive performance would improve the practical value of the work.

**Response EN.** We added an implementation-grounded analysis. For batch size (b), memory width (d_m), relation width (d_r), and at most (K_N) recent neighbors, recurrent updates cost (O(bd_m^2)), relation-conditioned projections cost (O(bd_md_r)), and local encoding costs (O(bK_Nd_m^2)) up to attention constants. Persistent state is (O(|V|d_m+|R|d_m+|V|K_N)). Per-relation memory is only (O(|R|d_m)); two node-memory buffers dominate. Saved run metadata records about 1.0--1.2 million trainable parameters depending on dataset dimensions. Existing wall-clock logs are from heterogeneous runs and are not a controlled comparison, so we do not present them as an efficiency table; controlled runtime and peak-memory profiling remain a limitation.

**中文回复。** 修订稿新增了基于实现的复杂度分析。批大小为 (b)、记忆宽度为 (d_m)、关系宽度为 (d_r)、每节点最多保存 (K_N) 个近期邻居时，递归更新、关系条件投影与局部编码的复杂度分别为 (O(bd_m^2))、(O(bd_md_r)) 和 (O(bK_Nd_m^2))（忽略注意力常数）。持久状态为 (O(|V|d_m+|R|d_m+|V|K_N))。逐关系记忆仅占 (O(|R|d_m))，主要开销来自两套节点记忆。运行元数据显示不同数据集配置约含 100–120 万可训练参数。现有墙钟日志来自不同运行条件，不能作为公平效率表，因此我们不报告不可比数值，并明确统一条件下的耗时和峰值显存仍是限制。

### Question 4
There are some grammatical issues throughout the paper, as well as minor inconsistencies in dataset naming. In addition, the provenance of the Patent-US dataset is not clearly described. The cited reference appears to describe patent statistics rather than the graph dataset itself, making it difficult to identify the original data source and reproduce the experiments. The authors should clearly document the dataset origin and preprocessing pipeline.

**Response EN.** We removed the misleading patent-statistics citation because it was not the graph source. The manuscript now records exactly what is locally verifiable. Patent-US is generated from archived `all-us.csv`, `IPCSubClass_US.csv`, and `clsIII-reInd.csv` files containing identifiers, dates, assignee names, IPC codes, and abstracts. We normalize assignee names, map IPC to level-III classes, timestamp by publication date, construct patent--organization, patent--class, organization--class, and organization--organization relations, and predict organization--class while treating the others as context. The project archive does not retain a verifiable upstream URL/version. We disclose this rather than inventing attribution; a release must add the archive checksum and redistribution URL/access instructions. Dataset names and the revised prose were also standardized.

**中文回复。** 我们已删除误导性的专利统计网页引用，因为它不是图数据来源。论文现在只记录能够本地核验的事实：Patent-US 来自归档的 `all-us.csv`、`IPCSubClass_US.csv` 和 `clsIII-reInd.csv`，字段包括标识符、日期、受让组织名、IPC 编码和摘要。预处理包括组织名归一化、IPC 映射到三级类别、按公开日期赋时间戳，构造专利—组织、专利—类别、组织—类别、组织—组织关系，并以组织—类别为预测目标，其余关系作为上下文。项目归档没有保留可验证的上游 URL/版本，因此我们如实披露，而不编造归因；公开包还必须补充归档校验和以及再分发链接或访问说明。数据集命名和修订文字也已统一。



Overall, this is a technically sound paper with a reasonable motivation and a comprehensive experimental evaluation. Although the core idea appears promising, the current manuscript would benefit from a clearer specification of the event-memory branch, a more explicit comparison with STHN, stronger evaluation protocols, and improved experimental reproducibility. Addressing these issues would strengthen the paper.

## REVIEW 2
----------------------- REVIEW 2 ---------------------

----------- Originality and Contribution -----------
SCORE: 1 (Weak Accept)
----------- Overall evaluation -----------
SCORE: 0 (Borderline)
----- TEXT:
Summary.
The paper proposes DHC, a dynamic heterogeneous context-learning framework for temporal link prediction. It combines (i) a relation-aware contextual branch with text-aware node initialization, GRU-based dynamic node and relation memory, relation-conditioned messages, and recent-neighbor temporal subgraphs, and (ii) a sequential event-memory branch, fused by a query-level gate. Experiments on four real-world datasets (AMiner-paper, DBLP, Patent-US, THGL-Software) under both transductive and inductive settings show DHC outperforms competitive baselines (TGN, STHN, TGN-edge, RepeatMixer, TAMI) on most metrics, supported by ablation, sensitivity, and case studies.

Strengths.

Well-motivated problem framing: modeling relation types as evolving states rather than static edge labels is a reasonable and useful angle for dynamic heterogeneous graphs.
Comprehensive empirical setup: four datasets across scholarly/patent/software domains, both transductive and inductive views, MRR and AUC, plus ablation, sensitivity, and case studies.
Consistent gains, especially in the harder inductive setting, where relation memory and recent-neighbor context plausibly compensate for sparse node histories.
The two-view protocol (context edges vs. target edges, with a leakage-controlled inductive split) is described carefully and reflects good evaluation practice.
Major concerns.
### Question 1
Evaluation protocol may be lenient. MRR/AUC are computed with only K=20 sampled negatives. Since TGB and TGB-2.0 are cited and emphasize stricter negative sampling, the current setup weakens the SOTA claims. The authors should justify K=20 and report sensitivity to the number of negatives (e.g., 100 or all-candidate ranking on at least one dataset).
Novelty delta vs. closest baselines is not fully isolated. STHN and TAMI already address continuous-time heterogeneous / temporal interaction modeling. Beyond the ablation, a component-level, head-to-head comparison clarifying what DHC adds over STHN in particular would strengthen the contribution.
Text-attribute pathway is underspecified. The text-derived node features x_u are central to the inductive claims, but their source, encoder, and per-dataset dimensionality are not clearly reported. This limits reproducibility of the key inductive gains.

**Response EN.** We addressed all three issues. (1) The revised paper qualifies K=20 as a controlled sampled-ranking protocol, not official TGB SOTA; unavailable K=100/all-candidate results are not fabricated. (2) It now gives a component-level STHN/TAMI comparison and explains where recurrent relation state enters message generation, context, and scoring; the relation-memory ablation is not over-interpreted. (3) Every stored node vector is 128-dimensional. AMiner-Paper and DBLP concatenate 64 standardized structural/type coordinates with 64 deterministic hybrid-hash text coordinates (word 1--2 grams and character 2--4 grams). Patent-US uses 120 character 2--4-gram TF--IDF coordinates plus eight structural coordinates, derived from abstracts, organization names, and IPC descriptions. THGL-Software has no natural-language field; its vector is padded structural/type information. No pretrained language model is fine-tuned. We additionally disclose that the reported THGL ``inductive'' run followed the builder's temporal fallback and is not a strict unseen-node evaluation.

**中文回复。** 三点均已处理：(1) 将 K=20 限定为受控采样排序协议，不宣称官方 TGB SOTA，也不虚构 K=100/全候选结果；(2) 新增与 STHN/TAMI 的组件级对比；(3) 四个数据集节点向量均为 128 维。AMiner-Paper、DBLP 为 64 维结构/类型加 64 维混合哈希文本；Patent-US 为 120 维字符 2--4 gram TF--IDF 加八维结构特征，文本来自摘要、组织名和 IPC 描述；THGL-Software 无自然语言字段，仅为补齐的结构/类型特征。实验未微调预训练语言模型。我们还披露：表中 THGL “inductive” 运行采用了构造器的时间回退，不能视为严格未见节点评估。

Minor concerns.
### Question 2
Please double-check the manuscript metadata and references: one author ORCID appears to be a placeholder (0000-1111-2222-3333), several references carry unusual/future access or publication dates (e.g., "accessed 5 May 2026" and multiple 2026 entries), and some DOIs appear malformed. These should be corrected in the camera-ready.
Dense grammatical issues throughout (e.g., abstract "framework fuses", "branch compact chronological", "generateing"; Sec. 4 "In addition."). A thorough language pass is needed.
Figures 3 and 4 report gains without absolute values or variance; please add tables with std over seeds.
Some symbols (e.g., the edge-type flag η, relation weights w_r) are used before being fully introduced; forward references to Table 1 would help.
The case study (Fig. 5) is qualitative; where possible, tie observations to a quantitative metric.

**Response EN.** We removed the placeholder ORCID rather than guessing an identifier. We corrected the malformed prose in the touched sections, standardized dataset capitalization, added a forward clarification for the edge-type flag and optional relation weights, and corrected stale notation-table labels. Reference years/access dates must be checked against authoritative records before submission; we do not guess them. Current artifacts do not provide repeated seeds for every point in Figs. 3--4, so adding standard deviations would be false precision. We explicitly mark multi-seed variance as missing. The revised case study uses only saved aggregate MRR/AP artifacts and labels them aggregate cases rather than per-edge causal explanations.

**中文回复。** 我们删除了占位 ORCID，没有猜测替代编号；同时修正所触及章节的病句、统一数据集大小写、为边类型开关与可选关系权重增加前向说明，并纠正符号表中过时标签。参考文献年份/访问日期必须在投稿前按权威记录核验，我们不会猜测。当前产物未为图 3–4 的每个点提供多随机种子重复，强行添加标准差会造成虚假精度，因此修订稿明确将多种子方差列为缺失项。案例分析仅使用已保存的聚合 MRR/AP 产物，并标注为聚合案例，而非逐边因果解释。

Recommendation. Borderline, leaning weak accept. The method and experiments are solid and above the borderline bar, but the lenient negative-sampling protocol and the underspecified text-feature pipeline should be addressed. It is better if the authors (a) justify or strengthen the negative-sampling evaluation, (b) clarify the text-feature pipeline and its role in the inductive gains, and (c) more explicitly isolate the contribution over STHN/TAMI.

### Question 3

Please justify K=20 negatives and report MRR under more negatives (e.g., 100 / all-candidate).
What encoder produces x_u, and what is d_x for each dataset?
Could you clarify the unusual reference dates and the placeholder ORCID?

**Response EN.** K=20 keeps scoring tractable and identical across all rerun methods, but is not an official TGB sampler; no incomplete 100-negative/all-candidate rerun is reported. The dimensions are explicit: 64-dimensional hybrid hashing plus 64 structural/type dimensions for AMiner-Paper and DBLP; 120-dimensional character TF--IDF plus eight structural dimensions for Patent-US; and 128 structural/type dimensions for THGL-Software. The placeholder ORCID was removed. Unusual bibliography fields will be checked against authoritative records before submission; the malformed duplicated DOI prefix in the TAMI entry was corrected.

**中文回复。** K=20 保持各方法打分规模一致且可承受，但不是 TGB 官方采样器；不报告未完成的 100 负例/全候选结果。AMiner-Paper、DBLP 为 64 维混合哈希文本加 64 维结构/类型；Patent-US 为 120 维字符 TF--IDF 加八维结构；THGL-Software 为 128 维结构/类型。占位 ORCID 已删除；异常文献字段将在投稿前依据权威记录核验，TAMI 的重复 DOI 前缀已修正。
