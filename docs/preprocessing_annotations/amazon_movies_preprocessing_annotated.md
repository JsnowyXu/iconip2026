# amazon-movies 预处理代码逐行注释

## 原始代码位置

- 原始读取与统一图构建函数：[prepare_dataset.py:430](/home/xuxj/2026/dyn_relation_memory/scripts/datasets/prepare_dataset.py:430)
- CLI 数据集分发入口：[prepare_dataset.py:1642](/home/xuxj/2026/dyn_relation_memory/scripts/datasets/prepare_dataset.py:1642)
- 任务视图生成入口：[build_paper_task_views.py:272](/home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py:272)
- 流程记录：[amazon_movies_process.md:9](/home/xuxj/2026/dyn_relation_memory/amazon_movies_process.md:9)

## 执行链路

1. `prepare_dataset.py --dataset amazon_movies --archive /home/xuxj/2026/datasets/Amazon_movies.zip ...`
2. `main()` 根据 `--dataset amazon_movies` 调用 `parse_amazon_movies(...)`。
3. `parse_amazon_movies(...)` 从 zip 内 `edge_list.csv` 读取用户、商品、时间、评分，构造成统一动态图字段。
4. `to_unified(...)` 把 numpy 字段转成统一 `.pt` 文件 `amazon_movies_unified.pt`。
5. `build_paper_task_views.py` 再基于统一 `.pt` 生成 `amazon_movies_edge_transductive.pt`、`amazon_movies_edge_inductive.pt`、`amazon_movies_node_multilabel.pt`。

## 逐行注释：`parse_amazon_movies`

```python
def parse_amazon_movies(zip_path: Path, max_edges: int):  # 定义 Amazon Movies 的原始数据解析函数，输入是 zip 路径和最多读取的边数。
    with zipfile.ZipFile(zip_path) as z:  # 打开 Amazon_movies.zip，不先解压到磁盘，直接在压缩包里读文件。
        with z.open("edge_list.csv") as f:  # 打开 zip 内的 edge_list.csv，这是用户-电影交互边表。
            df = pd.read_csv(f, nrows=max_edges)  # 用 pandas 读取前 max_edges 行，项目里正式处理时限制为 200000 条边。

    # Columns in this dataset are commonly [index, u, r, i, ts, label, idx].  # 说明原始列：索引、用户、原始关系/评论、电影、时间、评分、边编号。
    # In the released archive, `r` is a per-review identifier with review text in  # 说明 r 不是稳定边类型，而是近似每条评论一个 id。
    # relation_text.csv, so treating it as a relation type collapses link tasks.  # 如果把 r 当边类型，会产生大量稀疏关系，不适合边预测。
    # We therefore fall back to the discrete rating label when `r` is effectively  # 所以当 r 看起来像唯一评论 id 时，改用评分 label 作为关系类型。
    # near-unique, which keeps one processed dataset usable across edge/node tasks.  # 这样处理后的图既能做边任务，也能做节点任务。
    src = df["u"].to_numpy(dtype=np.int64)  # 取用户列 u 作为源节点数组。
    dst = df["i"].to_numpy(dtype=np.int64)  # 取电影/商品列 i 作为目标节点数组。
    rel_raw = df["r"].to_numpy(dtype=np.int64)  # 取原始 r 列，用来判断它到底像关系类型还是评论 id。
    ts = df["ts"].to_numpy(dtype=np.float32)  # 取时间戳列 ts，转成 float32 作为动态图时间。
    rating = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else None  # 如果存在评分列 label，就取出来作为候选关系类型。

    rel_vals, rel_cnt = np.unique(rel_raw, return_counts=True)  # 统计原始 r 的不同取值和每个取值出现次数。
    rel_is_edge_id_like = rel_vals.size > 0 and (  # 判断 r 是否更像每条边独有的 id。
        rel_vals.size >= max(32, int(0.2 * max(1, rel_raw.shape[0])))  # 条件一：r 的不同取值数量占边数比例很高。
        or int(rel_cnt.max()) <= 16  # 条件二：任何一个 r 出现次数都很少，说明它不是稳定关系类别。
    )  # 结束 r 是否 edge-id-like 的判断表达式。
    if rating is not None:  # 如果原始数据里有评分 label。
        rating_vals = np.unique(rating)  # 统计评分 label 的不同取值。
    else:  # 如果没有评分列。
        rating_vals = np.array([], dtype=np.int64)  # 用空数组占位，后面不会选择评分关系。

    if rel_is_edge_id_like and 1 < rating_vals.size <= 10:  # 当 r 像评论 id，且评分是少量离散类别时。
        rel_source = rating  # 使用评分 label 作为边关系来源。
        rel_source_name = "rating_label"  # 在元信息里记录关系来自 rating label。
    else:  # 当 r 不像评论 id，或者评分不可用时。
        rel_source = rel_raw  # 直接使用原始 r 作为关系来源。
        rel_source_name = "raw_relation_id"  # 在元信息里记录关系来自原始 r。

    rel_unique = {r: i for i, r in enumerate(sorted(np.unique(rel_source).tolist()))}  # 把原始关系值重编号为 0..R-1。
    rel = np.array([rel_unique[r] for r in rel_source], dtype=np.int64)  # 为每条边生成连续关系 id。

    n = int(max(src.max(), dst.max()) + 1)  # 节点总数用最大节点 id 加 1 推断，保持原始 id 空间。
    node_type = np.zeros(n, dtype=np.int64)  # 初始化所有节点类型为 0，表示用户。
    node_type[dst] = 1  # 所有出现在目标端的节点标记为 1，表示电影/商品。
    feat = build_degree_features(n, src, dst, ts, node_type, dim=64)  # 基于入度、出度、时间和节点类型构造 64 维结构特征。

    # Multiclass node label: dominant relation per source node.  # 节点多分类标签定义为用户最常见的评分/关系类型。
    rel_counter = [Counter() for _ in range(n)]  # 为每个节点准备一个关系计数器。
    for u, r in zip(src, rel):  # 遍历每条用户到电影的交互边。
        rel_counter[u][int(r)] += 1  # 给源用户节点累计该关系/评分出现次数。
    mc = np.zeros(n, dtype=np.int64)  # 初始化节点多分类标签，默认类别为 0。
    for i, c in enumerate(rel_counter):  # 遍历每个节点的关系计数器。
        if c:  # 如果这个节点作为用户发出过边。
            mc[i] = c.most_common(1)[0][0]  # 取出现最多的关系/评分作为该节点类别。

    ml = build_default_multilabel(n, src, dst, ts)  # 构造默认节点多标签目标，用于节点多标签任务视图。
    in_deg = np.bincount(dst, minlength=n).astype(np.int64)  # 统计每个节点作为目标电影被交互的次数。
    out_deg = np.bincount(src, minlength=n).astype(np.int64)  # 统计每个节点作为用户发出交互的次数。
    node_text = []  # 准备每个节点的文本描述，用于追加 hash/text 特征。
    for i in range(n):  # 遍历所有节点 id。
        ntype = "item" if int(node_type[i]) == 1 else "user"  # 根据 node_type 把节点解释为 item 或 user。
        node_text.append(f"{ntype} id_{i} indeg_{int(in_deg[i])} outdeg_{int(out_deg[i])}")  # 拼出节点文本摘要。
    feat = append_text_features(feat, node_text, text_dim=TEXT_DIM)  # 把文本编码特征拼接到结构特征后面。

    return {  # 返回中间数据字典，后续由 to_unified 转成统一 .pt。
        "name": "amazon_movies",  # 数据集名称。
        "src": src,  # 所有边的源节点数组。
        "dst": dst,  # 所有边的目标节点数组。
        "ts": ts,  # 所有边的时间戳数组。
        "rel": rel,  # 所有边的连续关系类型数组。
        "node_type": node_type,  # 每个节点的类型：0 用户，1 商品/电影。
        "node_feat": feat,  # 每个节点的最终特征。
        "node_multiclass": mc,  # 每个节点的多分类标签。
        "node_multilabel": ml,  # 每个节点的多标签标签。
        "meta_extra": {  # 额外元信息，写入统一图 meta.extra。
            **text_encoder_meta(),  # 记录文本编码器类型、维度等配置。
            "relation_source": rel_source_name,  # 记录本次边类型来自评分 label 还是原始 r。
            "raw_relation_unique": int(rel_vals.size),  # 记录原始 r 有多少不同取值。
            "edge_relation_unique": int(np.unique(rel).size),  # 记录最终边关系类型数量。
        },  # 结束 meta_extra。
    }  # 结束返回字典。
```

## 逐行注释：`prepare_dataset.py` 中的 CLI 分发

```python
def main():  # 命令行入口，负责根据 --dataset 选择具体解析函数。
    global TEXT_ENCODER, TEXT_DIM, TEXT_MODEL_NAME, TEXT_BATCH_SIZE  # 声明文本编码相关全局参数会在 main 中被命令行参数覆盖。
    p = argparse.ArgumentParser("Prepare dataset into unified .pt format")  # 创建命令行参数解析器。
    p.add_argument("--dataset", default='aminer_paper', required=True, choices=["amazon_movies", "raw_cn", "raw_us", "dblp", "aminer_paper"])  # 限定可处理的数据集，Amazon 用 amazon_movies。
    p.add_argument("--archive", required=True, type=str, default="/home/xuxj/2026/datasets/Aminer/AMiner-Paper.rar")  # 原始数据路径；Amazon 传入 Amazon_movies.zip。
    p.add_argument("--max-edges", type=int, default=200000)  # 最多读取多少条边，Amazon 正式流程使用 200000。
    p.add_argument("--out-pt", required=True, type=str, default="data/processed/aminer_paper_unified_debug.pt")  # 输出统一 .pt 文件路径。
    p.add_argument("--text-encoder", type=str, default="hybrid_hash", choices=["hybrid_hash", "hash_word", "sentence_transformer", "none"], help="Text encoder backend. hybrid_hash is Chinese-friendly (word+char ngram).")  # 选择节点文本特征编码器。
    p.add_argument("--text-dim", type=int, default=64, help="Output dimension for text features.")  # 设置追加文本特征维度。
    p.add_argument("--text-model-name", type=str, default="", help="Model name/path for sentence_transformer backend.")  # 如果用 sentence_transformer，这里给模型名。
    p.add_argument("--text-batch-size", type=int, default=128, help="Batch size for sentence_transformer encoding.")  # 文本编码批大小。
    p.add_argument("--aminer-min-year", type=int, default=2002)  # AMiner 专用参数，Amazon 不使用。
    p.add_argument("--aminer-max-year", type=int, default=2012)  # AMiner 专用参数，Amazon 不使用。
    p.add_argument("--export-baselines", action="store_true")  # 是否导出 TGN/TGB2 等 baseline 适配格式。
    p.add_argument("--baseline-out-dir", type=str, default="")  # baseline 适配文件输出目录。
    args = p.parse_args()  # 解析命令行参数。

    TEXT_ENCODER = str(args.text_encoder).strip().lower()  # 标准化文本编码器名称。
    TEXT_DIM = int(max(8, args.text_dim))  # 文本特征维度至少为 8。
    TEXT_MODEL_NAME = str(args.text_model_name).strip()  # 保存 sentence_transformer 模型名。
    TEXT_BATCH_SIZE = int(max(1, args.text_batch_size))  # 文本批大小至少为 1。

    archive = Path(args.archive)  # 把原始数据路径转成 Path。
    out_pt = Path(args.out_pt)  # 把输出 .pt 路径转成 Path。

    if args.dataset == "amazon_movies":  # 当命令行指定 Amazon Movies。
        d = parse_amazon_movies(archive, args.max_edges)  # 调用上面的 Amazon 原始解析函数。
    elif args.dataset == "raw_cn":  # 中国专利旧 raw_cn 分支。
        d = parse_raw_patent(archive, "CN", args.max_edges)  # 调用旧专利解析函数。
    elif args.dataset == "raw_us":  # 美国专利旧 raw_us 分支。
        d = parse_raw_patent(archive, "US", args.max_edges)  # 调用旧专利解析函数。
    elif args.dataset == "dblp":  # DBLP 分支。
        d = parse_dblp(archive, args.max_edges)  # 调用 DBLP 解析函数。
    elif args.dataset == "aminer_paper":  # AMiner Paper 分支。
        d = parse_aminer_paper(archive, args.max_edges, min_year=args.aminer_min_year, max_year=args.aminer_max_year)  # 调用 AMiner 解析函数。
    else:  # 理论上不会走到这里，因为 argparse choices 已限制。
        raise ValueError(args.dataset)  # 如果数据集名异常，直接报错。

    obj = to_unified(d, out_pt)  # 把解析出的 numpy 数据统一封装成项目标准 .pt 图对象。
    print(f"saved unified dataset to {out_pt}")  # 打印保存路径。
    print(f"nodes={obj['num_nodes']} edges={obj['edges']['src'].shape[0]} rels={obj['num_rels']}")  # 打印节点数、边数、关系数。
```

## 逐行注释：任务视图生成核心

```python
def choose_target_relation(rel: np.ndarray, target_rel: int = None) -> int:  # 选择边预测任务的目标关系类型。
    if target_rel is not None and int(target_rel) >= 0:  # 如果命令行显式指定了 target relation。
        return int(target_rel)  # 直接使用指定关系。
    vals, cnt = np.unique(rel, return_counts=True)  # 否则统计每个关系类型的边数。
    if vals.size == 0:  # 如果统一图里没有边。
        return 0  # 返回 0 作为兜底关系 id。
    best_rel = int(vals[np.argmax(cnt)])  # 默认选择边数最多的关系作为监督目标。
    return best_rel  # 返回选中的目标关系 id。

def build(input_pt: Path, out_dir: Path, prefix: str, target_rel: int = None):  # 从统一 .pt 构建 transductive/inductive/node 三类任务视图。
    obj = torch.load(input_pt, map_location="cpu")  # 读取 `amazon_movies_unified.pt` 到 CPU。
    rel = obj["edges"]["rel"].numpy().astype(np.int64)  # 取全部边关系类型。
    ts = obj["edges"]["ts"].numpy().astype(np.float32)  # 取全部边时间戳。
    src = obj["edges"]["src"].numpy().astype(np.int64)  # 取全部边源节点。
    dst = obj["edges"]["dst"].numpy().astype(np.int64)  # 取全部边目标节点。

    target_rel_id = choose_target_relation(rel, target_rel=target_rel)  # 选择要预测的关系类型，Amazon 默认是最多的评分关系。
    target_old = sort_old_indices(np.where(rel == int(target_rel_id))[0].astype(np.int64), ts)  # 找出目标关系的所有原始边，并按时间排序。
    if target_old.size == 0:  # 如果没有目标边。
        raise RuntimeError(f"no edges found for target relation id={target_rel_id}")  # 直接报错，避免生成空任务。

    trans_tr_old, trans_va_old, trans_te_old = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # 对目标边按时间切 train/val/test。
    trans_view = build_edge_view(obj=obj, out_name=f"{prefix}_edge_transductive", target_old=target_old, train_target_old=trans_tr_old, val_target_old=trans_va_old, test_target_old=trans_te_old, extra_info={"target_original_rel_id": int(target_rel_id), "protocol": "full_hetero_context_transductive"})  # 构建转导边预测视图，训练/验证/测试都来自同一节点集合的时间切分。

    ind_tr_old, ind_va_old, ind_te_old, unseen_nodes = inductive_partition_target(target_old=target_old, src=src, ts=ts, rng=np.random.RandomState(42))  # 构建归纳切分：按源节点留出一批测试节点。
    keep_mask = ~(np.isin(src, unseen_nodes) | np.isin(dst, unseen_nodes))  # 从上下文边里移除触碰 unseen 节点的边，防止归纳泄漏。
    ind_view = build_edge_view(obj=obj, out_name=f"{prefix}_edge_inductive", target_old=target_old, train_target_old=ind_tr_old, val_target_old=ind_va_old, test_target_old=ind_te_old, context_keep_mask=keep_mask, extra_info={"target_original_rel_id": int(target_rel_id), "protocol": "full_hetero_context_inductive", "unseen_source_node_ids": unseen_nodes.tolist()})  # 构建归纳边预测视图。

    node_view = build_sorted_full_view(obj, out_name=f"{prefix}_node_multilabel")  # 构建节点多标签任务视图，使用全关系时序图。

    out_dir.mkdir(parents=True, exist_ok=True)  # 确保任务输出目录存在。
    p1 = out_dir / f"{prefix}_edge_transductive.pt"  # 转导边任务输出路径。
    p2 = out_dir / f"{prefix}_edge_inductive.pt"  # 归纳边任务输出路径。
    p3 = out_dir / f"{prefix}_node_multilabel.pt"  # 节点多标签任务输出路径。
    torch.save(trans_view, p1)  # 保存转导边任务。
    torch.save(ind_view, p2)  # 保存归纳边任务。
    torch.save(node_view, p3)  # 保存节点多标签任务。
    return [p1, p2, p3]  # 返回三个生成文件路径。
```

## 熟悉时重点看

- `u -> i` 是用户到电影/商品的时序交互边。
- 原始 `r` 更像评论 id，因此代码通常用 `label` 评分生成 5 类关系。
- `node_type=0/1` 分别对应 user/item。
- 边任务默认选边数最多的评分关系作为 `target_rel_id`。
- `edge_transductive` 是目标关系按时间切分；`edge_inductive` 会额外留出一批源用户节点，并从上下文中移除这些节点相关边。
