# DBLP 预处理源码注释

本文只标注 DBLP 数据从原始压缩包到论文任务视图的预处理链路。源码位置基于当前仓库：

- `/home/xuxj/2026/dyn_relation_memory/scripts/datasets/prepare_dataset.py`
- `/home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py`

## 执行链路

1. `prepare_dataset.py:1642-1705 main()`
   - 命令行传入 `--dataset dblp --archive ... --out-pt ...`。
   - `main()` 在 `elif args.dataset == "dblp"` 分支调用 `parse_dblp(archive, args.max_edges)`。
   - `parse_dblp()` 返回 numpy 格式的统一前置字典。
   - `to_unified(d, out_pt)` 按时间排序边、生成通用 `edge_feat/split/meta`，保存为 `dblp_unified.pt`。

2. `build_paper_task_views.py:332-352 main()`
   - 命令行传入 `--input data/processed/dblp_unified.pt --out-dir data/processed/paper_tasks --name-prefix dblp`。
   - `main()` 调用 `build(input_pt, out_dir, prefix, target_rel)`。
   - `build()` 读取统一图，选择目标关系，生成：
     - `dblp_edge_transductive.pt`
     - `dblp_edge_inductive.pt`
     - `dblp_node_multilabel.pt`

## 重点数据流

- 原始记录：`DBLP-Citation-network-V18.jsonl`，每行一个论文 JSON。
- 节点：
  - `paper:{pid}` 映射为论文节点，`node_type=0`。
  - `author:{aid}` 映射为作者节点，`node_type=1`。
- 边：
  - 作者写论文：`author_write_paper -> rel=0`，方向 `author -> paper`，时间戳为论文年份。
  - 论文引用论文：`paper_cite_paper -> rel=1`，方向 `paper -> referenced_paper`，时间戳为引用论文年份。
- 节点特征：
  - `build_degree_features()` 先构造度数、首次/末次出现时间、节点类型等结构特征。
  - `append_text_features()` 再把标题、venue、年份、作者名等文本哈希特征拼到结构特征后。
- 节点任务标签：
  - `node_multiclass`：基于论文年份三分位生成 3 桶；没有年份的作者节点默认落入低桶。
  - `node_multilabel`：`build_default_multilabel()` 基于总度、入度、出度、早期出现节点生成 4 维多标签。
- 统一保存：
  - `to_unified()` 对所有边按 `ts` 升序排序。
  - `edge_feat=[ts_norm, rel_norm]`。
  - `split=train/val/test` 是全图按时间的 70/15/15 划分。
- paper task view：
  - 默认 `target_rel=-1` 时，`choose_target_relation()` 选择边数最多的关系作为监督目标。
  - DBLP 里监督边通常会落在作者写论文或论文引用之一，实际由统一图中 `rel` 计数决定。
  - transductive view 保留完整异构上下文，监督目标只打在目标关系边上。
  - inductive view 从目标关系的源节点中抽取 unseen source nodes，并过滤所有接触这些节点的上下文边。
  - node_multilabel view 对全关系按时间排序，用同一套 `build_edge_view()` 结构保存，标记 `node_task_view=True`。

## `prepare_dataset.py` 支撑函数

### `build_default_multilabel`：默认节点多标签

原始位置：`scripts/datasets/prepare_dataset.py:124-155`

```python
def build_default_multilabel(num_nodes: int, src: np.ndarray, dst: np.ndarray, ts: np.ndarray) -> np.ndarray:
    # 定义默认节点多标签构造函数；输入是节点数、边源点、边终点和边时间。
    in_deg = np.bincount(dst, minlength=num_nodes)
    # 统计每个节点作为终点出现的次数，即入度。
    out_deg = np.bincount(src, minlength=num_nodes)
    # 统计每个节点作为源点出现的次数，即出度。
    total = in_deg + out_deg
    # 总度等于入度加出度，用来描述节点总体活跃度。

    def positive_quantile_mask(values: np.ndarray, q: float) -> np.ndarray:
        # 内部工具：只在正值节点中计算分位数，避免大量 0 度节点影响阈值。
        values = np.asarray(values, dtype=np.float32)
        # 将输入转为 float32，方便后续数值计算和保存。
        mask = np.zeros(values.shape[0], dtype=np.float32)
        # 初始化 0/1 标签列。
        active = values > 0
        # 只把值大于 0 的节点视为活跃节点。
        if not np.any(active):
            # 如果没有任何活跃节点，直接返回全 0 标签。
            return mask
        thr = np.quantile(values[active], q)
        # 在活跃节点上计算 q 分位阈值。
        mask[active] = (values[active] >= thr).astype(np.float32)
        # 活跃节点中高于阈值的标为 1。
        return mask
        # 返回这一列标签。

    labels = np.zeros((num_nodes, 4), dtype=np.float32)
    # 初始化 4 维多标签矩阵。
    labels[:, 0] = positive_quantile_mask(total, 0.75)
    # 第 0 维：总度位于活跃节点前 25%。
    labels[:, 1] = positive_quantile_mask(in_deg, 0.75)
    # 第 1 维：入度位于活跃节点前 25%。
    labels[:, 2] = positive_quantile_mask(out_deg, 0.75)
    # 第 2 维：出度位于活跃节点前 25%。

    node_first = np.full(num_nodes, np.inf, dtype=np.float32)
    # 为每个节点记录首次出现时间；初始为无穷大。
    for u, v, t in zip(src, dst, ts):
        # 遍历每条边，同时看源点和终点。
        node_first[u] = min(node_first[u], t)
        # 更新源点首次出现时间。
        node_first[v] = min(node_first[v], t)
        # 更新终点首次出现时间。

    active = np.isfinite(node_first)
    # 能在任一边中出现的节点才有有限首次时间。
    if np.any(active):
        # 如果存在活跃节点，构造早期出现标签。
        thr = np.quantile(node_first[active], 0.25)
        # 取首次出现时间最早 25% 的时间阈值。
        labels[active, 3] = (node_first[active] <= thr).astype(np.float32)
        # 第 3 维：早期出现节点标 1。
    return labels
    # 返回形状为 [num_nodes, 4] 的节点多标签。
```

### `append_text_features`：拼接文本特征

原始位置：`scripts/datasets/prepare_dataset.py:362-368`

```python
def append_text_features(base_feat: np.ndarray, node_text: List[str], text_dim: int = 64) -> np.ndarray:
    # 将结构特征和文本编码特征拼接，DBLP 节点文本来自论文标题/venue/年份或作者名。
    if base_feat.shape[0] == 0:
        # 没有节点时无需编码文本。
        return base_feat.astype(np.float32)
        # 保持 float32 输出格式。
    if len(node_text) != base_feat.shape[0]:
        # 如果文本条数和节点数不一致，做截断或补缺省文本，避免拼接维度错误。
        node_text = (node_text + ["[MISSING]"] * base_feat.shape[0])[: base_feat.shape[0]]
        # 补齐后再截到节点数量。
    txt = build_text_features(node_text, dim=text_dim)
    # 使用全局 `TEXT_ENCODER` 指定的编码器生成文本特征，默认是 hybrid_hash。
    return np.concatenate([base_feat.astype(np.float32), txt.astype(np.float32)], axis=1).astype(np.float32)
    # 沿特征维拼接结构特征和文本特征，并确保最终是 float32。
```

## `prepare_dataset.py` DBLP 解析函数

原始位置：`scripts/datasets/prepare_dataset.py:868-977`

```python
def parse_dblp(zip_path: Path, max_edges: int):
    # DBLP 专用解析入口；输入原始 zip 路径和最多保留的边数。
    node_map = {}
    # 字符串节点键到连续整数 id 的映射，例如 paper:xxx 或 author:yyy。
    node_text_map: Dict[int, str] = {}
    # 节点 id 到文本描述的映射，用于后续文本特征。

    def get_node(k: str):
        # 将外部节点键转换为内部连续整数 id。
        if k not in node_map:
            # 第一次看到该节点键时分配新 id。
            node_map[k] = len(node_map)
            # 新 id 等于当前已见节点数量。
        return node_map[k]
        # 返回该节点键对应的内部 id。

    def merge_text(nid: int, text: str) -> None:
        # 将某个节点的文本信息合并到 node_text_map。
        t = str(text).strip()
        # 文本统一转字符串并去掉首尾空白。
        if not t:
            # 空文本不参与特征构造。
            return
        old = node_text_map.get(int(nid), "")
        # 取该节点已有文本；没有则为空字符串。
        if old:
            # 如果已有文本，需要去重式追加。
            if t not in old:
                # 只有新片段不在旧文本中时才追加。
                node_text_map[int(nid)] = (old + " " + t)[:256]
                # 合并后截断到 256 字符，限制文本特征输入长度。
        else:
            # 如果节点还没有文本，直接写入。
            node_text_map[int(nid)] = t[:256]
            # 首次写入也限制为 256 字符。

    rel_map = {"author_write_paper": 0, "paper_cite_paper": 1}
    # DBLP 只定义两类关系：作者写论文为 0，论文引用论文为 1。
    src, dst, rel, ts = [], [], [], []
    # 四个边数组：源节点、目标节点、关系 id、时间戳。
    paper_year = {}
    # 记录论文节点的年份，用于后续构造 node_multiclass。

    with zipfile.ZipFile(zip_path) as z:
        # 打开 DBLP 原始 zip。
        with z.open("DBLP-Citation-network-V18.jsonl") as f:
            # 从 zip 内部读取固定文件名的 jsonl。
            for li, line in enumerate(f):
                # 逐行读取；li 是行号，也会作为缺失 id/year 的 fallback。
                if len(src) >= max_edges:
                    # 达到最大边数后停止，避免解析过多边。
                    break
                try:
                    # JSONL 每行是一篇论文记录。
                    obj = json.loads(line.decode("utf-8", errors="ignore"))
                    # 用 utf-8 解码，错误字符忽略。
                except json.JSONDecodeError:
                    # 坏行直接跳过。
                    continue

                pid = str(obj.get("id", f"paper_{li}"))
                # 论文外部 id；缺失时用行号构造 paper_{li}。
                p = get_node(f"paper:{pid}")
                # 取得论文节点内部 id。
                year = float(obj.get("year", li))
                # 论文年份；缺失时退回行号，仍能作为排序时间。
                paper_year[p] = year
                # 保存该论文节点年份。
                title = str(obj.get("title", "")).strip()
                # 读取论文标题文本。
                venue = str(obj.get("venue", "")).strip()
                # 读取 venue 文本。
                merge_text(p, f"{title} {venue} year_{int(year)}")
                # 将标题、venue、年份合并为论文节点文本。

                for a in obj.get("authors", []) or []:
                    # 遍历作者列表；None 时退化为空列表。
                    if isinstance(a, dict):
                        # 作者字段可能是 dict，包含 id/name。
                        aid = str(a.get("id", ""))
                        # 作者 id。
                        aname = str(a.get("name", "")).strip()
                        # 作者名。
                    else:
                        # 作者字段也可能是普通字符串。
                        aid = str(a).strip()
                        # 用字符串自身作为作者 id。
                        aname = str(a).strip()
                        # 用字符串自身作为作者名。
                    if not aid:
                        # 没有作者 id 就无法稳定建节点，跳过。
                        continue
                    au = get_node(f"author:{aid}")
                    # 取得作者节点内部 id。
                    merge_text(au, f"author {aname if aname else aid}")
                    # 给作者节点合并文本描述。
                    src.append(au)
                    # 作者写论文边的源节点是作者。
                    dst.append(p)
                    # 作者写论文边的目标节点是论文。
                    rel.append(rel_map["author_write_paper"])
                    # 关系 id 写为 0。
                    ts.append(year)
                    # 边时间戳使用论文年份。
                    if len(src) >= max_edges:
                        # 作者边追加后再次检查边数上限。
                        break
                if len(src) >= max_edges:
                    # 如果作者边已经达到上限，不再处理引用边。
                    break

                for ref in obj.get("references", []) or []:
                    # 遍历当前论文引用的论文 id。
                    rp = get_node(f"paper:{ref}")
                    # 为被引用论文取得或创建论文节点。
                    src.append(p)
                    # 引用边源节点是当前论文。
                    dst.append(rp)
                    # 引用边目标节点是被引用论文。
                    rel.append(rel_map["paper_cite_paper"])
                    # 关系 id 写为 1。
                    ts.append(year)
                    # 引用事件时间使用当前论文年份。
                    if len(src) >= max_edges:
                        # 引用边追加后检查边数上限。
                        break

    src = np.array(src[:max_edges], dtype=np.int64)
    # 将源节点列表截断到 max_edges 并转 int64 numpy 数组。
    dst = np.array(dst[:max_edges], dtype=np.int64)
    # 将终点列表截断到 max_edges 并转 int64。
    rel = np.array(rel[:max_edges], dtype=np.int64)
    # 将关系列表截断到 max_edges 并转 int64。
    ts = np.array(ts[:max_edges], dtype=np.float32)
    # 将时间戳列表截断到 max_edges 并转 float32。
    n = len(node_map)
    # 节点总数等于 node_map 中已分配的节点数。

    node_type = np.zeros(n, dtype=np.int64)
    # 初始化节点类型数组，默认 0。
    for k, i in node_map.items():
        # 遍历所有节点键和内部 id。
        node_type[i] = 0 if k.startswith("paper:") else 1
        # paper 节点类型为 0，author 节点类型为 1。

    feat = build_degree_features(n, src, dst, ts, node_type, dim=64)
    # 先用度数、时间和节点类型构造 64 维结构特征。
    node_text = []
    # 准备按节点 id 顺序排列的文本列表。
    for i in range(n):
        # 对每个内部节点 id 取文本。
        txt = node_text_map.get(i, "")
        # 优先使用解析过程中合并的文本。
        if not txt:
            # 如果没有文本，构造兜底文本。
            txt = f"paper node_{i}" if int(node_type[i]) == 0 else f"author node_{i}"
            # 论文节点和作者节点使用不同 fallback 文本。
        node_text.append(txt)
        # 加入文本列表，顺序必须与节点 id 对齐。
    feat = append_text_features(feat, node_text, text_dim=TEXT_DIM)
    # 将文本特征拼接到结构特征之后。

    years = np.array([paper_year.get(i, 0.0) for i in range(n)], dtype=np.float32)
    # 取每个节点对应的论文年份；作者节点或未记录年份节点为 0。
    if np.any(years > 0):
        # 如果至少有一个论文年份有效。
        bins = np.quantile(years[years > 0], [0.33, 0.66])
        # 在有效年份上计算 33% 和 66% 分位点。
        mc = np.digitize(years, bins).astype(np.int64)
        # 将所有节点按年份分桶成 0/1/2；年份 0 会落入低桶。
    else:
        # 如果没有任何有效年份。
        mc = np.zeros(n, dtype=np.int64)
        # 多分类标签全置 0。
    ml = build_default_multilabel(n, src, dst, ts)
    # 基于图结构和时间生成默认 4 维多标签。

    return {
        # 返回尚未 torch 化的统一前置对象，交给 to_unified 继续处理。
        "name": "dblp_v18",
        # 数据集内部名称，后续写入 meta.dataset_name。
        "src": src,
        # 源节点数组。
        "dst": dst,
        # 目标节点数组。
        "ts": ts,
        # 时间戳数组。
        "rel": rel,
        # 关系 id 数组。
        "node_type": node_type,
        # 节点类型数组。
        "node_feat": feat,
        # 节点特征矩阵，包含结构特征和文本特征。
        "node_multiclass": mc,
        # 节点多分类标签。
        "node_multilabel": ml,
        # 节点多标签标签。
        "meta_extra": text_encoder_meta(),
        # 保存文本编码器配置到元信息。
    }
```

## `prepare_dataset.py` 统一保存入口

原始位置：`scripts/datasets/prepare_dataset.py:1533-1597`

```python
def to_unified(dataset_obj: Dict, out_pt: Path):
    # 将 parse_dblp 返回的 numpy 字典转成项目统一的 torch .pt 格式。
    src = dataset_obj["src"].astype(np.int64)
    # 读取并规范化源节点数组类型。
    dst = dataset_obj["dst"].astype(np.int64)
    # 读取并规范化目标节点数组类型。
    ts = dataset_obj["ts"].astype(np.float32)
    # 读取并规范化时间戳数组类型。
    rel = dataset_obj["rel"].astype(np.int64)
    # 读取并规范化关系 id 数组类型。

    order = np.argsort(ts)
    # 按时间戳升序得到边排序索引。
    src = src[order]
    # 源节点按时间重排。
    dst = dst[order]
    # 目标节点按时间重排。
    ts = ts[order]
    # 时间戳按时间重排。
    rel = rel[order]
    # 关系 id 按时间重排，保持每条边字段对齐。

    num_nodes = dataset_obj["node_feat"].shape[0]
    # 节点数来自节点特征矩阵行数。
    num_rels = int(rel.max()) + 1 if rel.size > 0 else 1
    # 关系数为最大关系 id 加 1；空图时至少为 1。

    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)
    # 将时间戳归一化到近似 [0,1]。
    rel_norm = rel / max(1, num_rels - 1)
    # 将关系 id 归一化，DBLP 两关系时 rel=0/1。
    edge_feat = np.stack([ts_norm, rel_norm], axis=1).astype(np.float32)
    # 构造 2 维边特征：[归一化时间, 归一化关系]。

    tr, va, te = temporal_split(ts)
    # 在全图时间序列上做 70/15/15 train/val/test 划分。

    mc_node_ids = dataset_obj.get("node_multiclass_node_ids", np.arange(num_nodes, dtype=np.int64))
    # 多分类节点 id；DBLP 没单独提供时默认覆盖所有节点。
    mc_labels = dataset_obj["node_multiclass"]
    # 多分类标签。
    ml_node_ids = dataset_obj.get("node_multilabel_node_ids", np.arange(num_nodes, dtype=np.int64))
    # 多标签节点 id；DBLP 默认覆盖所有节点。
    ml_labels = dataset_obj["node_multilabel"]
    # 多标签标签矩阵。

    obj = {
        # 统一 .pt 对象主体。
        "num_nodes": int(num_nodes),
        # 节点数量。
        "num_rels": int(num_rels),
        # 关系数量。
        "num_node_types": int(dataset_obj["node_type"].max()) + 1,
        # 节点类型数量。
        "node_feat": torch.from_numpy(dataset_obj["node_feat"].astype(np.float32)),
        # 节点特征转为 torch Tensor。
        "node_type": torch.from_numpy(dataset_obj["node_type"].astype(np.int64)),
        # 节点类型转为 torch Tensor。
        "edges": {
            # 边字段集合。
            "src": torch.from_numpy(src),
            # 时间排序后的源节点。
            "dst": torch.from_numpy(dst),
            # 时间排序后的目标节点。
            "ts": torch.from_numpy(ts),
            # 时间排序后的时间戳。
            "rel": torch.from_numpy(rel),
            # 时间排序后的关系 id。
            "edge_feat": torch.from_numpy(edge_feat),
            # 边特征。
            "edge_label": torch.ones(src.shape[0], dtype=torch.float32),
            # 统一图阶段所有边都标为正例/存在边。
        },
        "split": {
            # 全图时间划分索引，索引指向排序后的边数组。
            "train": torch.from_numpy(tr.astype(np.int64)),
            # 训练边索引。
            "val": torch.from_numpy(va.astype(np.int64)),
            # 验证边索引。
            "test": torch.from_numpy(te.astype(np.int64)),
            # 测试边索引。
        },
        "node_multiclass": {
            # 节点多分类任务字段。
            "node_ids": torch.from_numpy(np.asarray(mc_node_ids, dtype=np.int64)),
            # 有多分类标签的节点 id。
            "labels": torch.from_numpy(np.asarray(mc_labels, dtype=np.int64)),
            # 多分类标签 tensor。
        },
        "node_multilabel": {
            # 节点多标签任务字段。
            "node_ids": torch.from_numpy(np.asarray(ml_node_ids, dtype=np.int64)),
            # 有多标签标签的节点 id。
            "labels": torch.from_numpy(np.asarray(ml_labels, dtype=np.float32)),
            # 多标签矩阵 tensor。
        },
        "meta": {
            # 元信息。
            "dataset_name": dataset_obj["name"],
            # DBLP 为 dblp_v18。
            "num_edges": int(src.shape[0]),
            # 边数。
            "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
            # 记录默认划分比例。
            "extra": dataset_obj.get("meta_extra", {}),
            # 保存 parse_dblp 传来的文本编码器等额外信息。
        },
    }

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    # 确保输出目录存在。
    torch.save(obj, out_pt)
    # 将统一对象保存为 .pt。
    return obj
    # 返回对象给 main 打印统计或导出 baseline。
```

## `prepare_dataset.py` DBLP main 分支

原始位置：`scripts/datasets/prepare_dataset.py:1642-1705`

```python
def main():
    # 命令行入口，负责解析参数并分派到具体数据集解析函数。
    global TEXT_ENCODER, TEXT_DIM, TEXT_MODEL_NAME, TEXT_BATCH_SIZE
    # 这些全局变量控制文本编码器行为，会被 parse_dblp -> append_text_features 使用。
    p = argparse.ArgumentParser("Prepare dataset into unified .pt format")
    # 创建参数解析器。
    p.add_argument("--dataset", default='aminer_paper', required=True, choices=["amazon_movies", "raw_cn", "raw_us", "dblp", "aminer_paper"]) #"kg4ai"
    # 数据集选择参数；DBLP 必须传 --dataset dblp。
    p.add_argument("--archive", required=True, type=str, default="/home/xuxj/2026/datasets/Aminer/AMiner-Paper.rar")
    # 原始归档路径；DBLP 时应指向包含 DBLP-Citation-network-V18.jsonl 的 zip。
    p.add_argument("--max-edges", type=int, default=200000)
    # 最大解析边数，parse_dblp 会在作者边/引用边累计达到该值后停止。
    p.add_argument("--out-pt", required=True, type=str, default="data/processed/aminer_paper_unified_debug.pt")
    # 统一 .pt 输出路径，DBLP 常见输出是 data/processed/dblp_unified.pt。
    p.add_argument("--text-encoder", type=str, default="hybrid_hash", choices=["hybrid_hash", "hash_word", "sentence_transformer", "none"], help="Text encoder backend. hybrid_hash is Chinese-friendly (word+char ngram).")
    # 文本编码器选择；默认 hybrid_hash。
    p.add_argument("--text-dim", type=int, default=64, help="Output dimension for text features.")
    # 文本特征维度。
    p.add_argument("--text-model-name", type=str, default="", help="Model name/path for sentence_transformer backend.")
    # sentence_transformer 后端的模型名或路径；DBLP 默认不用。
    p.add_argument("--text-batch-size", type=int, default=128, help="Batch size for sentence_transformer encoding.")
    # sentence_transformer 批大小；hash 后端基本不受影响。
    p.add_argument("--aminer-min-year", type=int, default=2002)
    # AMiner 专用参数；DBLP 分支不使用。
    p.add_argument("--aminer-max-year", type=int, default=2012)
    # AMiner 专用参数；DBLP 分支不使用。
    p.add_argument("--export-baselines", action="store_true")
    # 可选导出 baseline 适配格式；DBLP 主统一 .pt 不依赖它。
    p.add_argument("--baseline-out-dir", type=str, default="")
    # baseline 导出目录。
    args = p.parse_args()
    # 解析命令行参数。

    TEXT_ENCODER = str(args.text_encoder).strip().lower()
    # 更新全局文本编码器。
    TEXT_DIM = int(max(8, args.text_dim))
    # 更新全局文本维度，最低 8。
    TEXT_MODEL_NAME = str(args.text_model_name).strip()
    # 更新 sentence_transformer 模型名。
    TEXT_BATCH_SIZE = int(max(1, args.text_batch_size))
    # 更新文本编码 batch size。

    archive = Path(args.archive)
    # 将原始归档路径转为 Path。
    out_pt = Path(args.out_pt)
    # 将输出 .pt 路径转为 Path。

    if args.dataset == "amazon_movies":
        # Amazon Movies 分支。
        d = parse_amazon_movies(archive, args.max_edges)
    elif args.dataset == "raw_cn":
        # 中国专利分支。
        d = parse_raw_patent(archive, "CN", args.max_edges)
    elif args.dataset == "raw_us":
        # 美国专利分支。
        d = parse_raw_patent(archive, "US", args.max_edges)
    elif args.dataset == "dblp":
        # DBLP 分支：本任务关注的入口。
        d = parse_dblp(archive, args.max_edges)
        # 调用 DBLP 解析函数，返回 numpy 字典。
    elif args.dataset == "aminer_paper":
        # AMiner Paper 分支。
        d = parse_aminer_paper(archive, args.max_edges, min_year=args.aminer_min_year, max_year=args.aminer_max_year)
    else:
        # 理论上 choices 已限制，不应到达。
        raise ValueError(args.dataset)

    obj = to_unified(d, out_pt)
    # 将解析结果转成统一 torch .pt 并保存。
    print(f"saved unified dataset to {out_pt}")
    # 打印保存路径。
    print(f"nodes={obj['num_nodes']} edges={obj['edges']['src'].shape[0]} rels={obj['num_rels']}")
    # 打印节点数、边数、关系数。

    if args.export_baselines:
        # 可选导出 baseline 格式。
        if not args.baseline_out_dir:
            # 开启导出时必须提供导出目录。
            raise ValueError("--baseline-out-dir is required when --export-baselines is set")
        bdir = Path(args.baseline_out_dir)
        # baseline 输出目录。
        export_tgn_format(obj, bdir / "tgn", d["name"])
        # 导出 TGN 风格文件。
        export_tgb2_like_csv(obj, bdir / "tgb2", d["name"])
        # 导出 TGB2-like CSV。
        print(f"baseline adapters exported to {bdir}")
        # 打印导出完成信息。

if __name__ == "__main__":
    # 脚本直接运行时进入 main。
    main()
    # 触发上述参数解析和 DBLP 分支。
```

## `build_paper_task_views.py` 核心视图构造

### `build_edge_view`：从统一图切出边任务视图

原始位置：`scripts/datasets/build_paper_task_views.py:117-243`

```python
def build_edge_view(obj: Dict, out_name: str, target_old: np.ndarray, train_target_old: np.ndarray, val_target_old: np.ndarray, test_target_old: np.ndarray, context_keep_mask: np.ndarray = None, extra_info: Dict = None) -> Dict:
    # 通用边任务视图构造函数；DBLP transductive/inductive/node view 都复用它。
    src_all = obj["edges"]["src"].numpy().astype(np.int64)
    # 从统一图读取全部源节点。
    dst_all = obj["edges"]["dst"].numpy().astype(np.int64)
    # 从统一图读取全部目标节点。
    ts_all = obj["edges"]["ts"].numpy().astype(np.float32)
    # 从统一图读取全部时间戳。
    rel_all = obj["edges"]["rel"].numpy().astype(np.int64)
    # 从统一图读取全部关系 id。
    raw_num_rels = int(obj.get("num_rels", int(rel_all.max()) + 1 if rel_all.size > 0 else 1))
    # 保留原始关系总数，不因为当前视图只含部分关系而重编号。

    all_old_sorted = sort_old_indices(np.arange(ts_all.shape[0], dtype=np.int64), ts_all)
    # 得到全图边的原始索引，并按时间排序。
    keep_mask = np.ones(ts_all.shape[0], dtype=bool) if context_keep_mask is None else np.asarray(context_keep_mask, dtype=bool)
    # 上下文边保留掩码；transductive 默认全保留，inductive 会过滤 unseen 节点相关边。
    keep_mask = keep_mask.reshape(-1)
    # 保证掩码是一维。
    if keep_mask.shape[0] != ts_all.shape[0]:
        # 掩码长度必须和全图边数一致。
        raise ValueError("context_keep_mask length mismatch")

    train_cut = float(ts_all[train_target_old[-1]]) if train_target_old.size > 0 else float(ts_all[all_old_sorted[0]])
    # 训练窗口截止时间来自目标训练边的最后一条边。
    val_cut = float(ts_all[val_target_old[-1]]) if val_target_old.size > 0 else train_cut
    # 验证窗口截止时间来自目标验证边的最后一条边。

    train_old = all_old_sorted[keep_mask[all_old_sorted] & (ts_all[all_old_sorted] <= train_cut)]
    # 训练窗口包含 cutoff 之前的所有保留上下文边。
    val_old = all_old_sorted[keep_mask[all_old_sorted] & (ts_all[all_old_sorted] > train_cut) & (ts_all[all_old_sorted] <= val_cut)]
    # 验证窗口包含 train_cut 到 val_cut 之间的保留上下文边。
    assigned = np.zeros(ts_all.shape[0], dtype=bool)
    # 标记已分到 train/val 的边。
    assigned[train_old] = True
    # 训练边标记为已分配。
    assigned[val_old] = True
    # 验证边标记为已分配。
    test_old = all_old_sorted[~assigned[all_old_sorted]]
    # 其余时间排序边进入测试窗口；注意这里未直接用 keep_mask，因此显式测试目标边可被保留。

    explicit_val_old = sort_old_indices(val_target_old, ts_all)
    # 目标验证边显式排序。
    explicit_test_old = sort_old_indices(test_target_old, ts_all)
    # 目标测试边显式排序。
    if explicit_val_old.size > 0:
        # 确保目标验证边一定在验证窗口。
        train_old = remove_indices(train_old, explicit_val_old, ts_all)
        # 从训练窗口移除目标验证边。
        test_old = remove_indices(test_old, explicit_val_old, ts_all)
        # 从测试窗口移除目标验证边。
        val_old = sort_old_indices(np.concatenate([val_old, explicit_val_old]), ts_all)
        # 加入验证窗口并重新按时间排序。
    if explicit_test_old.size > 0:
        # 确保目标测试边一定在测试窗口。
        train_old = remove_indices(train_old, explicit_test_old, ts_all)
        # 从训练窗口移除目标测试边。
        val_old = remove_indices(val_old, explicit_test_old, ts_all)
        # 从验证窗口移除目标测试边。
        test_old = sort_old_indices(np.concatenate([test_old, explicit_test_old]), ts_all)
        # 加入测试窗口并重新按时间排序。

    if val_old.size == 0 and np.asarray(val_target_old).size > 0:
        # 如果过滤上下文后验证窗口为空，用显式目标验证边兜底。
        val_old = sort_old_indices(val_target_old, ts_all)
        # 验证窗口改为目标验证边。
        train_old = remove_indices(train_old, val_old, ts_all)
        # 保证训练窗口不重复包含这些边。
        test_old = remove_indices(test_old, val_old, ts_all)
        # 保证测试窗口不重复包含这些边。
    if test_old.size == 0 and np.asarray(test_target_old).size > 0:
        # 如果测试窗口为空，用显式目标测试边兜底。
        test_old = sort_old_indices(test_target_old, ts_all)
        # 测试窗口改为目标测试边。
        train_old = remove_indices(train_old, test_old, ts_all)
        # 保证训练窗口不重复包含这些边。
        val_old = remove_indices(val_old, test_old, ts_all)
        # 保证验证窗口不重复包含这些边。

    selected_old = np.concatenate([train_old, val_old, test_old], axis=0).astype(np.int64)
    # 当前视图实际保留的旧边索引。
    selected_old = sort_old_indices(selected_old, ts_all)
    # 视图边再次按时间排序。
    old_to_new = np.full((ts_all.shape[0],), -1, dtype=np.int64)
    # 构造旧边索引到视图内新边索引的映射。
    old_to_new[selected_old] = np.arange(selected_old.shape[0], dtype=np.int64)
    # 被选中的旧边获得连续新索引。

    src = src_all[selected_old]
    # 切出视图源节点。
    dst = dst_all[selected_old]
    # 切出视图目标节点。
    ts = ts_all[selected_old]
    # 切出视图时间戳。
    rel = rel_all[selected_old]
    # 切出视图关系 id。
    rel_vals = sorted(np.unique(rel).tolist()) if rel.size > 0 else [0]
    # 记录视图中实际出现的关系集合。
    num_rels = max(1, raw_num_rels)
    # 关系数沿用统一图全局关系数。

    supervised_old = sort_old_indices(np.concatenate([train_target_old, val_target_old, test_target_old], axis=0), ts_all)
    # 监督目标边集合，仅包含目标关系的 train/val/test 边。
    supervised_mask = np.isin(selected_old, supervised_old)
    # 标记视图内哪些边属于监督目标。
    edge_label = supervised_mask.astype(np.float32)
    # 监督目标边 label=1，其它上下文边 label=0。

    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6) if ts.size > 0 else np.array([], dtype=np.float32)
    # 视图内时间归一化。
    rel_norm = rel / max(1, num_rels - 1) if rel.size > 0 else np.array([], dtype=np.float32)
    # 视图内关系 id 归一化，但关系 id 不重映射。
    edge_feat = np.stack([ts_norm, rel_norm], axis=1).astype(np.float32) if ts.size > 0 else np.zeros((0, 2), dtype=np.float32)
    # 生成视图边特征。

    train_new = map_old_to_new(train_old, old_to_new)
    # 将训练旧索引映射为视图新索引。
    val_new = map_old_to_new(val_old, old_to_new)
    # 将验证旧索引映射为视图新索引。
    test_new = map_old_to_new(test_old, old_to_new)
    # 将测试旧索引映射为视图新索引。

    meta = dict(obj.get("meta", {}))
    # 复制统一图 meta。
    extra = dict(meta.get("extra", {}))
    # 复制 meta.extra。
    extra.pop("task_splits", None)
    # 移除旧任务划分，避免和当前视图划分冲突。
    extra["task_view_name"] = out_name
    # 写入当前任务视图名。
    extra["source_rel_ids"] = rel_vals
    # 写入视图中出现的关系 id。
    extra["context_protocol"] = {
        # 记录上下文协议和监督边统计。
        "enabled": True,
        # 标明启用了上下文协议。
        "supervision_mode": "edge_label_mask",
        # 监督由 edge_label 掩码表达。
        "preserve_global_rel_ids": True,
        # 保留全局关系 id。
        "selected_edges": int(selected_old.size),
        # 当前视图边数。
        "supervised_edges": int(edge_label.sum()),
        # 当前视图中监督边数量。
        "train_window_edges": int(train_new.size),
        # 训练窗口边数。
        "val_window_edges": int(val_new.size),
        # 验证窗口边数。
        "test_window_edges": int(test_new.size),
        # 测试窗口边数。
        "target_original_rel_ids": sorted(np.unique(rel_all[supervised_old]).tolist()) if supervised_old.size > 0 else [],
        # 监督目标对应的原始关系 id。
        "train_supervised_edges": int(np.isin(train_old, supervised_old).sum()),
        # 训练窗口中监督边数量。
        "val_supervised_edges": int(np.isin(val_old, supervised_old).sum()),
        # 验证窗口中监督边数量。
        "test_supervised_edges": int(np.isin(test_old, supervised_old).sum()),
        # 测试窗口中监督边数量。
    }
    if isinstance(extra_info, dict):
        # 允许调用方补充 transductive/inductive/node task 专属字段。
        extra["context_protocol"].update(extra_info)
        # 合并补充信息。
    meta["dataset_name"] = out_name
    # 当前 .pt 的数据集名改成任务视图名。
    meta["extra"] = extra
    # 写回扩展元信息。

    return {
        # 返回和统一图相同结构的任务视图对象。
        "num_nodes": int(obj["num_nodes"]),
        # 节点数沿用统一图，不按当前视图裁剪节点。
        "num_rels": int(num_rels),
        # 关系数量沿用全局关系数量。
        "num_node_types": int(obj.get("num_node_types", 1)),
        # 节点类型数量沿用统一图。
        "node_feat": obj["node_feat"],
        # 节点特征直接复用统一图。
        "node_type": obj["node_type"],
        # 节点类型直接复用统一图。
        "edges": {
            # 当前任务视图的边字段。
            "src": torch.from_numpy(src),
            # 视图内源节点。
            "dst": torch.from_numpy(dst),
            # 视图内目标节点。
            "ts": torch.from_numpy(ts),
            # 视图内时间戳。
            "rel": torch.from_numpy(rel),
            # 视图内关系 id，仍为全局关系 id。
            "edge_feat": torch.from_numpy(edge_feat),
            # 视图内边特征。
            "edge_label": torch.from_numpy(edge_label),
            # 监督掩码：目标关系监督边为 1，上下文边为 0。
        },
        "split": {
            # 当前视图内的新边索引划分。
            "train": torch.from_numpy(train_new),
            # 训练窗口的新索引。
            "val": torch.from_numpy(val_new),
            # 验证窗口的新索引。
            "test": torch.from_numpy(test_new),
            # 测试窗口的新索引。
        },
        "node_multiclass": obj.get("node_multiclass"),
        # 节点多分类任务字段沿用统一图。
        "node_multilabel": obj.get("node_multilabel"),
        # 节点多标签任务字段沿用统一图。
        "meta": meta,
        # 写入当前任务视图 meta。
    }
```

### `build` 和 `main`：paper task view 生成入口

原始位置：`scripts/datasets/build_paper_task_views.py:272-352`

```python
def build(input_pt: Path, out_dir: Path, prefix: str, target_rel: int = None):
    # 从统一 .pt 生成三个 paper task view 的主函数。
    obj = torch.load(input_pt, map_location="cpu")
    # 读取 prepare_dataset.py 生成的 dblp_unified.pt。
    rel = obj["edges"]["rel"].numpy().astype(np.int64)
    # 取全部关系 id。
    ts = obj["edges"]["ts"].numpy().astype(np.float32)
    # 取全部时间戳。
    src = obj["edges"]["src"].numpy().astype(np.int64)
    # 取全部源节点。
    dst = obj["edges"]["dst"].numpy().astype(np.int64)
    # 取全部目标节点。

    target_rel_id = choose_target_relation(rel, target_rel=target_rel)
    # 选择监督目标关系；target_rel=None 时选边数最多的关系。
    target_old = sort_old_indices(np.where(rel == int(target_rel_id))[0].astype(np.int64), ts)
    # 找到所有目标关系边的旧索引，并按时间排序。
    if target_old.size == 0:
        # 如果目标关系没有边，无法构造任务。
        raise RuntimeError(f"no edges found for target relation id={target_rel_id}")

    trans_tr_old, trans_va_old, trans_te_old = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)
    # 对目标关系边做 transductive 监督划分。
    trans_view = build_edge_view(
        # 构造 transductive 边任务视图。
        obj=obj,
        # 传入统一图对象。
        out_name=f"{prefix}_edge_transductive",
        # DBLP prefix=dblp 时输出视图名 dblp_edge_transductive。
        target_old=target_old,
        # 所有目标关系边。
        train_target_old=trans_tr_old,
        # 目标关系训练边。
        val_target_old=trans_va_old,
        # 目标关系验证边。
        test_target_old=trans_te_old,
        # 目标关系测试边。
        extra_info={
            # 追加 transductive 协议信息。
            "target_original_rel_id": int(target_rel_id),
            # 记录目标关系 id。
            "protocol": "full_hetero_context_transductive",
            # 协议名：完整异构上下文 transductive。
        },
    )

    ind_tr_old, ind_va_old, ind_te_old, unseen_nodes = inductive_partition_target(
        # 构造 inductive 划分，并选择未见源节点。
        target_old=target_old,
        # 只在目标关系边上选择监督划分。
        src=src,
        # 源节点数组用于按 source node holdout。
        ts=ts,
        # 时间戳用于排序和兜底时间划分。
        rng=np.random.RandomState(42),
        # 固定随机种子，保证视图可复现。
    )
    keep_mask = ~(np.isin(src, unseen_nodes) | np.isin(dst, unseen_nodes))
    # inductive 上下文过滤：任何接触 unseen source nodes 的边都不保留。
    ind_view = build_edge_view(
        # 构造 inductive 边任务视图。
        obj=obj,
        # 传入统一图对象。
        out_name=f"{prefix}_edge_inductive",
        # DBLP prefix=dblp 时输出视图名 dblp_edge_inductive。
        target_old=target_old,
        # 所有目标关系边。
        train_target_old=ind_tr_old,
        # inductive 训练目标边。
        val_target_old=ind_va_old,
        # inductive 验证目标边。
        test_target_old=ind_te_old,
        # inductive 测试目标边。
        context_keep_mask=keep_mask,
        # 传入上下文保留掩码，过滤 unseen 节点相关上下文。
        extra_info={
            # 追加 inductive 协议信息。
            "target_original_rel_id": int(target_rel_id),
            # 记录目标关系 id。
            "protocol": "full_hetero_context_inductive",
            # 协议名：完整异构上下文 inductive。
            "unseen_source_node_ids": unseen_nodes.tolist(),
            # 记录被 holdout 的源节点 id。
        },
    )

    node_view = build_sorted_full_view(obj, out_name=f"{prefix}_node_multilabel")
    # 构造节点多标签任务视图：全关系时间排序，视图名 dblp_node_multilabel。

    out_dir.mkdir(parents=True, exist_ok=True)
    # 确保 paper task 输出目录存在。
    p1 = out_dir / f"{prefix}_edge_transductive.pt"
    # transductive 输出路径。
    p2 = out_dir / f"{prefix}_edge_inductive.pt"
    # inductive 输出路径。
    p3 = out_dir / f"{prefix}_node_multilabel.pt"
    # node_multilabel 输出路径。
    torch.save(trans_view, p1)
    # 保存 transductive 视图。
    torch.save(ind_view, p2)
    # 保存 inductive 视图。
    torch.save(node_view, p3)
    # 保存节点多标签视图。
    return [p1, p2, p3]
    # 返回三个输出路径，供 main 打印。

def main():
    # paper task view 脚本入口。
    ap = argparse.ArgumentParser()
    # 创建参数解析器。
    ap.add_argument("--input", required=True, type=str)
    # 输入统一 .pt，例如 data/processed/dblp_unified.pt。
    ap.add_argument("--out-dir", required=True, type=str)
    # 输出目录，例如 data/processed/paper_tasks。
    ap.add_argument("--name-prefix", required=True, type=str)
    # 输出名前缀，DBLP 应传 dblp。
    ap.add_argument("--target-rel", type=int, default=-1)
    # 可选目标关系；默认 -1 表示自动选边数最多的关系。
    args = ap.parse_args()
    # 解析命令行参数。

    outs = build(
        # 调用视图生成主函数。
        input_pt=Path(args.input),
        # 输入 .pt 路径。
        out_dir=Path(args.out_dir),
        # 输出目录路径。
        prefix=args.name_prefix,
        # 输出名前缀。
        target_rel=None if args.target_rel < 0 else int(args.target_rel),
        # -1 转成 None，触发自动目标关系选择；否则使用指定关系 id。
    )
    print("generated:")
    # 打印生成提示。
    for p in outs:
        # 遍历三个输出路径。
        print(p)
        # 打印每个生成的 .pt 文件路径。

if __name__ == "__main__":
    # 脚本直接运行时进入 main。
    main()
    # 触发 paper task view 生成流程。
```

## DBLP 预处理命令形态

以下是链路形态说明，不在本文档生成过程中执行：

```bash
python scripts/datasets/prepare_dataset.py \
  --dataset dblp \
  --archive /path/to/DBLP.zip \
  --out-pt data/processed/dblp_unified.pt \
  --max-edges 200000

python scripts/datasets/build_paper_task_views.py \
  --input data/processed/dblp_unified.pt \
  --out-dir data/processed/paper_tasks \
  --name-prefix dblp
```

## 检查点

- DBLP 原始解析函数：`prepare_dataset.py:868-977 parse_dblp`。
- DBLP main 分支：`prepare_dataset.py:1684-1685`。
- 统一保存入口：`prepare_dataset.py:1533-1597 to_unified`。
- paper task 生成入口：`build_paper_task_views.py:332-352 main`。
- paper task 主构造函数：`build_paper_task_views.py:272-329 build`。
- 边任务视图核心：`build_paper_task_views.py:117-243 build_edge_view`。
