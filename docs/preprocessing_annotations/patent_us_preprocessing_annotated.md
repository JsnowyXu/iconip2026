# Patent-US 预处理源码注释

## 原始源码位置

- 正式入口和核心实现：`scripts/experiments/run_patent_preprocessing.py`
- 执行记录：`patent_us_process.md`
- 后续/平台通用 task view 生成入口：`scripts/datasets/build_task_views.py`
- 平台 preprocess job 组装入口：`platform/backend/core/services/job_builder.py`
- 平台数据集配置：`platform/backend/core/services/registry.py`

`patent_us_process.md` 记录的实际命令是：

```bash
/home/xuxj/.conda/envs/pytorch/bin/python scripts/experiments/run_patent_preprocessing.py --project-root /home/xuxj/2026/dyn_relation_memory --dataset-root /home/xuxj/2026/datasets --python-bin /home/xuxj/.conda/envs/pytorch/bin/python
```

因此 Patent-US 的正式预处理链路不是平台按钮触发的通用 `build_task_views.py`，而是 `run_patent_preprocessing.py` 内部一次性完成：读取原始 CSV -> 构建统一异构动态图 -> 生成三个 task view -> 导出 TGN/TGB2 适配格式 -> 写过程日志和 `patent.md` 摘要。

## 执行链路

1. `main()` 解析 `--project-root`、`--dataset-root`、`--feature-dim`、`--datasets` 等参数。
2. 遍历 `DATASET_SPECS`，当未传 `--datasets` 时会处理 `patent_us`、`patent_cn`、`patent_in`；若只跑 US，可传 `--datasets patent_us`。
3. 对 `patent_us` 调用 `build_unified_dataset(spec, dataset_root, feature_dim)`。
4. `build_unified_dataset()` 读取 `/home/xuxj/2026/datasets/patent_us/all-us.csv`、`clsIII-reInd.csv`、`IPCSubClass_US.csv`，构建统一图对象 `unified`。
5. `main()` 保存 `data/processed/patent_us_unified.pt`。
6. `main()` 调用同文件内的 `build_task_views(unified)`，生成：
   - `patent_us_edge_org_codeIII_transductive.pt`
   - `patent_us_edge_org_codeIII_inductive.pt`
   - `patent_us_node_patent_multilabel.pt`
7. 每个 task view 通过 `export_baseline_bundle()` 继续导出 TGN 与 TGB2/STHN 风格适配文件。
8. `analyze_one()` 生成统计图，`write_process_md()` 写 `patent_us_process.md`，`write_summary_md()` 重写专利总说明 `patent.md`。

## 配置片段：Patent-US 数据集定义

原始位置：`scripts/experiments/run_patent_preprocessing.py:18`

```python
DATASET_SPECS = [  # 数据集配置表，main() 会逐个遍历这里的 spec。
    {  # 第一个配置就是 Patent-US。
        "name": "patent_us",  # 输出数据集名；后续文件名前缀也是 patent_us。
        "short": "us",  # 简写标识，当前主流程中主要作为描述性字段。
        "csv": "all-us.csv",  # Patent-US 主表，位于 dataset_root / patent_us / all-us.csv。
        "ipc": "IPCSubClass_US.csv",  # US IPC subclass 描述表，用来给 codeIII 节点补文本描述。
        "cls": "clsIII-reInd.csv",  # codeIIIId 与重编号 codeIIIIdreId、codeIII 名称的映射表。
        "org_name_cols": ["orgname"],  # US 数据里优先从 orgname 读取机构显示名。
        "title_cols": ["title"],  # 专利标题候选列。
        "summary_cols": ["summary"],  # 专利摘要候选列。
        "date_cols": ["publicationDate", "date_time", "date_parsed"],  # 发布时间候选列，按顺序取第一个存在列。
    },
```

关键点：

- `name=patent_us` 决定输入目录为 `/home/xuxj/2026/datasets/patent_us`。
- `csv/ipc/cls` 决定三个原始输入文件。
- `org_name_cols/title_cols/summary_cols/date_cols` 不要求固定大小写，实际用 `choose_first()` 在 CSV 列名中做大小写无关匹配。

## 辅助函数：字段选择、多值拆分、时间和索引

原始位置：`scripts/experiments/run_patent_preprocessing.py:60`

```python
def choose_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:  # 从候选列名里挑第一个真实存在的列。
    lowered = {c.lower(): c for c in columns}  # 建立小写列名到原始列名的映射，保留原 CSV 中的实际大小写。
    for cand in candidates:  # 按 spec 中声明的优先级遍历候选列。
        if cand.lower() in lowered:  # 候选列大小写无关命中。
            return lowered[cand.lower()]  # 返回 CSV 里的真实列名，后续 df[row_col] 可以直接使用。
    return None  # 如果一个候选都不存在，调用方需要处理缺失列。

def split_multi_value(value) -> List[str]:  # 统一拆分 orgID、codeIIIId、orgname 等多值字段。
    if value is None:  # 空值直接变成空列表。
        return []
    if isinstance(value, float) and math.isnan(value):  # pandas 读出的 NaN 也视作空。
        return []
    if isinstance(value, (list, tuple, set)):  # 如果原值已经是容器，递归拆每个元素。
        out = []
        for item in value:
            out.extend(split_multi_value(item))  # 展平嵌套列表。
        return out

    text = str(value).strip()  # 其余类型先转字符串并去掉首尾空白。
    if not text or text.lower() in {"nan", "none", "null"}:  # 常见空字符串标记。
        return []

    if text.startswith("[") and text.endswith("]"):  # 兼容 "['a', 'b']" 这类字符串列表。
        try:
            parsed = ast.literal_eval(text)  # 安全解析 Python 字面量。
            return split_multi_value(parsed)  # 解析后递归走统一逻辑。
        except Exception:
            pass  # 解析失败则继续按分号/竖线拆分。

    text = text.replace("|", ";")  # 同时支持竖线和分号分隔。
    parts = [p.strip().strip("'").strip('"') for p in text.split(";")]  # 拆分并清理引号。
    return [p for p in parts if p and p.lower() not in {"nan", "none", "null"}]  # 过滤空值。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:145`

```python
def sort_old_indices(indices: Iterable[int], ts: np.ndarray) -> np.ndarray:  # 对原始边索引按时间排序。
    idx = np.asarray(list(indices), dtype=np.int64).reshape(-1)  # 统一转成一维 int64 数组。
    idx = idx[(idx >= 0) & (idx < ts.shape[0])]  # 丢弃越界索引。
    if idx.size == 0:  # 空输入直接返回空数组。
        return np.array([], dtype=np.int64)
    idx = np.unique(idx)  # 去重，避免同一条边重复进入 task view。
    order = np.lexsort((idx, ts[idx]))  # 主键时间戳，次键原索引，保证同时间边顺序稳定。
    return idx[order].astype(np.int64)  # 返回排序后的旧索引。

def temporal_partition_sorted(sorted_old: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # 将已按时间排序的边切成 train/val/test。
    sorted_old = np.asarray(sorted_old, dtype=np.int64).reshape(-1)  # 标准化输入。
    if sorted_old.size == 0:  # 没有可监督边时返回三个空数组。
        z = np.array([], dtype=np.int64)
        return z, z, z
    n = sorted_old.size  # 可监督边数量。
    n_train = max(1, int(n * train_ratio))  # 默认 70% 训练，至少 1 条。
    n_val = max(1, int(n * val_ratio))  # 默认 15% 验证，至少 1 条。
    if n_train + n_val >= n:  # 小数据集保护，避免 test 被挤空。
        n_train = max(1, n - 2)
        n_val = max(1, n - n_train - 1)
    train_old = sorted_old[:n_train]  # 前段时间边作为训练监督边。
    val_old = sorted_old[n_train : n_train + n_val]  # 中段时间边作为验证监督边。
    test_old = sorted_old[n_train + n_val :]  # 后段时间边作为测试监督边。
    if test_old.size == 0 and val_old.size > 0:  # 如果 test 仍为空，从 val 挪最后一条过去。
        test_old = val_old[-1:]
        val_old = val_old[:-1]
    if val_old.size == 0 and train_old.size > 1:  # 如果 val 为空，从 train 挪最后一条过去。
        val_old = train_old[-1:]
        train_old = train_old[:-1]
    return train_old.astype(np.int64), val_old.astype(np.int64), test_old.astype(np.int64)  # 返回旧边索引。
```

## 核心函数：构建统一 Patent-US 异构动态图

原始位置：`scripts/experiments/run_patent_preprocessing.py:521`

```python
def build_unified_dataset(spec: Dict, dataset_root: Path, feature_dim: int) -> Tuple[Dict, Dict]:  # 从一个专利 spec 构建统一图和统计计数。
    ds_dir = dataset_root / spec["name"]  # Patent-US 时为 /home/xuxj/2026/datasets/patent_us。
    print(f"[prepare] start {spec['name']}", flush=True)  # 打印进度，不改变数据。
    df = pd.read_csv(ds_dir / spec["csv"])  # 读取主表 all-us.csv。
    cls_df = pd.read_csv(ds_dir / spec["cls"])  # 读取 clsIII-reInd.csv，提供 codeIII 重编号和名称。
    ipc_df = load_ipc_table(ds_dir / spec["ipc"])  # 读取 IPCSubClass_US.csv，失败时有宽松 fallback。

    title_col = choose_first(df.columns, spec["title_cols"])  # 找到标题列。
    summary_col = choose_first(df.columns, spec["summary_cols"])  # 找到摘要列。
    date_col = choose_first(df.columns, spec["date_cols"])  # 找到发布日期列。
    org_name_col = choose_first(df.columns, spec["org_name_cols"])  # 找到机构名列。
```

这段完成输入表读取和列名适配。Patent-US 的三个输入表分别承担：

- `all-us.csv`：专利行、专利 ID、机构 ID、codeIIIId、标题、摘要、日期。
- `clsIII-reInd.csv`：把原始 `codeIIIId` 映射到重编号 `codeIIIIdreId` 和 `codeIII` 名。
- `IPCSubClass_US.csv`：给 `codeIII` 补 subclass 描述文本。

原始位置：`scripts/experiments/run_patent_preprocessing.py:547`

```python
    code_meta_by_reid: Dict[str, Dict] = {}  # 以重编号 codeIIIIdreId 为 key 保存 code 节点元信息。
    for _, row in cls_df.iterrows():  # 遍历 clsIII-reInd.csv 每一行。
        codeiiiid = str(row["codeIIIId"]).strip()  # 原始 codeIIIId，后续用于多标签类别 ID。
        codeiii = str(row["codeIII"]).strip()  # IPC subclass 名称。
        reid = str(row["codeIIIIdreId"]).strip()  # 重编号 ID，主表 codeIIIId 用它来过滤和连边。
        desc = ipc_desc_by_code.get(codeiii, "")  # 从 IPC 描述表中找文字描述。
        code_meta_by_reid[reid] = {  # 建立重编号 -> 元信息映射。
            "codeIIIId": codeiiiid,  # 保留原始类别 ID。
            "codeIIIIdreId": reid,  # 保留重编号 ID。
            "codeIII": codeiii,  # 保留 IPC subclass 名。
            "description": desc,  # 保留 IPC 描述文本。
        }
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:560`

```python
    node_key_to_id: Dict[str, int] = {}  # 全局节点去重表，key 形如 patent:xxx、org:xxx、code:xxx。
    node_type: List[int] = []  # 每个节点的类型 ID；0=patent，1=org，2=codeIII。
    node_text: List[str] = []  # 用于 TF-IDF 的节点文本。
    node_name: List[str] = []  # 节点显示名，文本为空时可回退。
    patent_nodes: List[int] = []  # 专利节点 ID 列表。
    patent_raw_ids: List[str] = []  # 原始 patentId 列表。
    patent_titles: List[str] = []  # 专利标题列表。
    patent_summaries: List[str] = []  # 专利摘要列表。
    patent_dates: List[str] = []  # 专利日期文本列表。
    patent_ts: List[float] = []  # 专利时间戳列表。
    patent_code_ids: Dict[int, List[str]] = {}  # 每个专利节点对应的重编号 codeIIIIdreId。
    patent_label_ids: Dict[int, List[str]] = {}  # 每个专利节点对应的原始 codeIIIId，多标签分类用。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:582`

```python
    def get_node(key: str, ntype: int, text: str, display_name: str, registry: str, raw_id: str, extra_text: str = "") -> int:  # 注册或复用节点。
        if key in node_key_to_id:  # 如果同一个专利/机构/code 已出现过。
            return node_key_to_id[key]  # 直接返回已有节点 ID。
        nid = len(node_type)  # 新节点 ID 等于当前节点数。
        node_key_to_id[key] = nid  # 写入去重表。
        node_type.append(ntype)  # 记录节点类型。
        full_text = " ".join([t for t in [text, extra_text] if t]).strip()  # 标题+摘要、机构名、code 名+描述等合并文本。
        node_text.append(full_text)  # 保存用于文本特征。
        node_name.append(display_name)  # 保存显示名。
        if registry == "patent":  # 专利节点分支。
            patent_nodes.append(nid)  # 加入专利节点列表。
            patent_node_to_idx[nid] = len(patent_nodes) - 1  # 建立节点 ID 到专利数组下标的映射。
            patent_raw_ids.append(raw_id)  # 保存 patentId。
            patent_titles.append(text)  # 保存标题。
            patent_summaries.append(extra_text)  # 保存摘要。
            patent_dates.append("")  # 先占位，后面按行写入真实日期。
            patent_ts.append(0.0)  # 先占位，后面按行写入真实时间戳。
            patent_code_ids[nid] = []  # 初始化该专利的 code 重编号集合。
            patent_label_ids[nid] = []  # 初始化该专利的多标签类别集合。
        elif registry == "org":  # 机构节点分支。
            org_nodes.append(nid)  # 加入机构节点列表。
            org_raw_ids.append(raw_id)  # 保存原始 orgID。
            org_names.append(display_name)  # 保存机构名。
        elif registry == "code":  # codeIII 节点分支。
            code_nodes.append(nid)  # 加入 codeIII 节点列表。
            code_raw_ids.append(raw_id)  # 保存原始 codeIIIId。
            code_reindexed_ids.append(key.split(":", 1)[1])  # 保存 codeIIIIdreId。
            code_names.append(display_name)  # 保存 codeIII 名。
            code_descs.append(extra_text)  # 保存 IPC 描述。
        return nid  # 返回新建节点 ID。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:613`

```python
    rel_map = {"patent_org": 0, "patent_codeIII": 1, "org_codeIII": 2, "org_coorg": 3}  # 四类边的全局关系 ID。
    src_list: List[int] = []  # 边源节点数组，后续转 numpy/torch。
    dst_list: List[int] = []  # 边目标节点数组。
    ts_list: List[float] = []  # 边时间戳数组。
    rel_list: List[int] = []  # 边关系类型数组。
```

四类关系含义：

- `patent_org`：专利 -> 机构。
- `patent_codeIII`：专利 -> IPC subclass/codeIII。
- `org_codeIII`：机构 -> codeIII，是后续边预测的目标关系。
- `org_coorg`：机构 <-> 共同申请机构，按同一专利中的机构两两双向生成。

原始位置：`scripts/experiments/run_patent_preprocessing.py:622`

```python
    for _, row in df.iterrows():  # 逐行遍历 Patent-US 主表，每行通常对应一个专利记录。
        patent_raw = str(row["patentId"]).strip()  # 读取原始专利 ID。
        patent_title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""  # 读取标题，缺失则空。
        patent_summary = str(row[summary_col]).strip() if summary_col and pd.notna(row[summary_col]) else ""  # 读取摘要，缺失则空。
        patent_ts_val, patent_date_text = parse_publication_ts(row[date_col] if date_col else "")  # 日期转 Unix timestamp 和 yyyy-mm-dd 文本。
        org_ids = split_multi_value(row["orgID"])  # 拆分多机构 ID。
        code_ids = split_multi_value(row["codeIIIId"])  # 拆分多个 codeIII 重编号 ID。
        org_names_row = split_multi_value(row[org_name_col]) if org_name_col else []  # 拆分机构名，尽量与 orgID 对齐。
        class_codes_row = split_multi_value(row["classCode"]) if "classCode" in df.columns else []  # 统计 classCode 是否非空。

        org_ids = [x for x in org_ids if x]  # 清理空机构 ID。
        code_ids = [x for x in code_ids if x in code_meta_by_reid]  # 只保留能在 cls 映射表中找到的 codeIIIIdreId。
        if not org_ids or not code_ids:  # 机构或 code 任一缺失都无法生成目标图结构。
            row_stats["rows_skipped_missing_org_or_code"] += 1  # 记录跳过行数。
            continue  # 跳过该专利。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:638`

```python
        patent_node = get_node(  # 注册专利节点。
            key=f"patent:{patent_raw}",  # 用 patentId 去重。
            ntype=0,  # 节点类型 0 表示 patent。
            text=patent_title,  # 主文本为标题。
            display_name=f"patent:{patent_raw}",  # 显示名为 patent:ID。
            registry="patent",  # 进入 get_node 的专利元数据分支。
            raw_id=patent_raw,  # 保存原始 patentId。
            extra_text=patent_summary,  # 额外文本为摘要。
        )
        pat_idx = patent_node_to_idx[patent_node]  # 找到该专利在专利属性数组中的位置。
        patent_dates[pat_idx] = patent_date_text  # 写入格式化日期。
        patent_ts[pat_idx] = patent_ts_val  # 写入数值时间戳。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:651`

```python
        org_node_ids = []  # 当前专利关联的机构节点 ID。
        for i, org_raw in enumerate(dict.fromkeys(org_ids)):  # dict.fromkeys 保留顺序去重。
            org_name = normalize_org_name(org_names_row[i]) if i < len(org_names_row) else f"org:{org_raw}"  # 优先取对齐的机构名。
            org_node = get_node(  # 注册或复用机构节点。
                key=f"org:{org_raw}",  # 用 orgID 去重。
                ntype=1,  # 节点类型 1 表示 org。
                text=org_name,  # 机构文本就是机构名。
                display_name=org_name,  # 显示名也是机构名。
                registry="org",  # 进入 get_node 的机构元数据分支。
                raw_id=str(org_raw),  # 保存原始 orgID。
            )
            org_node_ids.append(org_node)  # 记录当前专利关联机构。

        code_node_ids = []  # 当前专利关联的 codeIII 节点 ID。
        for code_reid in dict.fromkeys(code_ids):  # 对 codeIIIIdreId 保序去重。
            meta = code_meta_by_reid[code_reid]  # 取 code 元信息。
            code_name = meta["codeIII"]  # codeIII 名称。
            code_desc = meta["description"]  # IPC 描述。
            code_node = get_node(  # 注册或复用 codeIII 节点。
                key=f"code:{code_reid}",  # 用重编号 ID 去重。
                ntype=2,  # 节点类型 2 表示 codeIII。
                text=code_name,  # 主文本为 codeIII 名。
                display_name=code_name,  # 显示名为 codeIII 名。
                registry="code",  # 进入 get_node 的 code 元数据分支。
                raw_id=str(meta["codeIIIId"]),  # 保存原始 codeIIIId。
                extra_text=code_desc,  # 额外文本为 IPC 描述。
            )
            code_node_ids.append(code_node)  # 记录当前专利关联 code 节点。
            patent_code_ids[patent_node].append(code_reid)  # 保存重编号 code，用于追踪。
            patent_label_ids[patent_node].append(meta["codeIIIId"])  # 保存原始类别 ID，用于多标签矩阵。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:685`

```python
        for org_node in org_node_ids:  # 为每个机构生成专利-机构边。
            src_list.append(patent_node)  # 源节点是专利。
            dst_list.append(org_node)  # 目标节点是机构。
            ts_list.append(patent_ts_val)  # 边时间使用专利发布时间。
            rel_list.append(rel_map["patent_org"])  # 关系类型 patent_org。
        row_stats["patent_org_edges"] += len(org_node_ids)  # 累计该类边数量。

        for code_node in code_node_ids:  # 为每个 code 生成专利-code 边。
            src_list.append(patent_node)  # 源节点是专利。
            dst_list.append(code_node)  # 目标节点是 codeIII。
            ts_list.append(patent_ts_val)  # 时间同样用专利发布时间。
            rel_list.append(rel_map["patent_codeIII"])  # 关系类型 patent_codeIII。
        row_stats["patent_codeIII_edges"] += len(code_node_ids)  # 累计专利-code 边数量。

        for org_node in org_node_ids:  # 遍历机构。
            for code_node in code_node_ids:  # 遍历 code。
                src_list.append(org_node)  # 源节点是机构。
                dst_list.append(code_node)  # 目标节点是 codeIII。
                ts_list.append(patent_ts_val)  # 用该专利时间作为机构-code 交互时间。
                rel_list.append(rel_map["org_codeIII"])  # 关系类型 org_codeIII，也是后续边预测目标。
                org_code_counter[(org_node, code_node)] += 1  # 统计唯一机构-code 对出现情况。
        row_stats["org_codeIII_edges"] += len(org_node_ids) * len(code_node_ids)  # 笛卡尔积边数量。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:708`

```python
        for i in range(len(org_node_ids)):  # 枚举当前专利内的机构。
            for j in range(i + 1, len(org_node_ids)):  # 取不同机构对。
                a, b = org_node_ids[i], org_node_ids[j]  # 两个共同申请机构节点。
                src_list.extend([a, b])  # 加 a->b 和 b->a 两条边。
                dst_list.extend([b, a])  # 目标节点对应反向。
                ts_list.extend([patent_ts_val, patent_ts_val])  # 两条边同一时间。
                rel_list.extend([rel_map["org_coorg"], rel_map["org_coorg"]])  # 都是 org_coorg 关系。
                org_pair_counter[(a, b)] += 1  # 统计唯一共申请机构对。
        row_stats["org_coorg_edges"] += max(0, len(org_node_ids) * (len(org_node_ids) - 1))  # 双向边数量。
        row_stats["rows_processed"] += 1  # 当前专利成功进入图构建。
        row_stats["class_code_text_nonempty"] += int(len(class_codes_row) > 0)  # 记录 classCode 是否有内容。
```

## 特征和统一对象组装

原始位置：`scripts/experiments/run_patent_preprocessing.py:729`

```python
    order = np.lexsort((np.arange(ts.size), ts))  # 所有边按时间排序，同时间按原始生成顺序稳定排序。
    src = src[order]  # 重排源节点。
    dst = dst[order]  # 重排目标节点。
    ts = ts[order]  # 重排时间。
    rel = rel[order]  # 重排关系类型。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:750`

```python
    texts = [t if t else n for t, n in zip(node_text, node_name)]  # 节点文本为空时回退到显示名。
    text_dim = max(8, feature_dim - 8)  # 总特征维度默认 128，其中 8 维留给结构特征。
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=text_dim, lowercase=False)  # 字符 2-4 gram TF-IDF。
    text_feat = vectorizer.fit_transform(texts).toarray().astype(np.float32)  # 得到节点文本特征矩阵。
    if text_feat.shape[1] < text_dim:  # 如果实际词表小于目标维度。
        pad = np.zeros((text_feat.shape[0], text_dim - text_feat.shape[1]), dtype=np.float32)  # 补零。
        text_feat = np.concatenate([text_feat, pad], axis=1)  # 保证固定维度。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:758`

```python
    struct_feat = np.stack(  # 8 维结构/时间/类型特征。
        [
            np.log1p(in_deg) / max(np.log1p(in_deg).max(), 1.0),  # 归一化入度。
            np.log1p(out_deg) / max(np.log1p(out_deg).max(), 1.0),  # 归一化出度。
            np.log1p(deg) / max(np.log1p(deg).max(), 1.0),  # 归一化总度。
            first_ts_norm,  # 节点首次出现时间归一化。
            last_ts_norm,  # 节点最后出现时间归一化。
            (node_type_arr == 0).astype(np.float32),  # 是否 patent。
            (node_type_arr == 1).astype(np.float32),  # 是否 org。
            (node_type_arr == 2).astype(np.float32),  # 是否 codeIII。
        ],
        axis=1,
    ).astype(np.float32)
    node_feat = np.concatenate([text_feat, struct_feat], axis=1).astype(np.float32)  # 文本特征 + 结构特征合并成 node_feat。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:773`

```python
    patent_node_arr = np.asarray(patent_nodes, dtype=np.int64)  # 专利节点 ID 数组。
    label_id_order = sorted({cid for labels in patent_label_ids.values() for cid in labels}, key=lambda x: int(x))  # 收集所有原始 codeIIIId 标签并按数值排序。
    label_id_to_col = {cid: i for i, cid in enumerate(label_id_order)}  # 标签 ID 到多标签矩阵列号。
    label_matrix = np.zeros((len(patent_nodes), len(label_id_order)), dtype=np.float32)  # 行是专利，列是 codeIIIId 类别。
    for i, pnode in enumerate(patent_nodes):  # 遍历每个专利节点。
        labels = [label_id_to_col[cid] for cid in patent_label_ids.get(pnode, []) if cid in label_id_to_col]  # 找到该专利正标签列。
        if labels:
            label_matrix[i, labels] = 1.0  # 多标签正例置 1。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:788`

```python
    rel_norm = rel.astype(np.float32) / max(int(rel.max()), 1)  # 关系 ID 归一化。
    ts_norm = ((ts - ts_min) / ts_span).astype(np.float32)  # 边时间归一化。
    edge_feat = np.stack(  # 每条边 4 维特征。
        [
            ts_norm,  # 归一化时间。
            rel_norm,  # 归一化关系类型。
            node_type_arr[src].astype(np.float32) / 2.0,  # 源节点类型归一化。
            node_type_arr[dst].astype(np.float32) / 2.0,  # 目标节点类型归一化。
        ],
        axis=1,
    ).astype(np.float32)
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:836`

```python
    unified = {  # 统一图对象，后续 torch.save 成 patent_us_unified.pt。
        "num_nodes": int(len(node_type)),  # 全部 patent/org/codeIII 节点数。
        "num_rels": int(len(rel_map)),  # 关系类型数，固定为 4。
        "num_node_types": 3,  # 节点类型数，patent/org/codeIII。
        "node_feat": torch.from_numpy(node_feat.astype(np.float32)),  # 节点特征矩阵。
        "node_type": torch.from_numpy(node_type_arr.astype(np.int64)),  # 每个节点的类型 ID。
        "edges": {  # 全量时间边流。
            "src": torch.from_numpy(src.astype(np.int64)),  # 源节点。
            "dst": torch.from_numpy(dst.astype(np.int64)),  # 目标节点。
            "ts": torch.from_numpy(ts.astype(np.float32)),  # 时间戳。
            "rel": torch.from_numpy(rel.astype(np.int64)),  # 关系 ID。
            "edge_feat": torch.from_numpy(edge_feat.astype(np.float32)),  # 边特征。
            "edge_label": torch.ones(src.shape[0], dtype=torch.float32),  # 统一图中所有边先视为存在边。
        },
        "split": {  # 基于 transductive org_codeIII 时间切分推导出的全图窗口。
            "train": torch.from_numpy(base_tr),  # 训练窗口内全部关系边。
            "val": torch.from_numpy(base_va),  # 验证窗口内全部关系边。
            "test": torch.from_numpy(base_te),  # 测试窗口内全部关系边。
        },
        "node_multiclass": None,  # 没有单标签节点分类任务。
        "node_multilabel": {  # 专利节点多标签任务。
            "node_ids": torch.from_numpy(patent_node_arr.astype(np.int64)),  # 有标签的专利节点。
            "labels": torch.from_numpy(label_matrix.astype(np.float32)),  # 专利-codeIIIId 多热标签矩阵。
            "label_names": [...],  # 每个标签列对应的 codeIIIId/codeIII/description。
        },
        "meta": {...},  # relation_map、node_type_map、属性表、task_splits 等元信息。
    }
```

## Task View 生成：核心函数

原始位置：`scripts/experiments/run_patent_preprocessing.py:343`

```python
def build_task_view(  # 从统一图切出一个具体 task view。
    obj: Dict,  # 输入统一图对象。
    out_name: str,  # 输出 task view 名。
    train_old: np.ndarray,  # 训练窗口使用的旧边索引。
    val_old: np.ndarray,  # 验证窗口使用的旧边索引。
    test_old: np.ndarray,  # 测试窗口使用的旧边索引。
    supervised_old: np.ndarray,  # 哪些旧边是真正的监督目标边。
    extra_info: Dict,  # 写入 meta.extra 的任务协议信息。
) -> Dict:
    src_all = obj["edges"]["src"].numpy().astype(np.int64)  # 全图源节点数组。
    dst_all = obj["edges"]["dst"].numpy().astype(np.int64)  # 全图目标节点数组。
    ts_all = obj["edges"]["ts"].numpy().astype(np.float32)  # 全图时间戳数组。
    rel_all = obj["edges"]["rel"].numpy().astype(np.int64)  # 全图关系 ID 数组。
    edge_feat_all = obj["edges"]["edge_feat"].numpy().astype(np.float32)  # 全图边特征数组。

    selected_old = np.concatenate([train_old, val_old, test_old], axis=0).astype(np.int64)  # 合并三段窗口边。
    selected_old = sort_old_indices(selected_old, ts_all)  # 对选中旧边重新按时间排序。

    old_to_new = np.full((ts_all.shape[0],), -1, dtype=np.int64)  # 旧边索引到 task view 新边索引的映射表。
    old_to_new[selected_old] = np.arange(selected_old.shape[0], dtype=np.int64)  # 对选中边写入新编号。

    train_new = map_old_to_new(train_old, old_to_new)  # 训练旧边索引映射为新边索引。
    val_new = map_old_to_new(val_old, old_to_new)  # 验证旧边索引映射为新边索引。
    test_new = map_old_to_new(test_old, old_to_new)  # 测试旧边索引映射为新边索引。
    edge_label = np.isin(selected_old, supervised_old).astype(np.float32)  # 只有目标关系边标 1，其它上下文边标 0。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:369`

```python
    meta = dict(obj.get("meta", {}))  # 复制统一图元信息。
    extra = dict(meta.get("extra", {}))  # 复制 meta.extra。
    extra["task_view_name"] = out_name  # 写入 task view 名。
    extra["context_protocol"] = {  # 记录上下文协议，训练代码据此知道 edge_label 是监督掩码。
        "enabled": True,  # 启用上下文协议。
        "supervision_mode": "edge_label_mask",  # 用 edge_label 区分监督边和上下文边。
        "preserve_global_rel_ids": True,  # 保留统一图里的原始 relation id。
        "selected_edges": int(selected_old.size),  # task view 中总边数。
        "supervised_edges": int(edge_label.sum()),  # 监督目标边数量。
        "train_window_edges": int(train_new.size),  # 训练窗口边数。
        "val_window_edges": int(val_new.size),  # 验证窗口边数。
        "test_window_edges": int(test_new.size),  # 测试窗口边数。
    }
    extra.update(extra_info)  # 合并任务特有字段，如 target_relation_name、protocol、unseen_org_node_ids。
    meta["dataset_name"] = out_name  # 将数据集名改成 task view 名。
    meta["extra"] = extra  # 写回扩展元信息。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:386`

```python
    return {  # 返回一个完整的 task view 对象。
        "num_nodes": int(obj["num_nodes"]),  # 保留全图节点数。
        "num_rels": int(obj["num_rels"]),  # 保留全图关系数。
        "num_node_types": int(obj["num_node_types"]),  # 保留节点类型数。
        "node_feat": obj["node_feat"],  # 节点特征不裁剪。
        "node_type": obj["node_type"],  # 节点类型不裁剪。
        "edges": {
            "src": torch.from_numpy(src_all[selected_old].astype(np.int64)),  # 只取选中的边源节点。
            "dst": torch.from_numpy(dst_all[selected_old].astype(np.int64)),  # 只取选中的边目标节点。
            "ts": torch.from_numpy(ts_all[selected_old].astype(np.float32)),  # 只取选中的边时间。
            "rel": torch.from_numpy(rel_all[selected_old].astype(np.int64)),  # 保留原始关系 ID。
            "edge_feat": torch.from_numpy(edge_feat_all[selected_old].astype(np.float32)),  # 只取选中边特征。
            "edge_label": torch.from_numpy(edge_label.astype(np.float32)),  # 监督边掩码，org_codeIII 为 1，上下文边为 0。
        },
        "split": {
            "train": torch.from_numpy(train_new),  # 新编号训练边索引。
            "val": torch.from_numpy(val_new),  # 新编号验证边索引。
            "test": torch.from_numpy(test_new),  # 新编号测试边索引。
        },
        "node_multiclass": obj.get("node_multiclass"),  # 透传节点单标签字段。
        "node_multilabel": obj.get("node_multilabel"),  # 透传节点多标签字段。
        "meta": meta,  # 写入更新后的元信息。
    }
```

## Task View 生成：Patent-US 三个视图

原始位置：`scripts/experiments/run_patent_preprocessing.py:411`

```python
def build_task_views(unified: Dict) -> Dict[str, Dict]:  # 为专利统一图生成三个正式任务视图。
    rel_map = unified["meta"]["extra"]["relation_map"]  # 读取关系名到 ID 的映射。
    target_rel = int(rel_map["org_codeIII"])  # Patent-US 边预测目标关系固定为 org_codeIII。
    src = unified["edges"]["src"].numpy().astype(np.int64)  # 全图源节点。
    dst = unified["edges"]["dst"].numpy().astype(np.int64)  # 全图目标节点。
    ts = unified["edges"]["ts"].numpy().astype(np.float32)  # 全图时间戳。
    rel = unified["edges"]["rel"].numpy().astype(np.int64)  # 全图关系 ID。
    n_edges = rel.shape[0]  # 全图边数。

    all_sorted = sort_old_indices(np.arange(n_edges, dtype=np.int64), ts)  # 全部边按时间排序。
    target_old = sort_old_indices(np.where(rel == target_rel)[0], ts)  # 只取 org_codeIII 边作为监督目标边。
    tr_tgt, va_tgt, te_tgt = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # 目标边按时间切分。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:424`

```python
    train_cut = float(ts[tr_tgt[-1]]) if tr_tgt.size > 0 else float(ts[all_sorted[-1]])  # 训练监督边最后时间作为训练截止。
    val_cut = float(ts[va_tgt[-1]]) if va_tgt.size > 0 else train_cut  # 验证监督边最后时间作为验证截止。

    tr_old = all_sorted[ts[all_sorted] <= train_cut]  # 训练窗口保留所有关系类型的上下文边。
    va_old = all_sorted[(ts[all_sorted] > train_cut) & (ts[all_sorted] <= val_cut)]  # 验证窗口保留所有关系类型的上下文边。
    te_old = all_sorted[ts[all_sorted] > val_cut]  # 测试窗口保留所有关系类型的上下文边。
    trans_view = build_task_view(  # 构造 transductive 边预测视图。
        unified,
        out_name=f"{unified['meta']['dataset_name']}_edge_org_codeIII_transductive",  # 输出名 patent_us_edge_org_codeIII_transductive。
        train_old=tr_old,  # 全关系训练窗口。
        val_old=va_old,  # 全关系验证窗口。
        test_old=te_old,  # 全关系测试窗口。
        supervised_old=target_old,  # 只有所有 org_codeIII 边是监督目标。
        extra_info={
            "target_original_rel_id": target_rel,  # 目标关系 ID，Patent-US 中是 2。
            "target_relation_name": "org_codeIII",  # 目标关系名。
            "protocol": "full_hetero_context_transductive",  # 协议：全异构上下文 + transductive。
        },
    )
```

这一步的关键是：task view 保留所有关系作为时间上下文，但 `edge_label` 只把 `org_codeIII` 监督边标为 1。

原始位置：`scripts/experiments/run_patent_preprocessing.py:444`

```python
    org_first_ts = {}  # 记录每个机构首次出现在目标关系 org_codeIII 中的时间。
    org_nodes = src[target_old]  # org_codeIII 的源节点就是机构节点。
    for edge_id, org_node in zip(target_old.tolist(), org_nodes.tolist()):  # 遍历每条目标边及其机构。
        org_first_ts.setdefault(int(org_node), float(ts[int(edge_id)]))  # 只记录第一次出现时间。
    org_sorted = sorted(org_first_ts.items(), key=lambda x: (x[1], x[0]))  # 机构按首次出现时间排序。
    org_ids = np.array([k for k, _ in org_sorted], dtype=np.int64)  # 得到机构节点 ID 序列。
    if org_ids.size >= 3:  # 至少 3 个机构时按机构做归纳切分。
        n_org = org_ids.size  # 机构数。
        n_train_org = max(1, int(n_org * 0.70))  # 前 70% 机构作为训练已见机构。
        n_val_org = max(1, int(n_org * 0.15))  # 中间 15% 机构作为验证未见机构。
        if n_train_org + n_val_org >= n_org:  # 保护测试机构不为空。
            n_train_org = max(1, n_org - 2)
            n_val_org = max(1, n_org - n_train_org - 1)
        train_orgs = org_ids[:n_train_org]  # 训练机构。
        val_orgs = org_ids[n_train_org : n_train_org + n_val_org]  # 验证未见机构。
        test_orgs = org_ids[n_train_org + n_val_org :]  # 测试未见机构。
        if test_orgs.size == 0 and val_orgs.size > 0:  # 再次保证 test 有机构。
            test_orgs = val_orgs[-1:]
            val_orgs = val_orgs[:-1]
    else:
        train_orgs = org_ids  # 机构过少时全部视为训练机构。
        val_orgs = np.array([], dtype=np.int64)  # 无验证未见机构。
        test_orgs = np.array([], dtype=np.int64)  # 无测试未见机构。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:468`

```python
    target_src = src[target_old]  # 目标边的源机构。
    ind_train_target = target_old[np.isin(target_src, train_orgs)]  # 训练目标边来自已见机构。
    ind_val_target = target_old[np.isin(target_src, val_orgs)]  # 验证目标边来自验证未见机构。
    ind_test_target = target_old[np.isin(target_src, test_orgs)]  # 测试目标边来自测试未见机构。
    if ind_val_target.size == 0 or ind_test_target.size == 0:  # 如果按机构切分失败。
        ind_train_target, ind_val_target, ind_test_target = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # 回退到时间切分。
        unseen_orgs = np.array([], dtype=np.int64)  # 回退时不声明 unseen org。
    else:
        unseen_orgs = np.concatenate([val_orgs, test_orgs], axis=0).astype(np.int64)  # 归纳评估中的未见机构集合。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:478`

```python
    train_cut_ind = float(ts[ind_train_target[-1]]) if ind_train_target.size > 0 else float(ts[all_sorted[-1]])  # 归纳训练截止时间。
    val_cut_ind = float(ts[ind_val_target[-1]]) if ind_val_target.size > 0 else train_cut_ind  # 归纳验证截止时间。
    involve_unseen = np.isin(src, unseen_orgs) | np.isin(dst, unseen_orgs) if unseen_orgs.size > 0 else np.zeros(n_edges, dtype=bool)  # 标记涉及未见机构的任何边。
    ind_tr_old = all_sorted[(ts[all_sorted] <= train_cut_ind) & (~involve_unseen[all_sorted])]  # 训练窗口排除所有涉及未见机构的上下文边。
    ind_va_old = all_sorted[(ts[all_sorted] > train_cut_ind) & (ts[all_sorted] <= val_cut_ind)]  # 验证窗口保留时间范围内上下文。
    ind_te_old = all_sorted[ts[all_sorted] > val_cut_ind]  # 测试窗口保留后续上下文。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:490`

```python
    ind_view = build_task_view(  # 构造 inductive 边预测视图。
        unified,
        out_name=f"{unified['meta']['dataset_name']}_edge_org_codeIII_inductive",  # 输出名 patent_us_edge_org_codeIII_inductive。
        train_old=ind_tr_old,  # 训练窗口，不含未见机构相关边。
        val_old=ind_va_old,  # 验证窗口。
        test_old=ind_te_old,  # 测试窗口。
        supervised_old=np.concatenate([ind_train_target, ind_val_target, ind_test_target], axis=0),  # 监督边只来自按归纳策略切出的目标边。
        extra_info={
            "target_original_rel_id": target_rel,  # 目标关系 ID。
            "target_relation_name": "org_codeIII",  # 目标关系名。
            "protocol": "full_hetero_context_inductive",  # 协议：全异构上下文 + inductive。
            "unseen_org_node_ids": unseen_orgs.tolist(),  # 记录未见机构节点 ID。
        },
    )
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:505`

```python
    node_view = build_task_view(  # 构造专利节点多标签任务视图。
        unified,
        out_name=f"{unified['meta']['dataset_name']}_node_patent_multilabel",  # 输出名 patent_us_node_patent_multilabel。
        train_old=tr_old,  # 使用 transductive 的全关系训练窗口。
        val_old=va_old,  # 使用 transductive 的全关系验证窗口。
        test_old=te_old,  # 使用 transductive 的全关系测试窗口。
        supervised_old=np.array([], dtype=np.int64),  # 节点任务没有监督边，edge_label 全 0。
        extra_info={"node_task": "patent_multilabel_codeIII", "node_task_view": True},  # 标记节点多标签任务。
    )
    return {
        "edge_org_codeIII_transductive": trans_view,  # 返回 transductive 边预测视图。
        "edge_org_codeIII_inductive": ind_view,  # 返回 inductive 边预测视图。
        "node_patent_multilabel": node_view,  # 返回节点多标签视图。
    }
```

## 入口 main：保存输出和导出适配器

原始位置：`scripts/experiments/run_patent_preprocessing.py:1024`

```python
def main() -> None:  # 脚本入口。
    ap = argparse.ArgumentParser()  # 创建命令行参数解析器。
    ap.add_argument("--project-root", default="/home/xuxj/2026/dyn_relation_memory")  # 项目根目录。
    ap.add_argument("--dataset-root", default="/home/xuxj/2026/datasets")  # 原始数据根目录。
    ap.add_argument("--python-bin", default="/home/xuxj/.conda/envs/pytorch/bin/python")  # 写入日志的 Python 路径。
    ap.add_argument("--feature-dim", type=int, default=128)  # 节点特征总维度，默认 128。
    ap.add_argument("--gpu-summary", default="GPU 0 preferred after nvidia-smi check")  # 写日志用 GPU 摘要。
    ap.add_argument("--datasets", default="", help="Comma-separated dataset names, e.g. patent_us,patent_cn")  # 可限制只处理 patent_us。
    args = ap.parse_args()  # 解析参数。

    project_root = Path(args.project_root).resolve()  # 规范化项目路径。
    dataset_root = Path(args.dataset_root).resolve()  # 规范化数据路径。
    processed_root = ensure_dir(project_root / "data" / "processed")  # 统一图输出目录。
    task_root = ensure_dir(processed_root / "tasks")  # task view 输出目录。
    adapter_root = ensure_dir(project_root / "data" / "interim" / "baseline_adapters")  # baseline adapter 输出目录。
    report_root = ensure_dir(project_root / "data" / "reports")  # 图表和统计输出目录。
```

原始位置：`scripts/experiments/run_patent_preprocessing.py:1050`

```python
    for spec in DATASET_SPECS:  # 遍历 patent_us、patent_cn、patent_in。
        if selected and spec["name"] not in selected:  # 如果 --datasets 指定了子集。
            continue  # 跳过未选中的数据集。
        unified, counts = build_unified_dataset(spec, dataset_root, args.feature_dim)  # 构建统一异构图。
        unified_path = processed_root / f"{spec['name']}_unified.pt"  # Patent-US 输出 data/processed/patent_us_unified.pt。
        torch.save(unified, unified_path)  # 保存统一图。
        print(f"[save] unified {unified_path}", flush=True)  # 打印保存路径。

        task_views = build_task_views(unified)  # 生成三个 Patent-US task view。
        task_paths = []  # 记录 task view 路径。
        for suffix, task_obj in task_views.items():  # 遍历 transductive、inductive、node 多标签。
            out_path = task_root / f"{spec['name']}_{suffix}.pt"  # 拼出 task view 文件路径。
            torch.save(task_obj, out_path)  # 保存 task view。
            export_baseline_bundle(task_obj, adapter_root / out_path.stem, out_path.stem)  # 同步导出 TGN/TGB2 适配格式。
            task_paths.append(out_path)  # 记录输出路径。
            print(f"[save] task {out_path}", flush=True)  # 打印 task 保存路径。
```

## 后续通用 task view 生成入口

正式 Patent-US 预处理已经在 `run_patent_preprocessing.py` 内部生成 task view。仓库里另有一个通用/平台入口 `scripts/datasets/build_task_views.py`，它用于从已有统一图再生成 patent 类 task views；平台 `builder == 'patent'` 时会调用它。

原始位置：`platform/backend/core/services/registry.py:55`

```python
    'raw_us': {  # 平台配置里的 US Patent 数据集键，不是正式脚本里的 patent_us 名。
        'label': 'US Patent',  # 前端显示名。
        'category': 'Patent Graph',  # 数据集类别。
        'description': '美国专利-公司-IPC 异构动态图，适合跨关系增强和社区演化分析。',  # 前端说明。
        'report_dir': 'raw_us',  # 报告目录名。
        'unified_path': 'data/processed/raw_us_unified.pt',  # 平台默认统一图路径。
        'tasks': ['edge_patent_ipc', 'edge_assignee_ipc_transductive', 'edge_assignee_ipc_inductive', 'node_patent_multilabel'],  # 平台声明的任务。
        'builder': 'patent',  # 让 job_builder 使用 scripts/datasets/build_task_views.py。
    },
```

原始位置：`platform/backend/core/services/job_builder.py:64`

```python
    if mode in {'build_tasks', 'prepare_and_build_tasks'}:  # 平台任务需要生成 task view 时进入。
        builder = dataset_cfg['builder']  # 读取 registry.py 中的 builder。
        if builder == 'paper':  # paper 类数据集走 build_paper_task_views.py。
            cmd = [
                'python', 'scripts/datasets/build_paper_task_views.py',
                '--input', str(unified_path),
                '--out-dir', str(task_dir),
                '--name-prefix', dataset_key,
                '--target-rel', str(int(payload.get('target_rel', -1))),
            ]
        else:  # patent 类 raw_us/raw_cn 走这里。
            cmd = [
                'python', 'scripts/datasets/build_task_views.py',  # 通用 patent task view 生成入口。
                '--input', str(unified_path),  # 输入已有统一图。
                '--out-dir', str(task_dir),  # 输出平台 task 目录。
                '--name-prefix', dataset_key,  # 输出前缀，平台 raw_us 时为 raw_us。
            ]
        commands.append(cmd)  # 写入最终 job 脚本。
```

原始位置：`scripts/datasets/build_task_views.py:286`

```python
def main():  # 通用 task view 生成脚本入口。
    ap = argparse.ArgumentParser("Build task-specific unified views from processed datasets")  # 参数解析器。
    ap.add_argument("--input", required=True, type=str, help="Path to unified .pt")  # 输入统一图 .pt。
    ap.add_argument("--out-dir", required=True, type=str)  # 输出目录。
    ap.add_argument("--name-prefix", default="", type=str)  # 输出文件名前缀。
    args = ap.parse_args()  # 解析命令行参数。

    in_path = Path(args.input)  # 输入路径对象。
    out_dir = Path(args.out_dir)  # 输出目录对象。
    out_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在。

    obj = torch.load(in_path, map_location="cpu")  # 加载统一图到 CPU。
    prefix = args.name_prefix.strip() or in_path.stem  # 前缀优先用命令行，否则用输入文件名。
    outputs = build_patent_task_views(obj, out_dir=out_dir, name_prefix=prefix)  # 生成 patent task views。
    print("generated task views:")  # 打印输出列表。
    for p in outputs:
        print(p)  # 逐个打印生成文件。
```

原始位置：`scripts/datasets/build_task_views.py:189`

```python
def build_patent_task_views(obj: Dict, out_dir: Path, name_prefix: str):  # 通用 patent task view 生成函数。
    meta_extra = obj.get("meta", {}).get("extra", {})  # 读取统一图 meta.extra。
    rel_patent_ipc = pick_relation_id(meta_extra, "patent_ipc")  # 查 patent_ipc 关系 ID。
    rel_assignee_ipc = pick_relation_id(meta_extra, "assignee_ipc")  # 查 assignee_ipc 关系 ID。
```

注意：这条通用路径查找的是 `patent_ipc` 和 `assignee_ipc`，而正式 Patent-US 预处理脚本使用的是 `patent_codeIII` 和 `org_codeIII`。所以对当前 `patent_us_process.md` 中的产物，准确入口仍是 `scripts/experiments/run_patent_preprocessing.py` 内置的 `build_task_views()`。

## 重点数据流

- 原始输入：
  - `/home/xuxj/2026/datasets/patent_us/all-us.csv`
  - `/home/xuxj/2026/datasets/patent_us/clsIII-reInd.csv`
  - `/home/xuxj/2026/datasets/patent_us/IPCSubClass_US.csv`
- 中间解析：
  - `orgID` 经 `split_multi_value()` 拆成机构 ID 列表。
  - `codeIIIId` 经 `split_multi_value()` 拆成 codeIII 重编号列表，并用 `code_meta_by_reid` 过滤。
  - `publicationDate/date_time/date_parsed` 经 `parse_publication_ts()` 变成浮点时间戳。
- 节点：
  - `patent:{patentId}` -> `node_type=0`
  - `org:{orgID}` -> `node_type=1`
  - `code:{codeIIIIdreId}` -> `node_type=2`
- 边：
  - 专利到机构：`patent_org=0`
  - 专利到 codeIII：`patent_codeIII=1`
  - 机构到 codeIII：`org_codeIII=2`
  - 共同申请机构双向边：`org_coorg=3`
- 特征：
  - 节点特征：字符 TF-IDF 文本特征 + 8 维结构/时间/类型特征。
  - 边特征：归一化时间、归一化关系 ID、源节点类型、目标节点类型。
- 统一图输出：
  - `data/processed/patent_us_unified.pt`
- 任务视图输出：
  - `data/processed/tasks/patent_us_edge_org_codeIII_transductive.pt`
  - `data/processed/tasks/patent_us_edge_org_codeIII_inductive.pt`
  - `data/processed/tasks/patent_us_node_patent_multilabel.pt`
- 监督掩码：
  - 边预测 task view 保留全异构上下文边。
  - `edge_label=1` 只表示目标监督边，主要是 `org_codeIII`。
  - 其它关系边在 task view 中作为上下文，`edge_label=0`。
- 归纳切分：
  - 按 `org_codeIII` 中机构首次出现时间排序机构。
  - 训练机构、验证未见机构、测试未见机构按约 70%/15%/15% 切分。
  - 归纳训练窗口排除涉及未见机构的上下文边。

## 与执行记录对应的 Patent-US 输出规模

来自 `patent_us_process.md`：

- `rows_total`: `200192`
- `rows_processed`: `200192`
- `rows_skipped_missing_org_or_code`: `0`
- `num_patent_nodes`: `200192`
- `num_org_nodes`: `520`
- `num_codeIII_nodes`: `609`
- `num_edges_patent_org`: `202361`
- `num_edges_patent_codeIII`: `417982`
- `num_edges_org_codeIII`: `422825`
- `num_edges_org_coorg`: `4596`
- `num_multilabel_classes`: `609`
- `num_labeled_patents`: `200192`
