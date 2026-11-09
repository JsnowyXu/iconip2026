# patent-cn 预处理代码注释

本文只标注 `patent_cn` 预处理的实际执行链路。根据 `patent_cn_process.md` 中记录的命令，实际入口是：

```bash
/home/xuxj/.conda/envs/pytorch/bin/python scripts/experiments/run_patent_preprocessing.py --project-root /home/xuxj/2026/dyn_relation_memory --dataset-root /home/xuxj/2026/datasets --python-bin /home/xuxj/.conda/envs/pytorch/bin/python
```

注意：仓库里还有 `scripts/datasets/build_task_views.py`，它是另一个 task view 生成入口，使用 `patent_ipc` / `assignee_ipc` 命名；本次 `patent_cn` 已产出的任务文件是 `patent_cn_edge_org_codeIII_transductive.pt`、`patent_cn_edge_org_codeIII_inductive.pt`、`patent_cn_node_patent_multilabel.pt`，对应的实际生成逻辑在 `scripts/experiments/run_patent_preprocessing.py` 内。

## 原始源码位置

- 入口与参数：`scripts/experiments/run_patent_preprocessing.py:1024-1145`
- patent-cn 配置：`scripts/experiments/run_patent_preprocessing.py:18-52`，其中 `patent_cn` 在 `31-40`
- 工具函数：`scripts/experiments/run_patent_preprocessing.py:55-196`
- baseline adapter 导出：`scripts/experiments/run_patent_preprocessing.py:199-238`
- task view 核心函数：`scripts/experiments/run_patent_preprocessing.py:343-518`
- unified graph 核心构建：`scripts/experiments/run_patent_preprocessing.py:521-939`
- 相邻但非本次实际入口：`scripts/datasets/build_task_views.py:67-240`

## 执行链路

1. `main()` 解析 `--project-root`、`--dataset-root`、`--feature-dim`、`--datasets`。
2. 若没有传 `--datasets`，遍历 `DATASET_SPECS` 中的 `patent_us`、`patent_cn`、`patent_in`；若传 `--datasets patent_cn`，只处理 `patent_cn`。
3. 对 `patent_cn` 调用 `build_unified_dataset(spec, dataset_root, feature_dim)`。
4. `build_unified_dataset()` 读取 `/home/xuxj/2026/datasets/patent_cn/all-cn.csv`、`clsIII-reInd.csv`、`IPCSubClass.csv`。
5. 每行专利拆出 `patentId`、`orgID`、`codeIIIId`、`publicationDate`、标题、摘要、申请人名。
6. 构造三类节点：`patent=0`、`org=1`、`codeIII=2`。
7. 构造四类边：`patent_org=0`、`patent_codeIII=1`、`org_codeIII=2`、`org_coorg=3`。
8. 生成节点特征：文本 TF-IDF 字符 ngram 特征 + 度数/时间/节点类型结构特征。
9. 生成 patent 节点多标签矩阵：每个 patent 对应多个 `codeIIIId` 标签列。
10. 保存 unified 数据到 `data/processed/patent_cn_unified.pt`。
11. 调用 `build_task_views(unified)`，生成三个任务视图：
    - `edge_org_codeIII_transductive`
    - `edge_org_codeIII_inductive`
    - `node_patent_multilabel`
12. 对每个任务视图保存 `.pt`，并调用 `export_baseline_bundle()` 导出 TGN 和 TGB2/STHN 格式适配文件。

## 逐行注释版代码片段

### 1. patent-cn 配置

原始位置：`scripts/experiments/run_patent_preprocessing.py:18-52`

```python
18  DATASET_SPECS = [  # 数据集配置列表；main() 会遍历这里决定处理哪些 patent 数据集。
19      {  # patent_us 配置块；不是本次 patent-cn 的重点。
20          "name": "patent_us",  # 输出文件名前缀和数据集目录名。
21          "short": "us",  # 简写，仅作元信息。
22          "csv": "all-us.csv",  # 原始主表文件名。
23          "ipc": "IPCSubClass_US.csv",  # IPC 子类描述表文件名。
24          "cls": "clsIII-reInd.csv",  # codeIII 重编号/映射表文件名。
25          "org_name_cols": ["orgname"],  # 组织名候选列。
26          "title_cols": ["title"],  # 标题候选列。
27          "summary_cols": ["summary"],  # 摘要候选列。
28          "date_cols": ["publicationDate", "date_time", "date_parsed"],  # 时间候选列。
29      },
30      {  # patent_cn 配置块；本次任务的关键配置。
31          "name": "patent_cn",  # 数据集名；用于定位 dataset_root/patent_cn 并生成 patent_cn_*.pt。
32          "short": "cn",  # 简写，当前核心流程里只作为描述性字段。
33          "csv": "all-cn.csv",  # 中国专利原始主表：/home/xuxj/2026/datasets/patent_cn/all-cn.csv。
34          "ipc": "IPCSubClass.csv",  # 中国/通用 IPC 子类说明表。
35          "cls": "clsIII-reInd.csv",  # codeIIIId、codeIII、codeIIIIdreId 的映射表。
36          "org_name_cols": ["applicantName", "orgname"],  # 申请人名优先读 applicantName，缺失时尝试 orgname。
37          "title_cols": ["title"],  # 专利标题列候选。
38          "summary_cols": ["summary"],  # 专利摘要列候选。
39          "date_cols": ["publicationDate"],  # patent-cn 使用 publicationDate 解析时间戳。
40      },
41      {  # patent_in 配置块；不是本次 patent-cn 的重点。
42          "name": "patent_in",  # 印度专利数据集名。
43          "short": "in",  # 简写。
44          "csv": "all-in.csv",  # 原始主表。
45          "ipc": "IPCSubClass_US.csv",  # IPC 表。
46          "cls": "clsIII-reInd.csv",  # codeIII 映射表。
47          "org_name_cols": ["orgname"],  # 组织名候选列。
48          "title_cols": ["title"],  # 标题候选列。
49          "summary_cols": ["summary"],  # 摘要候选列。
50          "date_cols": ["publicationDate", "date_time", "date_parsed"],  # 时间候选列。
51      },
52  ]  # 配置列表结束。
```

### 2. 常用清洗和划分工具

原始位置：`scripts/experiments/run_patent_preprocessing.py:55-196`

```python
55  def ensure_dir(path: Path) -> Path:  # 确保输出目录存在。
56      path.mkdir(parents=True, exist_ok=True)  # 递归创建目录；已存在时不报错。
57      return path  # 返回 Path，方便调用处链式使用。

60  def choose_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:  # 从候选列里选第一个真实存在的列。
61      lowered = {c.lower(): c for c in columns}  # 建立小写列名到原始列名的映射，避免大小写差异。
62      for cand in candidates:  # 按配置顺序尝试候选列。
63          if cand.lower() in lowered:  # 如果候选列存在。
64              return lowered[cand.lower()]  # 返回数据表里的原始列名。
65      return None  # 没找到则返回 None，调用处会走空字符串/空列表逻辑。

68  def load_ipc_table(path: Path) -> pd.DataFrame:  # 读取 IPC 子类描述表。
69      try:  # 优先按 CSV 读。
70          return pd.read_csv(path, engine="python")  # 使用 python engine，兼容不规则 CSV。
71      except Exception:  # 如果 CSV 解析失败，则降级为逐行文本解析。
72          lines = path.read_text(encoding="utf-8", errors="replace").splitlines()  # 容错读取文本。
73          if not lines:  # 空文件保护。
74              return pd.DataFrame({"Class Name": []})  # 返回空表但保留列名。
75          header = lines[0].strip().lstrip("\ufeff") or "Class Name"  # 去掉 BOM，取第一行为表头。
76          rows = [{"Class Name": line.strip()} for line in lines[1:] if line.strip()]  # 非空行转成 Class Name。
77          return pd.DataFrame(rows, columns=[header])  # 返回兼容 DataFrame。

80  def split_multi_value(value) -> List[str]:  # 解析 orgID/codeIIIId 这种多值字段。
81      if value is None:  # None 表示缺失。
82          return []  # 缺失字段返回空列表。
83      if isinstance(value, float) and math.isnan(value):  # pandas 缺失值常见为 NaN。
84          return []  # NaN 返回空列表。
85      if isinstance(value, (list, tuple, set)):  # 如果已经是集合类对象。
86          out = []  # 累积拆分结果。
87          for item in value:  # 递归处理每个元素。
88              out.extend(split_multi_value(item))  # 支持嵌套列表。
89          return out  # 返回扁平化结果。
91      text = str(value).strip()  # 转字符串并去首尾空白。
92      if not text or text.lower() in {"nan", "none", "null"}:  # 空值或文本形式缺失值。
93          return []  # 返回空列表。
95      if text.startswith("[") and text.endswith("]"):  # 字符串形如 "['1','2']"。
96          try:  # 尝试按 Python literal 解析。
97              parsed = ast.literal_eval(text)  # 安全解析列表/元组字面量。
98              return split_multi_value(parsed)  # 解析后递归拆分。
99          except Exception:  # 解析失败则继续按分隔符处理。
100             pass  # 保持容错。
102     text = text.replace("|", ";")  # 统一分隔符，把 | 当作 ;。
103     parts = [p.strip().strip("'").strip('"') for p in text.split(";")]  # 按分号拆分并去引号。
104     return [p for p in parts if p and p.lower() not in {"nan", "none", "null"}]  # 过滤空值后返回。

107 def normalize_org_name(raw: str) -> str:  # 标准化组织名称。
108     text = str(raw).strip().strip("'").strip('"')  # 转字符串并去掉外围空格/引号。
109     return " ".join(text.split())  # 把连续空白压成单个空格。

112 def parse_publication_ts(value) -> Tuple[float, str]:  # 把 publicationDate 转成时间戳和 yyyy-mm-dd 文本。
113     if value is None:  # 缺失时间。
114         return 0.0, ""  # 时间戳用 0，日期文本空。
115     if isinstance(value, float) and math.isnan(value):  # NaN 时间。
116         return 0.0, ""  # 同样返回默认值。
118     raw = str(value).strip()  # 转字符串。
119     if not raw:  # 空字符串。
120         return 0.0, ""  # 返回默认值。
122     dt = None  # 初始化解析结果。
123     for candidate in [raw]:  # 当前只尝试原始文本，但保留循环形式。
124         try:  # 严格解析。
125             dt = pd.to_datetime(candidate, errors="raise")  # pandas 时间解析。
126             break  # 成功则退出。
127         except Exception:  # 失败时记录 None。
128             dt = None  # 清空解析结果。
129     if dt is None and raw.isdigit() and len(raw) == 8:  # 兼容 20200131 这种格式。
130         try:  # 尝试按固定格式解析。
131             dt = datetime.strptime(raw, "%Y%m%d")  # 转 datetime。
132         except Exception:  # 固定格式也失败。
133             dt = None  # 保持失败状态。
134     if dt is None:  # 前面都没解析成功。
135         try:  # 最后用 coerce 容错解析。
136             dt = pd.to_datetime(raw, errors="coerce")  # 失败会得到 NaT。
137         except Exception:  # 极端异常。
138             dt = None  # 保持失败状态。
139     if dt is None or pd.isna(dt):  # 解析仍失败。
140         return 0.0, raw  # 时间戳默认 0，但保留原始文本。
141     py_dt = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt  # 统一成 Python datetime。
142     return float(py_dt.timestamp()), py_dt.strftime("%Y-%m-%d")  # 返回秒级时间戳和标准日期文本。

145 def sort_old_indices(indices: Iterable[int], ts: np.ndarray) -> np.ndarray:  # 按时间排序旧边索引。
146     idx = np.asarray(list(indices), dtype=np.int64).reshape(-1)  # 转成一维 int64 数组。
147     idx = idx[(idx >= 0) & (idx < ts.shape[0])]  # 过滤越界索引。
148     if idx.size == 0:  # 空输入保护。
149         return np.array([], dtype=np.int64)  # 返回空 int64。
150     idx = np.unique(idx)  # 去重，避免同一边重复进入视图。
151     order = np.lexsort((idx, ts[idx]))  # 主键时间戳，副键原索引，保证稳定顺序。
152     return idx[order].astype(np.int64)  # 返回排序后的旧索引。

155 def temporal_partition_sorted(sorted_old: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # 对已排序索引做时间切分。
156     sorted_old = np.asarray(sorted_old, dtype=np.int64).reshape(-1)  # 标准化输入形状。
157     if sorted_old.size == 0:  # 没有边时。
158         z = np.array([], dtype=np.int64)  # 构造空数组。
159         return z, z, z  # train/val/test 全空。
160     n = sorted_old.size  # 边数。
161     n_train = max(1, int(n * train_ratio))  # 至少保留一个 train。
162     n_val = max(1, int(n * val_ratio))  # 至少保留一个 val。
163     if n_train + n_val >= n:  # 如果切分挤占了 test。
164         n_train = max(1, n - 2)  # 调整 train，给 val/test 留空间。
165         n_val = max(1, n - n_train - 1)  # 调整 val。
166     train_old = sorted_old[:n_train]  # 时间最早部分为 train。
167     val_old = sorted_old[n_train : n_train + n_val]  # 中间部分为 val。
168     test_old = sorted_old[n_train + n_val :]  # 时间最新部分为 test。
169     if test_old.size == 0 and val_old.size > 0:  # test 为空时。
170         test_old = val_old[-1:]  # 从 val 尾部挪一个到 test。
171         val_old = val_old[:-1]  # val 删除被挪走的边。
172     if val_old.size == 0 and train_old.size > 1:  # val 为空时。
173         val_old = train_old[-1:]  # 从 train 尾部挪一个到 val。
174         train_old = train_old[:-1]  # train 删除被挪走的边。
175     return train_old.astype(np.int64), val_old.astype(np.int64), test_old.astype(np.int64)  # 返回三份旧索引。

178 def map_old_to_new(old_indices: np.ndarray, old_to_new: np.ndarray) -> np.ndarray:  # 把原图旧边索引映射到 task view 内的新边索引。
179     arr = np.asarray(old_indices, dtype=np.int64).reshape(-1)  # 标准化输入。
180     if arr.size == 0:  # 空输入。
181         return np.array([], dtype=np.int64)  # 返回空数组。
182     arr = arr[(arr >= 0) & (arr < old_to_new.shape[0])]  # 过滤越界旧索引。
183     mapped = old_to_new[arr]  # 查映射表。
184     mapped = mapped[mapped >= 0]  # -1 表示该旧边不在当前视图里，要过滤掉。
185     if mapped.size == 0:  # 没有有效映射。
186         return np.array([], dtype=np.int64)  # 返回空数组。
187     return np.unique(mapped).astype(np.int64)  # 去重并返回新索引。

190 def remove_indices(base: np.ndarray, to_remove: np.ndarray, ts: np.ndarray) -> np.ndarray:  # 从一组旧索引里移除另一组。
191     base = np.asarray(base, dtype=np.int64).reshape(-1)  # 标准化 base。
192     to_remove = np.asarray(to_remove, dtype=np.int64).reshape(-1)  # 标准化待移除集合。
193     if base.size == 0 or to_remove.size == 0:  # 任意一边为空时。
194         return sort_old_indices(base, ts)  # 只需保证排序。
195     mask = ~np.isin(base, to_remove)  # 构造保留 mask。
196     return sort_old_indices(base[mask], ts)  # 移除并按时间排序。
```

### 3. unified graph 构建入口与原始表读取

原始位置：`scripts/experiments/run_patent_preprocessing.py:521-558`

```python
521 def build_unified_dataset(spec: Dict, dataset_root: Path, feature_dim: int) -> Tuple[Dict, Dict]:  # 构建一个 patent 数据集的统一动态图对象。
522     ds_dir = dataset_root / spec["name"]  # 对 patent_cn 来说是 /home/xuxj/2026/datasets/patent_cn。
523     print(f"[prepare] start {spec['name']}", flush=True)  # 打印开始处理日志。
524     df = pd.read_csv(ds_dir / spec["csv"])  # 读取 all-cn.csv 主表。
525     cls_df = pd.read_csv(ds_dir / spec["cls"])  # 读取 clsIII-reInd.csv，提供 codeIIIId/codeIIIIdreId 映射。
526     ipc_df = load_ipc_table(ds_dir / spec["ipc"])  # 读取 IPCSubClass.csv，提供 codeIII 描述文本。

528     title_col = choose_first(df.columns, spec["title_cols"])  # patent_cn 中通常选中 title。
529     summary_col = choose_first(df.columns, spec["summary_cols"])  # patent_cn 中通常选中 summary。
530     date_col = choose_first(df.columns, spec["date_cols"])  # patent_cn 中选中 publicationDate。
531     org_name_col = choose_first(df.columns, spec["org_name_cols"])  # patent_cn 优先 applicantName，其次 orgname。

533     ipc_desc_by_code: Dict[str, str] = {}  # codeIII 字符串到 IPC 描述文本的映射。
534     if {"Class", "Name"}.issubset(ipc_df.columns):  # 如果 IPC 表是标准 Class/Name 两列。
535         for _, row in ipc_df.iterrows():  # 遍历 IPC 行。
536             code = str(row["Class"]).strip()  # 读取 IPC class code。
537             desc = str(row["Name"]).strip()  # 读取 IPC 描述。
538             ipc_desc_by_code[code] = desc  # 存入描述映射。
539     else:  # 如果 IPC 表不是标准两列。
540         col = ipc_df.columns[0]  # 取第一列作为原始文本列。
541         for _, row in ipc_df.iterrows():  # 遍历每一行。
542             text = str(row[col]).strip()  # 取整行文本。
543             code = text.split()[0].strip().strip('"')  # 第一个 token 当作 code。
544             desc = text[len(code):].strip().strip('"')  # 剩余文本当作描述。
545             ipc_desc_by_code[code] = desc or text  # 若描述为空则保留整行文本。

547     code_meta_by_reid: Dict[str, Dict] = {}  # 以重编号 codeIIIIdreId 为 key 的 code 元信息。
548     for _, row in cls_df.iterrows():  # 遍历 codeIII 映射表。
549         codeiiiid = str(row["codeIIIId"]).strip()  # 原始 codeIIIId，用于多标签列名。
550         codeiii = str(row["codeIII"]).strip()  # IPC subclass 代码，如 A01B。
551         reid = str(row["codeIIIIdreId"]).strip()  # 重编号 id，all-cn.csv 的 codeIIIId 会用它对齐。
552         desc = ipc_desc_by_code.get(codeiii, "")  # 取 IPC 描述，缺失则为空。
553         code_meta_by_reid[reid] = {  # 记录重编号到真实标签/文本的映射。
554             "codeIIIId": codeiiiid,  # 原始标签 id。
555             "codeIIIIdreId": reid,  # 重编号标签 id。
556             "codeIII": codeiii,  # IPC subclass 代码。
557             "description": desc,  # IPC 描述文本。
558         }  # 单个 code 元信息结束。
```

### 4. 节点注册与关系定义

原始位置：`scripts/experiments/run_patent_preprocessing.py:560-621`

```python
560     node_key_to_id: Dict[str, int] = {}  # 全局节点 key 到连续 node id 的映射。
561     node_type: List[int] = []  # 每个 node id 的节点类型；patent=0/org=1/codeIII=2。
562     node_text: List[str] = []  # 用于 TF-IDF 的节点文本。
563     node_name: List[str] = []  # 节点展示名或 fallback 文本。
564     patent_nodes: List[int] = []  # patent 类型节点 id 列表。
565     patent_raw_ids: List[str] = []  # 原始 patentId 列表。
566     patent_titles: List[str] = []  # 专利标题列表。
567     patent_summaries: List[str] = []  # 专利摘要列表。
568     patent_dates: List[str] = []  # 标准化日期文本列表。
569     patent_ts: List[float] = []  # 专利发布时间戳列表。
570     patent_code_ids: Dict[int, List[str]] = {}  # patent 节点到重编号 codeIIIIdreId 的映射。
571     patent_label_ids: Dict[int, List[str]] = {}  # patent 节点到原始 codeIIIId 标签的映射。
572     patent_node_to_idx: Dict[int, int] = {}  # patent node id 到 patent_nodes 内部行号的映射。
573     org_nodes: List[int] = []  # org 类型节点 id 列表。
574     org_raw_ids: List[str] = []  # 原始 orgID 列表。
575     org_names: List[str] = []  # 组织名列表。
576     code_nodes: List[int] = []  # codeIII 类型节点 id 列表。
577     code_raw_ids: List[str] = []  # 原始 codeIIIId 列表。
578     code_reindexed_ids: List[str] = []  # 重编号 codeIIIIdreId 列表。
579     code_names: List[str] = []  # codeIII 字符串，如 A01B。
580     code_descs: List[str] = []  # codeIII 描述文本。

582     def get_node(key: str, ntype: int, text: str, display_name: str, registry: str, raw_id: str, extra_text: str = "") -> int:  # 注册或复用节点。
583         if key in node_key_to_id:  # 如果节点已出现过。
584             return node_key_to_id[key]  # 直接返回已有 node id。
585         nid = len(node_type)  # 新节点 id 是当前节点数。
586         node_key_to_id[key] = nid  # 建立 key 到 node id 的映射。
587         node_type.append(ntype)  # 记录节点类型。
588         full_text = " ".join([t for t in [text, extra_text] if t]).strip()  # 合并主文本和补充文本。
589         node_text.append(full_text)  # 存入文本特征来源。
590         node_name.append(display_name)  # 存入展示名。
591         if registry == "patent":  # patent 节点分支。
592             patent_nodes.append(nid)  # 记录 patent node id。
593             patent_node_to_idx[nid] = len(patent_nodes) - 1  # 记录 patent 内部行号。
594             patent_raw_ids.append(raw_id)  # 记录原始 patentId。
595             patent_titles.append(text)  # 记录标题。
596             patent_summaries.append(extra_text)  # 记录摘要。
597             patent_dates.append("")  # 日期先占位，随后按行更新。
598             patent_ts.append(0.0)  # 时间戳先占位，随后按行更新。
599             patent_code_ids[nid] = []  # 初始化 code reid 列表。
600             patent_label_ids[nid] = []  # 初始化原始 label id 列表。
601         elif registry == "org":  # 组织节点分支。
602             org_nodes.append(nid)  # 记录 org node id。
603             org_raw_ids.append(raw_id)  # 记录原始 orgID。
604             org_names.append(display_name)  # 记录组织名。
605         elif registry == "code":  # codeIII 节点分支。
606             code_nodes.append(nid)  # 记录 code node id。
607             code_raw_ids.append(raw_id)  # 记录原始 codeIIIId。
608             code_reindexed_ids.append(key.split(":", 1)[1])  # 从 key 里取 codeIIIIdreId。
609             code_names.append(display_name)  # 记录 codeIII 名称。
610             code_descs.append(extra_text)  # 记录 IPC 描述。
611         return nid  # 返回新建 node id。

613     rel_map = {"patent_org": 0, "patent_codeIII": 1, "org_codeIII": 2, "org_coorg": 3}  # 四种边类型的全局 id。
614     src_list: List[int] = []  # 边源节点列表。
615     dst_list: List[int] = []  # 边目标节点列表。
616     ts_list: List[float] = []  # 边时间戳列表。
617     rel_list: List[int] = []  # 边类型列表。
619     row_stats = Counter()  # 行级统计计数器。
620     org_pair_counter = Counter()  # 组织共现对计数。
621     org_code_counter = Counter()  # 组织-code 组合计数。
```

### 5. 每行专利转成异构动态图边

原始位置：`scripts/experiments/run_patent_preprocessing.py:622-718`

```python
622     for _, row in df.iterrows():  # 遍历 all-cn.csv 的每一条专利记录。
623         patent_raw = str(row["patentId"]).strip()  # 读取原始 patentId。
624         patent_title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""  # 读取标题，缺失则空。
625         patent_summary = str(row[summary_col]).strip() if summary_col and pd.notna(row[summary_col]) else ""  # 读取摘要，缺失则空。
626         patent_ts_val, patent_date_text = parse_publication_ts(row[date_col] if date_col else "")  # 解析 publicationDate。
627         org_ids = split_multi_value(row["orgID"])  # 拆分多申请人/组织 id。
628         code_ids = split_multi_value(row["codeIIIId"])  # 拆分多个 codeIIIIdreId。
629         org_names_row = split_multi_value(row[org_name_col]) if org_name_col else []  # 拆分申请人名称。
630         class_codes_row = split_multi_value(row["classCode"]) if "classCode" in df.columns else []  # 仅用于统计是否有 classCode 文本。

632         org_ids = [x for x in org_ids if x]  # 过滤空组织 id。
633         code_ids = [x for x in code_ids if x in code_meta_by_reid]  # 只保留能在 clsIII-reInd.csv 对齐的 code id。
634         if not org_ids or not code_ids:  # 没有组织或没有有效 code 的专利不能形成核心关系。
635             row_stats["rows_skipped_missing_org_or_code"] += 1  # 统计跳过行。
636             continue  # 进入下一条专利。

638         patent_node = get_node(  # 注册 patent 节点。
639             key=f"patent:{patent_raw}",  # patent 节点唯一 key。
640             ntype=0,  # 节点类型 0 表示 patent。
641             text=patent_title,  # 主文本用标题。
642             display_name=f"patent:{patent_raw}",  # 展示名用 patent id。
643             registry="patent",  # 进入 get_node 的 patent 分支。
644             raw_id=patent_raw,  # 保存原始 patentId。
645             extra_text=patent_summary,  # 附加文本用摘要。
646         )
647         pat_idx = patent_node_to_idx[patent_node]  # 找到该 patent 在 patent 属性数组中的位置。
648         patent_dates[pat_idx] = patent_date_text  # 回填标准日期。
649         patent_ts[pat_idx] = patent_ts_val  # 回填时间戳。

651         org_node_ids = []  # 当前专利对应的 org node id 列表。
652         for i, org_raw in enumerate(dict.fromkeys(org_ids)):  # 去重并保持原顺序遍历组织 id。
653             org_name = normalize_org_name(org_names_row[i]) if i < len(org_names_row) else f"org:{org_raw}"  # 对齐组织名，缺失时用 org:id。
654             org_node = get_node(  # 注册或复用 org 节点。
655                 key=f"org:{org_raw}",  # org 节点唯一 key。
656                 ntype=1,  # 节点类型 1 表示 org。
657                 text=org_name,  # 组织名作为文本特征。
658                 display_name=org_name,  # 展示名也是组织名。
659                 registry="org",  # 进入 org 分支。
660                 raw_id=str(org_raw),  # 保存原始 orgID。
661             )
662             org_node_ids.append(org_node)  # 将 org node id 放入当前专利组织集合。

664         code_node_ids = []  # 当前专利对应的 codeIII node id 列表。
665         for code_reid in dict.fromkeys(code_ids):  # 去重并保持顺序遍历 codeIIIIdreId。
666             meta = code_meta_by_reid[code_reid]  # 取 code 元信息。
667             code_name = meta["codeIII"]  # IPC subclass 名称。
668             code_desc = meta["description"]  # IPC 描述文本。
669             code_node = get_node(  # 注册或复用 codeIII 节点。
670                 key=f"code:{code_reid}",  # code 节点 key 使用重编号。
671                 ntype=2,  # 节点类型 2 表示 codeIII。
672                 text=code_name,  # 主文本为 code 名称。
673                 display_name=code_name,  # 展示名为 code 名称。
674                 registry="code",  # 进入 code 分支。
675                 raw_id=str(meta["codeIIIId"]),  # 保存原始 codeIIIId。
676                 extra_text=code_desc,  # 附加文本为 IPC 描述。
677             )
678             code_node_ids.append(code_node)  # 当前专利连接到该 code 节点。
679             patent_code_ids[patent_node].append(code_reid)  # 保存重编号 code id。
680             patent_label_ids[patent_node].append(meta["codeIIIId"])  # 保存原始标签 id，用于多标签矩阵。

682         patent_code_ids[patent_node] = sorted(set(patent_code_ids[patent_node]))  # patent 的 code reid 去重排序。
683         patent_label_ids[patent_node] = sorted(set(patent_label_ids[patent_node]), key=lambda x: int(x))  # patent 标签 id 去重并按数值排序。

685         for org_node in org_node_ids:  # patent 到每个组织建边。
686             src_list.append(patent_node)  # 源节点是 patent。
687             dst_list.append(org_node)  # 目标节点是 org。
688             ts_list.append(patent_ts_val)  # 边时间为专利公开时间。
689             rel_list.append(rel_map["patent_org"])  # 边类型为 patent_org。
690         row_stats["patent_org_edges"] += len(org_node_ids)  # 统计 patent_org 边数。

692         for code_node in code_node_ids:  # patent 到每个 codeIII 建边。
693             src_list.append(patent_node)  # 源节点是 patent。
694             dst_list.append(code_node)  # 目标节点是 codeIII。
695             ts_list.append(patent_ts_val)  # 边时间为专利公开时间。
696             rel_list.append(rel_map["patent_codeIII"])  # 边类型为 patent_codeIII。
697         row_stats["patent_codeIII_edges"] += len(code_node_ids)  # 统计 patent_codeIII 边数。

699         for org_node in org_node_ids:  # 每个组织和每个 codeIII 组合建边。
700             for code_node in code_node_ids:  # 遍历当前专利的 codeIII。
701                 src_list.append(org_node)  # 源节点是 org。
702                 dst_list.append(code_node)  # 目标节点是 codeIII。
703                 ts_list.append(patent_ts_val)  # 时间仍用专利公开时间。
704                 rel_list.append(rel_map["org_codeIII"])  # 这是 edge 预测主任务的目标关系。
705                 org_code_counter[(org_node, code_node)] += 1  # 统计组织-code 对出现次数。
706         row_stats["org_codeIII_edges"] += len(org_node_ids) * len(code_node_ids)  # 统计 org_codeIII 边数。

708         for i in range(len(org_node_ids)):  # 构造同一专利下组织共现边。
709             for j in range(i + 1, len(org_node_ids)):  # 避免重复 pair。
710                 a, b = org_node_ids[i], org_node_ids[j]  # 两个共同申请组织。
711                 src_list.extend([a, b])  # 加双向边的源节点。
712                 dst_list.extend([b, a])  # 加双向边的目标节点。
713                 ts_list.extend([patent_ts_val, patent_ts_val])  # 两条边时间相同。
714                 rel_list.extend([rel_map["org_coorg"], rel_map["org_coorg"]])  # 关系类型都是 org_coorg。
715                 org_pair_counter[(a, b)] += 1  # 统计组织共现 pair。
716         row_stats["org_coorg_edges"] += max(0, len(org_node_ids) * (len(org_node_ids) - 1))  # 统计双向共现边。
717         row_stats["rows_processed"] += 1  # 成功处理行数加一。
718         row_stats["class_code_text_nonempty"] += int(len(class_codes_row) > 0)  # 记录 classCode 是否非空。
```

### 6. 特征、标签和 unified 对象

原始位置：`scripts/experiments/run_patent_preprocessing.py:720-939`

```python
720     src = np.asarray(src_list, dtype=np.int64)  # 边源节点数组。
721     dst = np.asarray(dst_list, dtype=np.int64)  # 边目标节点数组。
722     ts = np.asarray(ts_list, dtype=np.float32)  # 边时间戳数组。
723     rel = np.asarray(rel_list, dtype=np.int64)  # 边关系类型数组。
724     node_type_arr = np.asarray(node_type, dtype=np.int64)  # 节点类型数组。
726     if ts.size == 0:  # 如果没有生成任何边，说明原始数据/映射有严重问题。
727         raise RuntimeError(f"{spec['name']} produced no edges")  # 直接失败，避免保存空数据。
729     order = np.lexsort((np.arange(ts.size), ts))  # 按时间戳排序，索引作为稳定副键。
730     src = src[order]  # 边源按时间重排。
731     dst = dst[order]  # 边目标按时间重排。
732     ts = ts[order]  # 时间戳重排。
733     rel = rel[order]  # 关系类型重排。

735     in_deg = np.bincount(dst, minlength=len(node_type))  # 节点入度。
736     out_deg = np.bincount(src, minlength=len(node_type))  # 节点出度。
737     deg = in_deg + out_deg  # 节点总度数。
738     first_ts = np.full(len(node_type), ts.max(), dtype=np.float32)  # 初始化节点首次出现时间。
739     last_ts = np.zeros(len(node_type), dtype=np.float32)  # 初始化节点最后出现时间。
740     for s, d, t in zip(src, dst, ts):  # 遍历所有边更新节点时间范围。
741         first_ts[s] = min(first_ts[s], t)  # 更新源节点首次时间。
742         first_ts[d] = min(first_ts[d], t)  # 更新目标节点首次时间。
743         last_ts[s] = max(last_ts[s], t)  # 更新源节点最后时间。
744         last_ts[d] = max(last_ts[d], t)  # 更新目标节点最后时间。
745     ts_min, ts_max = float(ts.min()), float(ts.max())  # 全图时间范围。
746     ts_span = max(ts_max - ts_min, 1.0)  # 时间跨度，至少为 1 防止除零。
747     first_ts_norm = ((first_ts - ts_min) / ts_span).astype(np.float32)  # 归一化首次时间。
748     last_ts_norm = ((last_ts - ts_min) / ts_span).astype(np.float32)  # 归一化最后时间。

750     texts = [t if t else n for t, n in zip(node_text, node_name)]  # 节点文本为空时用节点名兜底。
751     text_dim = max(8, feature_dim - 8)  # 总特征维度中留 8 维给结构特征。
752     vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=text_dim, lowercase=False)  # 字符级 TF-IDF，适合中英文和代码混合文本。
753     text_feat = vectorizer.fit_transform(texts).toarray().astype(np.float32)  # 得到节点文本特征矩阵。
754     if text_feat.shape[1] < text_dim:  # 如果实际词表不足指定维度。
755         pad = np.zeros((text_feat.shape[0], text_dim - text_feat.shape[1]), dtype=np.float32)  # 构造补零列。
756         text_feat = np.concatenate([text_feat, pad], axis=1)  # 补齐文本特征维度。

758     struct_feat = np.stack(  # 组装 8 维结构特征。
760             np.log1p(in_deg) / max(np.log1p(in_deg).max(), 1.0),  # 归一化入度。
761             np.log1p(out_deg) / max(np.log1p(out_deg).max(), 1.0),  # 归一化出度。
762             np.log1p(deg) / max(np.log1p(deg).max(), 1.0),  # 归一化总度。
763             first_ts_norm,  # 节点首次出现时间。
764             last_ts_norm,  # 节点最后出现时间。
765             (node_type_arr == 0).astype(np.float32),  # patent 类型 one-hot。
766             (node_type_arr == 1).astype(np.float32),  # org 类型 one-hot。
767             (node_type_arr == 2).astype(np.float32),  # codeIII 类型 one-hot。
771     node_feat = np.concatenate([text_feat, struct_feat], axis=1).astype(np.float32)  # 拼接成最终节点特征。

773     patent_node_arr = np.asarray(patent_nodes, dtype=np.int64)  # node_multilabel 任务关注的 patent 节点 id。
774     label_id_order = sorted({cid for labels in patent_label_ids.values() for cid in labels}, key=lambda x: int(x))  # 所有出现过的 codeIIIId 标签列。
775     label_meta_by_id = {}  # 标签 id 到 code 名称/描述的映射。
776     for meta in code_meta_by_reid.values():  # 遍历 code 元信息。
777         label_meta_by_id.setdefault(  # 只在第一次遇到该 codeIIIId 时写入。
778             meta["codeIIIId"],  # 标签 key。
779             {"codeIII": meta["codeIII"], "description": meta["description"]},  # 标签展示信息。
780         )
781     label_id_to_col = {cid: i for i, cid in enumerate(label_id_order)}  # 标签 id 到矩阵列号。
782     label_matrix = np.zeros((len(patent_nodes), len(label_id_order)), dtype=np.float32)  # patent 多标签矩阵。
783     for i, pnode in enumerate(patent_nodes):  # 遍历每个 patent 节点。
784         labels = [label_id_to_col[cid] for cid in patent_label_ids.get(pnode, []) if cid in label_id_to_col]  # 找到该 patent 的标签列。
785         if labels:  # 如果有标签。
786             label_matrix[i, labels] = 1.0  # 对应列置 1。

788     rel_norm = rel.astype(np.float32) / max(int(rel.max()), 1)  # 关系 id 归一化。
789     ts_norm = ((ts - ts_min) / ts_span).astype(np.float32)  # 边时间归一化。
790     edge_feat = np.stack(  # 构造边特征。
792             ts_norm,  # 归一化时间。
793             rel_norm,  # 归一化关系类型。
794             node_type_arr[src].astype(np.float32) / 2.0,  # 源节点类型归一化。
795             node_type_arr[dst].astype(np.float32) / 2.0,  # 目标节点类型归一化。
798     ).astype(np.float32)  # 最终边特征为 float32。

800     all_sorted = sort_old_indices(np.arange(ts.shape[0], dtype=np.int64), ts)  # 全部边按时间排序后的旧索引。
801     target_old = sort_old_indices(np.where(rel == rel_map["org_codeIII"])[0], ts)  # 主边预测目标：org_codeIII 边。
802     trans_tr, trans_va, trans_te = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # 对目标边做 transductive 时间切分。
803     org_first_ts = {}  # 每个 org 第一次出现在 org_codeIII 目标边中的时间。
804     for edge_id in target_old.tolist():  # 遍历目标边。
805         org_first_ts.setdefault(int(src[edge_id]), float(ts[edge_id]))  # 因 target_old 已按时间排序，所以第一次 set 即首次时间。
806     org_sorted = sorted(org_first_ts.items(), key=lambda x: (x[1], x[0]))  # 按首次时间排序 org。
807     org_nodes_sorted = np.array([k for k, _ in org_sorted], dtype=np.int64)  # 得到 org 节点序列。
808     if org_nodes_sorted.size >= 3:  # org 数足够时可做 inductive 划分。
809         n_org = org_nodes_sorted.size  # org 数量。
810         n_train_org = max(1, int(n_org * 0.70))  # 训练 org 数。
811         n_val_org = max(1, int(n_org * 0.15))  # 验证 org 数。
812         if n_train_org + n_val_org >= n_org:  # 保证 test org 不为空。
813             n_train_org = max(1, n_org - 2)  # 调整训练 org。
814             n_val_org = max(1, n_org - n_train_org - 1)  # 调整验证 org。
815         train_orgs = org_nodes_sorted[:n_train_org]  # 早期 org 做 train。
816         val_orgs = org_nodes_sorted[n_train_org : n_train_org + n_val_org]  # 中间 org 做 val。
817         test_orgs = org_nodes_sorted[n_train_org + n_val_org :]  # 后期 org 做 test。
818     else:  # org 数不足时。
819         train_orgs = org_nodes_sorted  # 全放 train。
820         val_orgs = np.array([], dtype=np.int64)  # val org 为空。
821         test_orgs = np.array([], dtype=np.int64)  # test org 为空。
823     train_org_set = set(train_orgs.tolist())  # train org 集合。
824     val_org_set = set(val_orgs.tolist())  # val org 集合。
825     test_org_set = set(test_orgs.tolist())  # test org 集合。
826     ind_train = [eid for eid in target_old.tolist() if int(src[eid]) in train_org_set]  # inductive train 监督边。
827     ind_val = [eid for eid in target_old.tolist() if int(src[eid]) in val_org_set]  # inductive val 监督边，源 org 未在 train org 中。
828     ind_test = [eid for eid in target_old.tolist() if int(src[eid]) in test_org_set]  # inductive test 监督边，源 org 未在 train org 中。

830     train_cut = float(ts[trans_tr[-1]]) if trans_tr.size > 0 else float(ts[all_sorted[-1]])  # transductive train 的时间边界。
831     val_cut = float(ts[trans_va[-1]]) if trans_va.size > 0 else train_cut  # transductive val 的时间边界。
832     base_tr = all_sorted[ts[all_sorted] <= train_cut]  # unified 默认 split 的 train 边窗口。
833     base_va = all_sorted[(ts[all_sorted] > train_cut) & (ts[all_sorted] <= val_cut)]  # unified 默认 split 的 val 边窗口。
834     base_te = all_sorted[ts[all_sorted] > val_cut]  # unified 默认 split 的 test 边窗口。

836     unified = {  # 统一数据对象，后续 task view 都从这里切。
837         "num_nodes": int(len(node_type)),  # 全图节点数。
838         "num_rels": int(len(rel_map)),  # 关系类型数，固定 4。
839         "num_node_types": 3,  # 节点类型数，固定 patent/org/codeIII。
840         "node_feat": torch.from_numpy(node_feat.astype(np.float32)),  # 节点特征 tensor。
841         "node_type": torch.from_numpy(node_type_arr.astype(np.int64)),  # 节点类型 tensor。
842         "edges": {  # 边数据。
843             "src": torch.from_numpy(src.astype(np.int64)),  # 源节点 tensor。
844             "dst": torch.from_numpy(dst.astype(np.int64)),  # 目标节点 tensor。
845             "ts": torch.from_numpy(ts.astype(np.float32)),  # 时间戳 tensor。
846             "rel": torch.from_numpy(rel.astype(np.int64)),  # 关系类型 tensor。
847             "edge_feat": torch.from_numpy(edge_feat.astype(np.float32)),  # 边特征 tensor。
848             "edge_label": torch.ones(src.shape[0], dtype=torch.float32),  # unified 全图边都标为正边。
849         },
850         "split": {  # unified 默认时间窗口切分。
851             "train": torch.from_numpy(base_tr),  # train 边旧索引。
852             "val": torch.from_numpy(base_va),  # val 边旧索引。
853             "test": torch.from_numpy(base_te),  # test 边旧索引。
854         },
855         "node_multiclass": None,  # 本数据没有单标签节点分类。
856         "node_multilabel": {  # patent 多标签任务数据。
857             "node_ids": torch.from_numpy(patent_node_arr.astype(np.int64)),  # 被监督的 patent node id。
858             "labels": torch.from_numpy(label_matrix.astype(np.float32)),  # 多标签矩阵。
859             "label_names": [  # 标签元信息。
861                     "codeIIIId": cid,  # 原始标签 id。
862                     "codeIII": label_meta_by_id[cid]["codeIII"],  # IPC subclass 名。
863                     "description": label_meta_by_id[cid]["description"],  # 标签描述。
868         "meta": {  # 元信息。
869             "dataset_name": spec["name"],  # patent_cn。
870             "extra": {  # 额外元信息。
871                 "relation_map": rel_map,  # 关系名到 id。
872                 "node_type_map": {"patent": 0, "org": 1, "codeIII": 2},  # 节点类型名到 id。
881                 "patent_attributes": {  # patent 节点属性。
882                     "node_ids": patent_nodes,  # patent node id。
883                     "patent_id": patent_raw_ids,  # 原始 patentId。
884                     "title": patent_titles,  # 标题。
885                     "summary": patent_summaries,  # 摘要。
886                     "publication_date": patent_dates,  # 日期文本。
887                     "publication_ts": patent_ts,  # 时间戳。
889                 "org_attributes": {  # org 节点属性。
890                     "node_ids": org_nodes,  # org node id。
891                     "org_id": org_raw_ids,  # 原始 orgID。
892                     "name": org_names,  # 组织名。
894                 "codeIII_attributes": {  # codeIII 节点属性。
895                     "node_ids": code_nodes,  # code node id。
896                     "codeIIIId": code_raw_ids,  # 原始 codeIIIId。
897                     "codeIIIIdreId": code_reindexed_ids,  # 重编号 code id。
898                     "codeIII": code_names,  # IPC subclass 名。
899                     "description": code_descs,  # 描述文本。
901                 "task_splits": {  # task view 可复用的监督边切分信息。
902                     "org_codeIII_transductive": {  # transductive 目标边切分。
903                         "edge_indices_train": trans_tr.tolist(),  # 训练目标边旧索引。
904                         "edge_indices_val": trans_va.tolist(),  # 验证目标边旧索引。
905                         "edge_indices_test": trans_te.tolist(),  # 测试目标边旧索引。
907                     "org_codeIII_inductive": {  # inductive 目标边切分。
908                         "edge_indices_train": ind_train,  # seen org 上的训练目标边。
909                         "edge_indices_val": ind_val,  # unseen/后续 org 上的验证目标边。
910                         "edge_indices_test": ind_test,  # unseen/后续 org 上的测试目标边。
911                         "unseen_org_node_ids": np.concatenate([val_orgs, test_orgs], axis=0).astype(np.int64).tolist(),  # inductive 未见 org。
914                 "hetero_main_benchmark_ok": True,  # 标记该异构图可用于主 benchmark。
919     counts = {  # 处理统计，写入 patent_cn_process.md。
920         "rows_total": int(len(df)),  # 原始行数。
921         "rows_processed": int(row_stats["rows_processed"]),  # 成功处理行数。
922         "rows_skipped_missing_org_or_code": int(row_stats["rows_skipped_missing_org_or_code"]),  # 跳过行数。
923         "num_patent_nodes": int(len(patent_nodes)),  # patent 节点数。
924         "num_org_nodes": int(len(org_nodes)),  # org 节点数。
925         "num_codeIII_nodes": int(len(code_nodes)),  # codeIII 节点数。
926         "num_edges_patent_org": int((rel == rel_map["patent_org"]).sum()),  # patent_org 边数。
927         "num_edges_patent_codeIII": int((rel == rel_map["patent_codeIII"]).sum()),  # patent_codeIII 边数。
928         "num_edges_org_codeIII": int((rel == rel_map["org_codeIII"]).sum()),  # org_codeIII 边数。
929         "num_edges_org_coorg": int((rel == rel_map["org_coorg"]).sum()),  # org_coorg 边数。
930         "num_unique_org_code_pairs": int(len(org_code_counter)),  # 唯一 org-code 对数量。
931         "num_unique_org_org_pairs": int(len(org_pair_counter)),  # 唯一 org-org 共现对数量。
932         "num_multilabel_classes": int(len(label_id_order)),  # 多标签类别数。
933         "num_labeled_patents": int(label_matrix.shape[0]),  # 有标签 patent 行数。
935     print(  # 输出完成日志。
936         f"[prepare] finished {spec['name']} nodes={unified['num_nodes']} edges={unified['edges']['src'].shape[0]} labels={unified['node_multilabel']['labels'].shape[1]}",
937         flush=True,  # 立即刷新。
938     )
939     return unified, counts  # 返回统一图对象和统计。
```

### 7. task view 单视图构造

原始位置：`scripts/experiments/run_patent_preprocessing.py:343-408`

```python
343 def build_task_view(  # 把 unified 全图按窗口切成一个训练任务视图。
344     obj: Dict,  # unified 图对象。
345     out_name: str,  # task view 名称，如 patent_cn_edge_org_codeIII_transductive。
346     train_old: np.ndarray,  # train 窗口内的旧边索引。
347     val_old: np.ndarray,  # val 窗口内的旧边索引。
348     test_old: np.ndarray,  # test 窗口内的旧边索引。
349     supervised_old: np.ndarray,  # 需要作为监督目标的旧边索引，edge_label 会在这些边上置 1。
350     extra_info: Dict,  # task 协议等额外元信息。
351 ) -> Dict:  # 返回可保存为 .pt 的任务对象。
352     src_all = obj["edges"]["src"].numpy().astype(np.int64)  # 取全图源节点。
353     dst_all = obj["edges"]["dst"].numpy().astype(np.int64)  # 取全图目标节点。
354     ts_all = obj["edges"]["ts"].numpy().astype(np.float32)  # 取全图时间戳。
355     rel_all = obj["edges"]["rel"].numpy().astype(np.int64)  # 取全图关系 id。
356     edge_feat_all = obj["edges"]["edge_feat"].numpy().astype(np.float32)  # 取全图边特征。

358     selected_old = np.concatenate([train_old, val_old, test_old], axis=0).astype(np.int64)  # 当前视图保留的旧边集合。
359     selected_old = sort_old_indices(selected_old, ts_all)  # 按时间重新排序并去重。

361     old_to_new = np.full((ts_all.shape[0],), -1, dtype=np.int64)  # 初始化旧边索引到新边索引映射。
362     old_to_new[selected_old] = np.arange(selected_old.shape[0], dtype=np.int64)  # 为保留边写入新索引。

364     train_new = map_old_to_new(train_old, old_to_new)  # train split 转成视图内新索引。
365     val_new = map_old_to_new(val_old, old_to_new)  # val split 转成视图内新索引。
366     test_new = map_old_to_new(test_old, old_to_new)  # test split 转成视图内新索引。
367     edge_label = np.isin(selected_old, supervised_old).astype(np.float32)  # 只有监督目标边为 1，其它上下文边为 0。

369     meta = dict(obj.get("meta", {}))  # 复制 unified 元信息。
370     extra = dict(meta.get("extra", {}))  # 复制 extra。
371     extra["task_view_name"] = out_name  # 写入任务名。
372     extra["context_protocol"] = {  # 说明这是全异构上下文 + edge_label 监督 mask 的协议。
373         "enabled": True,  # 启用上下文协议。
374         "supervision_mode": "edge_label_mask",  # 监督由 edge_label 指示。
375         "preserve_global_rel_ids": True,  # 保持关系 id 与 unified 一致。
376         "selected_edges": int(selected_old.size),  # 视图中总边数。
377         "supervised_edges": int(edge_label.sum()),  # 监督边数。
378         "train_window_edges": int(train_new.size),  # train 窗口边数。
379         "val_window_edges": int(val_new.size),  # val 窗口边数。
380         "test_window_edges": int(test_new.size),  # test 窗口边数。
381     }
382     extra.update(extra_info)  # 合并 task 特有信息，如 protocol/target_relation_name。
383     meta["dataset_name"] = out_name  # 让下游把该对象识别为具体 task。
384     meta["extra"] = extra  # 写回 extra。

386     return {  # 返回 task view 对象。
387         "num_nodes": int(obj["num_nodes"]),  # 保留全图节点数。
388         "num_rels": int(obj["num_rels"]),  # 保留全图关系数。
389         "num_node_types": int(obj["num_node_types"]),  # 保留节点类型数。
390         "node_feat": obj["node_feat"],  # 节点特征不裁剪。
391         "node_type": obj["node_type"],  # 节点类型不裁剪。
392         "edges": {  # 只裁剪边。
393             "src": torch.from_numpy(src_all[selected_old].astype(np.int64)),  # 视图内边源节点。
394             "dst": torch.from_numpy(dst_all[selected_old].astype(np.int64)),  # 视图内边目标节点。
395             "ts": torch.from_numpy(ts_all[selected_old].astype(np.float32)),  # 视图内边时间。
396             "rel": torch.from_numpy(rel_all[selected_old].astype(np.int64)),  # 视图内边关系。
397             "edge_feat": torch.from_numpy(edge_feat_all[selected_old].astype(np.float32)),  # 视图内边特征。
398             "edge_label": torch.from_numpy(edge_label.astype(np.float32)),  # 监督 mask。
399         },
400         "split": {  # 视图内新边索引 split。
401             "train": torch.from_numpy(train_new),  # train 新索引。
402             "val": torch.from_numpy(val_new),  # val 新索引。
403             "test": torch.from_numpy(test_new),  # test 新索引。
404         },
405         "node_multiclass": obj.get("node_multiclass"),  # 传递节点单分类字段。
406         "node_multilabel": obj.get("node_multilabel"),  # 传递 patent 多标签字段。
407         "meta": meta,  # task 元信息。
408     }  # task view 结束。
```

### 8. 三个 patent-cn task view 的生成入口

原始位置：`scripts/experiments/run_patent_preprocessing.py:411-518`

```python
411 def build_task_views(unified: Dict) -> Dict[str, Dict]:  # 从 unified 图生成三个专利任务视图。
412     rel_map = unified["meta"]["extra"]["relation_map"]  # 读取关系名到 id 的映射。
413     target_rel = int(rel_map["org_codeIII"])  # 主边预测目标关系是 org_codeIII。
414     src = unified["edges"]["src"].numpy().astype(np.int64)  # 全图源节点。
415     dst = unified["edges"]["dst"].numpy().astype(np.int64)  # 全图目标节点。
416     ts = unified["edges"]["ts"].numpy().astype(np.float32)  # 全图时间戳。
417     rel = unified["edges"]["rel"].numpy().astype(np.int64)  # 全图关系 id。
418     n_edges = rel.shape[0]  # 全图边数。

420     all_sorted = sort_old_indices(np.arange(n_edges, dtype=np.int64), ts)  # 全部边按时间排序。
421     target_old = sort_old_indices(np.where(rel == target_rel)[0], ts)  # 所有 org_codeIII 目标边。
422     tr_tgt, va_tgt, te_tgt = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # 目标边时间切分。

424     train_cut = float(ts[tr_tgt[-1]]) if tr_tgt.size > 0 else float(ts[all_sorted[-1]])  # 训练时间边界。
425     val_cut = float(ts[va_tgt[-1]]) if va_tgt.size > 0 else train_cut  # 验证时间边界。
427     tr_old = all_sorted[ts[all_sorted] <= train_cut]  # transductive train 保留该时间前所有关系边。
428     va_old = all_sorted[(ts[all_sorted] > train_cut) & (ts[all_sorted] <= val_cut)]  # transductive val 保留中间窗口所有关系边。
429     te_old = all_sorted[ts[all_sorted] > val_cut]  # transductive test 保留后续所有关系边。
430     trans_view = build_task_view(  # 构造 transductive edge task。
431         unified,  # 输入 unified。
432         out_name=f"{unified['meta']['dataset_name']}_edge_org_codeIII_transductive",  # patent_cn_edge_org_codeIII_transductive。
433         train_old=tr_old,  # train 上下文边。
434         val_old=va_old,  # val 上下文边。
435         test_old=te_old,  # test 上下文边。
436         supervised_old=target_old,  # 只有 org_codeIII 边作为监督正边。
437         extra_info={  # 任务额外元信息。
438             "target_original_rel_id": target_rel,  # org_codeIII 的关系 id。
439             "target_relation_name": "org_codeIII",  # 目标关系名。
440             "protocol": "full_hetero_context_transductive",  # 全异构上下文 transductive 协议。
441         },
442     )

444     org_first_ts = {}  # inductive 划分需要每个 org 首次出现时间。
445     org_nodes = src[target_old]  # org_codeIII 边的源节点就是 org。
446     for edge_id, org_node in zip(target_old.tolist(), org_nodes.tolist()):  # 遍历目标边和对应 org。
447         org_first_ts.setdefault(int(org_node), float(ts[int(edge_id)]))  # 记录 org 首次目标边时间。
448     org_sorted = sorted(org_first_ts.items(), key=lambda x: (x[1], x[0]))  # 按首次时间排序 org。
449     org_ids = np.array([k for k, _ in org_sorted], dtype=np.int64)  # 得到 org id 序列。
450     if org_ids.size >= 3:  # 至少三个 org 才能切 train/val/test。
451         n_org = org_ids.size  # org 数量。
452         n_train_org = max(1, int(n_org * 0.70))  # 训练 org 数。
453         n_val_org = max(1, int(n_org * 0.15))  # 验证 org 数。
454         if n_train_org + n_val_org >= n_org:  # 防止 test org 为空。
455             n_train_org = max(1, n_org - 2)  # 给 val/test 留空间。
456             n_val_org = max(1, n_org - n_train_org - 1)  # 调整 val 数。
457         train_orgs = org_ids[:n_train_org]  # 早期出现的 org 是 seen org。
458         val_orgs = org_ids[n_train_org : n_train_org + n_val_org]  # 中间 org 是验证 unseen org。
459         test_orgs = org_ids[n_train_org + n_val_org :]  # 后期 org 是测试 unseen org。
460         if test_orgs.size == 0 and val_orgs.size > 0:  # 兜底保证 test 不空。
461             test_orgs = val_orgs[-1:]  # 从 val 挪最后一个到 test。
462             val_orgs = val_orgs[:-1]  # val 删除该 org。
463     else:  # org 太少时。
464         train_orgs = org_ids  # 全部放 train。
465         val_orgs = np.array([], dtype=np.int64)  # val 空。
466         test_orgs = np.array([], dtype=np.int64)  # test 空。

468     target_src = src[target_old]  # 目标边源节点，即 org。
469     ind_train_target = target_old[np.isin(target_src, train_orgs)]  # seen org 的目标边作为 train 监督。
470     ind_val_target = target_old[np.isin(target_src, val_orgs)]  # val unseen org 的目标边作为 val 监督。
471     ind_test_target = target_old[np.isin(target_src, test_orgs)]  # test unseen org 的目标边作为 test 监督。
472     if ind_val_target.size == 0 or ind_test_target.size == 0:  # 如果 inductive 划分退化。
473         ind_train_target, ind_val_target, ind_test_target = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # 回退时间切分。
474         unseen_orgs = np.array([], dtype=np.int64)  # 回退后没有显式 unseen org。
475     else:  # inductive 切分有效。
476         unseen_orgs = np.concatenate([val_orgs, test_orgs], axis=0).astype(np.int64)  # val/test org 都视为训练未见。

478     train_cut_ind = float(ts[ind_train_target[-1]]) if ind_train_target.size > 0 else float(ts[all_sorted[-1]])  # inductive train 时间边界。
479     val_cut_ind = float(ts[ind_val_target[-1]]) if ind_val_target.size > 0 else train_cut_ind  # inductive val 时间边界。
480     involve_unseen = np.isin(src, unseen_orgs) | np.isin(dst, unseen_orgs) if unseen_orgs.size > 0 else np.zeros(n_edges, dtype=bool)  # 标记涉及 unseen org 的边。
481     ind_tr_old = all_sorted[(ts[all_sorted] <= train_cut_ind) & (~involve_unseen[all_sorted])]  # train 上下文剔除 unseen org 相关边，避免泄漏。
482     ind_va_old = all_sorted[(ts[all_sorted] > train_cut_ind) & (ts[all_sorted] <= val_cut_ind)]  # val 窗口所有关系边。
483     ind_te_old = all_sorted[ts[all_sorted] > val_cut_ind]  # test 窗口所有关系边。
484     if ind_val_target.size > 0 and ind_va_old.size == 0:  # 如果 val 窗口没有上下文边。
485         ind_va_old = sort_old_indices(ind_val_target, ts)  # 至少放入 val 监督边。
486         ind_tr_old = remove_indices(ind_tr_old, ind_va_old, ts)  # 确保 train 不重复包含它。
487     if ind_test_target.size > 0 and ind_te_old.size == 0:  # 如果 test 窗口没有上下文边。
488         ind_te_old = sort_old_indices(ind_test_target, ts)  # 至少放入 test 监督边。
489         ind_tr_old = remove_indices(ind_tr_old, ind_te_old, ts)  # 确保 train 不重复包含它。
490     ind_view = build_task_view(  # 构造 inductive edge task。
491         unified,  # 输入 unified。
492         out_name=f"{unified['meta']['dataset_name']}_edge_org_codeIII_inductive",  # patent_cn_edge_org_codeIII_inductive。
493         train_old=ind_tr_old,  # inductive train 上下文边。
494         val_old=ind_va_old,  # inductive val 上下文边。
495         test_old=ind_te_old,  # inductive test 上下文边。
496         supervised_old=np.concatenate([ind_train_target, ind_val_target, ind_test_target], axis=0),  # 三段目标边共同构成监督 mask。
497         extra_info={  # 任务额外信息。
498             "target_original_rel_id": target_rel,  # 目标关系 id。
499             "target_relation_name": "org_codeIII",  # 目标关系名。
500             "protocol": "full_hetero_context_inductive",  # 全异构上下文 inductive 协议。
501             "unseen_org_node_ids": unseen_orgs.tolist(),  # 训练阶段未见 org 列表。
502         },
503     )

505     node_view = build_task_view(  # 构造节点多标签任务视图。
506         unified,  # 输入 unified。
507         out_name=f"{unified['meta']['dataset_name']}_node_patent_multilabel",  # patent_cn_node_patent_multilabel。
508         train_old=tr_old,  # 使用 transductive 的 train 时间窗口。
509         val_old=va_old,  # 使用 transductive 的 val 时间窗口。
510         test_old=te_old,  # 使用 transductive 的 test 时间窗口。
511         supervised_old=np.array([], dtype=np.int64),  # 节点任务不靠 edge_label 监督，所以这里为空。
512         extra_info={"node_task": "patent_multilabel_codeIII", "node_task_view": True},  # 标记为 patent 多标签节点任务。
513     )
514     return {  # 返回三个视图，由 main() 保存。
515         "edge_org_codeIII_transductive": trans_view,  # 输出 patent_cn_edge_org_codeIII_transductive.pt。
516         "edge_org_codeIII_inductive": ind_view,  # 输出 patent_cn_edge_org_codeIII_inductive.pt。
517         "node_patent_multilabel": node_view,  # 输出 patent_cn_node_patent_multilabel.pt。
518     }
```

### 9. baseline adapter 导出

原始位置：`scripts/experiments/run_patent_preprocessing.py:199-238`

```python
199 def export_tgn(obj: Dict, out_dir: Path, name: str) -> None:  # 导出 TGN 风格文件。
200     ensure_dir(out_dir)  # 创建输出目录。
201     src = obj["edges"]["src"].numpy()  # 取源节点。
202     dst = obj["edges"]["dst"].numpy()  # 取目标节点。
203     ts = obj["edges"]["ts"].numpy()  # 取时间戳。
204     edge_feat = obj["edges"]["edge_feat"].numpy()  # 取边特征。
206     with (out_dir / f"{name}.csv").open("w", encoding="utf-8") as f:  # 写一个简单 CSV。
207         f.write("u,i,ts,label,f0,f1,f2,f3\n")  # 表头。
208         for i in range(len(src)):  # 遍历每条边。
209             f0, f1, f2, f3 = [float(x) for x in edge_feat[i, :4]]  # 取前 4 维边特征。
210             f.write(f"{int(src[i])},{int(dst[i])},{float(ts[i])},1,{f0},{f1},{f2},{f3}\n")  # 写正边记录。
212     df = pd.DataFrame({"u": src + 1, "i": dst + 1, "ts": ts, "label": 1, "idx": range(1, len(src) + 1)})  # TGN ml_*.csv 使用 1-based 节点/边索引。
213     df.to_csv(out_dir / f"ml_{name}.csv", index=False)  # 写 ml_{task}.csv。
214     np.save(out_dir / f"ml_{name}.npy", np.vstack([np.zeros((1, edge_feat.shape[1]), dtype=np.float32), edge_feat]))  # 写边特征 npy，首行补 0。
215     node_feat = obj["node_feat"].numpy().astype(np.float32)  # 取节点特征。
216     np.save(out_dir / f"ml_{name}_node.npy", np.vstack([np.zeros((1, node_feat.shape[1]), dtype=np.float32), node_feat]))  # 写节点特征 npy，首行补 0。

219 def export_tgb2(obj: Dict, out_dir: Path, name: str) -> None:  # 导出 TGB2/STHN 风格文件。
220     ensure_dir(out_dir)  # 创建输出目录。
221     pd.DataFrame(  # 构造 edgelist 表。
223             "u": obj["edges"]["src"].numpy(),  # 源节点。
224             "i": obj["edges"]["dst"].numpy(),  # 目标节点。
225             "ts": obj["edges"]["ts"].numpy(),  # 时间戳。
226             "edge_type": obj["edges"]["rel"].numpy(),  # 关系类型。
227             "label": 1,  # 全部是正边记录。
228             "idx": range(obj["edges"]["src"].shape[0]),  # 0-based 边索引。
230     ).to_csv(out_dir / f"{name}_edgelist.csv", index=False)  # 写 {task}_edgelist.csv。
231     pd.DataFrame({"node_id": range(obj["num_nodes"]), "node_type": obj["node_type"].numpy()}).to_csv(  # 写节点类型表。
232         out_dir / f"{name}_nodetype.csv", index=False  # 输出 {task}_nodetype.csv。
233     )

236 def export_baseline_bundle(obj: Dict, out_root: Path, name: str) -> None:  # 一个任务统一导出两类 baseline 适配格式。
237     export_tgn(obj, out_root / "tgn", name)  # 导出 TGN 目录。
238     export_tgb2(obj, out_root / "tgb2", name)  # 导出 TGB2/STHN 目录。
```

### 10. 总入口 main

原始位置：`scripts/experiments/run_patent_preprocessing.py:1024-1145`

```python
1024 def main() -> None:  # patent 预处理总入口。
1025     ap = argparse.ArgumentParser()  # 创建命令行参数解析器。
1026     ap.add_argument("--project-root", default="/home/xuxj/2026/dyn_relation_memory")  # 项目根目录。
1027     ap.add_argument("--dataset-root", default="/home/xuxj/2026/datasets")  # 原始数据根目录。
1028     ap.add_argument("--python-bin", default="/home/xuxj/.conda/envs/pytorch/bin/python")  # 记录到 process md 的 Python 路径。
1029     ap.add_argument("--feature-dim", type=int, default=128)  # 节点特征总维度，默认 128。
1030     ap.add_argument("--gpu-summary", default="GPU 0 preferred after nvidia-smi check")  # 只写日志，不驱动重任务。
1031     ap.add_argument("--datasets", default="", help="Comma-separated dataset names, e.g. patent_us,patent_cn")  # 可指定只跑 patent_cn。
1032     args = ap.parse_args()  # 解析命令行参数。

1034     project_root = Path(args.project_root).resolve()  # 解析项目根目录绝对路径。
1035     dataset_root = Path(args.dataset_root).resolve()  # 解析数据根目录绝对路径。
1036     processed_root = ensure_dir(project_root / "data" / "processed")  # processed 输出根目录。
1037     task_root = ensure_dir(processed_root / "tasks")  # task view 输出目录。
1038     adapter_root = ensure_dir(project_root / "data" / "interim" / "baseline_adapters")  # baseline adapter 输出目录。
1039     report_root = ensure_dir(project_root / "data" / "reports")  # 分析报告输出目录。

1041     commands = [  # 写入 process md 的命令记录。
1042         "nvidia-smi",  # 记录当时查看 GPU 的命令。
1043         f"{args.python_bin} scripts/experiments/run_patent_preprocessing.py --project-root {project_root} --dataset-root {dataset_root} --python-bin {args.python_bin}",  # 记录预处理命令。
1044     ]

1046     dataset_summaries = []  # 汇总写入 patent.md 的数据结构。
1047     stats_rows = []  # 汇总写入 dataset_stats_summary.csv 的统计行。
1048     selected = {x.strip() for x in str(args.datasets).split(",") if x.strip()}  # 如果传 --datasets，构造筛选集合。

1050     for spec in DATASET_SPECS:  # 遍历 patent_us/patent_cn/patent_in 配置。
1051         if selected and spec["name"] not in selected:  # 如果指定了数据集且当前不是目标。
1052             continue  # 跳过。
1053         unified, counts = build_unified_dataset(spec, dataset_root, args.feature_dim)  # 构建 unified 图和统计。
1054         unified_path = processed_root / f"{spec['name']}_unified.pt"  # 对 patent_cn 是 data/processed/patent_cn_unified.pt。
1055         torch.save(unified, unified_path)  # 保存 unified 图。
1056         print(f"[save] unified {unified_path}", flush=True)  # 打印保存日志。

1058         task_views = build_task_views(unified)  # 生成三个 task view。
1059         task_paths = []  # 记录 task 输出路径。
1060         for suffix, task_obj in task_views.items():  # 遍历三个 task view。
1061             out_path = task_root / f"{spec['name']}_{suffix}.pt"  # 拼接输出路径，如 patent_cn_edge_org_codeIII_inductive.pt。
1062             torch.save(task_obj, out_path)  # 保存 task view。
1063             export_baseline_bundle(task_obj, adapter_root / out_path.stem, out_path.stem)  # 同步导出 TGN/TGB2 adapter。
1064             task_paths.append(out_path)  # 记录 task 路径。
1065             print(f"[save] task {out_path}", flush=True)  # 打印 task 保存日志。

1067         analyze_stats = [analyze_one(unified, report_root)]  # 对 unified 生成轻量分析图和 stats。
1068         for task_obj in task_views.values():  # 遍历 task view。
1069             analyze_stats.append(analyze_one(task_obj, report_root))  # 对每个 task 也生成 stats/图。
1070         stats_rows.extend(analyze_stats)  # 合并统计行。
1071         print(f"[analyze] completed {spec['name']}", flush=True)  # 打印分析完成。

1124         outputs = [unified_path] + task_paths + [  # 组织 process md 输出清单。
1125             adapter_root / p.stem / "tgn" / f"ml_{p.stem}.csv" for p in task_paths  # 每个 task 的 TGN ml CSV。
1126         ] + [
1127             adapter_root / p.stem / "tgb2" / f"{p.stem}_edgelist.csv" for p in task_paths  # 每个 task 的 TGB2 edgelist。
1128         ]
1129         write_process_md(  # 写 patent_cn_process.md 之类的运行记录。
1130             path=project_root / f"{spec['name']}_process.md",  # 对 patent_cn 是 patent_cn_process.md。
1131             dataset_name=spec["name"],  # 数据集名。
1132             python_bin=args.python_bin,  # Python 路径。
1133             gpu_summary=args.gpu_summary,  # GPU 摘要。
1134             commands=commands,  # 命令记录。
1135             counts=counts,  # 统计信息。
1136             outputs=outputs,  # 输出路径清单。
1137         )

1139     pd.DataFrame(stats_rows).drop_duplicates(subset=["dataset"]).to_csv(report_root / "dataset_stats_summary.csv", index=False)  # 写总统计 CSV。
1140     write_summary_md(project_root, dataset_summaries)  # 写 patent.md 总结。
1141     print("[done] patent preprocessing pipeline completed", flush=True)  # 打印总完成日志。

1144 if __name__ == "__main__":  # 脚本直接运行入口。
1145     main()  # 调用 main。
```

### 11. 相邻但非本次实际入口：scripts/datasets/build_task_views.py

原始位置：`scripts/datasets/build_task_views.py:67-240`

这份脚本也能从 unified `.pt` 生成 task view，但它查找的是 `patent_ipc` 和 `assignee_ipc` 关系，并输出 `*_edge_assignee_ipc_*`，与本次已产出的 `patent_cn_edge_org_codeIII_*` 不一致。因此它更像旧版/通用入口，不是 `patent_cn_process.md` 记录命令所走的实际代码。

```python
67  def build_edge_view(  # 通用 edge view 构造器，逻辑类似 run_patent_preprocessing.py 的 build_task_view。
68      obj: Dict,  # 输入 unified 图。
69      out_name: str,  # 输出任务名。
70      train_target_old: np.ndarray,  # train 目标边旧索引。
71      val_target_old: np.ndarray,  # val 目标边旧索引。
72      test_target_old: np.ndarray,  # test 目标边旧索引。
73      context_keep_mask: np.ndarray = None,  # 可选上下文边保留 mask。
74      extra_info: Dict = None,  # 额外协议信息。
75  ) -> Dict:  # 返回 task view。
76      src_all = obj["edges"]["src"].numpy().astype(np.int64)  # 全图源节点。
77      dst_all = obj["edges"]["dst"].numpy().astype(np.int64)  # 全图目标节点。
78      ts_all = obj["edges"]["ts"].numpy().astype(np.float32)  # 全图时间。
79      rel_all = obj["edges"]["rel"].numpy().astype(np.int64)  # 全图关系。
80      raw_num_rels = int(obj.get("num_rels", int(rel_all.max()) + 1 if rel_all.size > 0 else 1))  # 关系数量。
82      all_old_sorted = sort_old_indices(np.arange(ts_all.shape[0], dtype=np.int64), ts_all)  # 全边按时间排序。
83      keep_mask = np.ones(ts_all.shape[0], dtype=bool) if context_keep_mask is None else np.asarray(context_keep_mask, dtype=bool)  # 默认保留所有上下文边。
88      train_cut = float(ts_all[train_target_old[-1]]) if train_target_old.size > 0 else float(ts_all[all_old_sorted[0]])  # train 时间边界。
89      val_cut = float(ts_all[val_target_old[-1]]) if val_target_old.size > 0 else train_cut  # val 时间边界。
91      train_old = all_old_sorted[keep_mask[all_old_sorted] & (ts_all[all_old_sorted] <= train_cut)]  # train 上下文边。
92      val_old = all_old_sorted[keep_mask[all_old_sorted] & (ts_all[all_old_sorted] > train_cut) & (ts_all[all_old_sorted] <= val_cut)]  # val 上下文边。
96      test_old = all_old_sorted[~assigned[all_old_sorted]]  # 剩余边进入 test。
110     supervised_old = sort_old_indices(np.concatenate([train_target_old, val_target_old, test_target_old], axis=0), ts_all)  # 监督边集合。
111     edge_label = np.isin(selected_old, supervised_old).astype(np.float32)  # 监督 mask。
121     meta = dict(obj.get("meta", {}))  # 复制元信息。
123     extra.pop("task_splits", None)  # 该脚本会移除旧 task_splits，避免嵌套历史切分。
126     extra["context_protocol"] = {  # 记录上下文协议。
127         "enabled": True,  # 开启协议。
128         "supervision_mode": "edge_label_mask",  # edge_label 指定监督边。
129         "preserve_global_rel_ids": True,  # 保留全局关系 id。
135         "train_supervised_edges": int(np.isin(train_old, supervised_old).sum()),  # train 中监督边数。
136         "val_supervised_edges": int(np.isin(val_old, supervised_old).sum()),  # val 中监督边数。
137         "test_supervised_edges": int(np.isin(test_old, supervised_old).sum()),  # test 中监督边数。
144     return {  # 返回 task view。
150         "edges": {  # 裁剪后的边。
151             "src": torch.from_numpy(src.astype(np.int64)),  # 源节点。
152             "dst": torch.from_numpy(dst.astype(np.int64)),  # 目标节点。
153             "ts": torch.from_numpy(ts.astype(np.float32)),  # 时间戳。
154             "rel": torch.from_numpy(rel.astype(np.int64)),  # 关系类型。
155             "edge_feat": torch.from_numpy(edge_feat.astype(np.float32)),  # 该脚本重新构造 2 维边特征。
156             "edge_label": torch.from_numpy(edge_label.astype(np.float32)),  # 监督 mask。
158         "split": {  # 新索引 split。
159             "train": torch.from_numpy(train_new),  # train。
160             "val": torch.from_numpy(val_new),  # val。
161             "test": torch.from_numpy(test_new),  # test。
189 def build_patent_task_views(obj: Dict, out_dir: Path, name_prefix: str):  # 旧/通用 patent task view 生成入口。
190     meta_extra = obj.get("meta", {}).get("extra", {})  # 取元信息。
191     rel_patent_ipc = pick_relation_id(meta_extra, "patent_ipc")  # 查 patent_ipc 关系。
192     rel_assignee_ipc = pick_relation_id(meta_extra, "assignee_ipc")  # 查 assignee_ipc 关系。
201     if rel_patent_ipc >= 0:  # 如果 unified 中存在 patent_ipc 才生成。
206             out_name=f"{name_prefix}_edge_patent_ipc",  # 输出名不是 patent_cn_edge_org_codeIII。
219     task_splits = meta_extra.get("task_splits", {}) if isinstance(meta_extra, dict) else {}  # 读取旧切分。
220     if rel_assignee_ipc >= 0:  # 如果存在 assignee_ipc 才生成。
232             out_name=f"{name_prefix}_edge_assignee_ipc_transductive",  # 输出 assignee_ipc transductive。
```

## 熟悉代码后应重点看的数据流

最重要的数据流可以压缩成这一条：

`all-cn.csv 行` -> `patent/org/codeIII 三类节点` -> `patent_org、patent_codeIII、org_codeIII、org_coorg 四类边` -> `按 publicationDate 排序` -> `patent_cn_unified.pt` -> `org_codeIII 作为 edge 任务监督目标` -> `edge_label mask 标出监督边，其它关系保留为上下文边` -> `三个 task .pt` -> `TGN/TGB2 adapter`。

重点关注：

- `codeIIIId` 的两套 id：原始 `codeIIIId` 用作多标签列名，重编号 `codeIIIIdreId` 用于从原始行对齐 `code_meta_by_reid` 和 code 节点 key。
- `org_codeIII` 是边预测核心关系；`patent_org`、`patent_codeIII`、`org_coorg` 在 task view 里主要是上下文关系。
- `edge_label` 不是原始正负标签，而是“这条保留下来的边是否属于当前监督目标边集合”的 mask。
- transductive 任务按目标关系 `org_codeIII` 的时间切分确定时间边界，但每个窗口保留所有关系边作为异构上下文。
- inductive 任务按 org 首次出现时间切 seen/unseen org，并在 train 上下文中剔除涉及 unseen org 的边，避免训练期泄漏。
- node multilabel 任务的监督来自 `node_multilabel.labels`，不是来自 `edge_label`；因此 node view 中 `supervised_old` 为空是有意设计。
- `main()` 没有调用 `scripts/datasets/build_task_views.py`；后者不要误认为本次 `patent_cn_edge_org_codeIII_*` 的生成入口。
