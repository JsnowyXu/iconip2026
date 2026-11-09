# Patent-IN 预处理代码逐行注释

本文只记录 `/home/xuxj/2026/dyn_relation_memory` 中 `patent_in` 预处理的源码位置和执行链路。未运行任何重型预处理、训练或分析任务；信息来自只读检索和源码阅读。

## 1. 原始源码位置

- 入口脚本：`/home/xuxj/2026/dyn_relation_memory/scripts/experiments/run_patent_preprocessing.py`
- patent-in 配置：`run_patent_preprocessing.py:41-51` 的 `DATASET_SPECS` 第 3 项
- 通用工具函数：`run_patent_preprocessing.py:55-196`
- baseline adapter 导出函数：`run_patent_preprocessing.py:199-238`
- 单个 task view 构造函数：`run_patent_preprocessing.py:343-408`
- patent task view 生成入口：`run_patent_preprocessing.py:411-518`
- unified 异质图构造核心函数：`run_patent_preprocessing.py:521-939`
- CLI 主入口与文件落盘：`run_patent_preprocessing.py:1024-1145`

## 2. 执行链路

历史 process log 中记录的命令为：

```bash
/home/xuxj/.conda/envs/pytorch/bin/python scripts/experiments/run_patent_preprocessing.py \
  --project-root /home/xuxj/2026/dyn_relation_memory \
  --dataset-root /home/xuxj/2026/datasets \
  --python-bin /home/xuxj/.conda/envs/pytorch/bin/python
```

执行链路：

1. `main()` 解析参数，建立 `data/processed`、`data/processed/tasks`、`data/interim/baseline_adapters`、`data/reports`。
2. 遍历 `DATASET_SPECS`；若 `--datasets patent_in` 被指定则只处理 `patent_in`，否则按 `patent_us`、`patent_cn`、`patent_in` 全部处理。
3. 对 `patent_in` 调用 `build_unified_dataset(spec, dataset_root, feature_dim)`。
4. `build_unified_dataset()` 读取 `/home/xuxj/2026/datasets/patent_in/all-in.csv`、`clsIII-reInd.csv`、`IPCSubClass_US.csv`，生成统一异质动态图对象 `unified`。
5. `main()` 将 `unified` 保存为 `data/processed/patent_in_unified.pt`。
6. `main()` 调用 `build_task_views(unified)`，生成：
   - `patent_in_edge_org_codeIII_transductive.pt`
   - `patent_in_edge_org_codeIII_inductive.pt`
   - `patent_in_node_patent_multilabel.pt`
7. 每个 task view 再通过 `export_baseline_bundle()` 导出 TGN 与 TGB2/STHN 风格 adapter。
8. `analyze_one()` 生成轻量统计与图，不属于模型训练。
9. `write_process_md()` 与 `write_summary_md()` 更新 process 文档和 `patent.md`；本次任务未改这些文件。

## 3. 配置片段：`DATASET_SPECS`

原始位置：`scripts/experiments/run_patent_preprocessing.py:18-52`

```python
18  DATASET_SPECS = [  # 三个专利数据集共用同一套预处理管线，靠 spec 区分文件名和字段候选。
19      {  # Patent-US 配置块。
20          "name": "patent_us",  # 数据集名；会决定输出文件前缀 patent_us。
21          "short": "us",  # 简写，仅作为数据集标识保留。
22          "csv": "all-us.csv",  # 主原始 CSV 文件名。
23          "ipc": "IPCSubClass_US.csv",  # IPC subclass 描述表。
24          "cls": "clsIII-reInd.csv",  # codeIIIId 到 reId/codeIII 的映射表。
25          "org_name_cols": ["orgname"],  # 组织名称字段候选。
26          "title_cols": ["title"],  # 专利标题字段候选。
27          "summary_cols": ["summary"],  # 专利摘要字段候选。
28          "date_cols": ["publicationDate", "date_time", "date_parsed"],  # 发布时间字段候选。
29      },
30      {  # Patent-CN 配置块。
31          "name": "patent_cn",  # 数据集名；会决定输出文件前缀 patent_cn。
32          "short": "cn",  # 简写。
33          "csv": "all-cn.csv",  # CN 主 CSV。
34          "ipc": "IPCSubClass.csv",  # CN 使用的 IPC subclass 表名不同。
35          "cls": "clsIII-reInd.csv",  # re-index 映射表。
36          "org_name_cols": ["applicantName", "orgname"],  # CN 可能用 applicantName 或 orgname。
37          "title_cols": ["title"],  # 标题字段候选。
38          "summary_cols": ["summary"],  # 摘要字段候选。
39          "date_cols": ["publicationDate"],  # CN 发布时间字段。
40      },
41      {  # Patent-IN 配置块；patent-in 的预处理入口就是这个 spec。
42          "name": "patent_in",  # 数据集名；输出统一图为 patent_in_unified.pt，task 前缀为 patent_in。
43          "short": "in",  # 简写。
44          "csv": "all-in.csv",  # Patent-IN 主表：/home/xuxj/2026/datasets/patent_in/all-in.csv。
45          "ipc": "IPCSubClass_US.csv",  # Patent-IN 复用 US 版 IPC subclass 描述表。
46          "cls": "clsIII-reInd.csv",  # codeIIIId/codeIIIIdreId/codeIII 映射表。
47          "org_name_cols": ["orgname"],  # Patent-IN 组织名称字段候选。
48          "title_cols": ["title"],  # 标题字段候选。
49          "summary_cols": ["summary"],  # 摘要字段候选。
50          "date_cols": ["publicationDate", "date_time", "date_parsed"],  # 优先 publicationDate，备用 date_time/date_parsed。
51      },
52  ]  # main() 会遍历这个列表，按 name 选择要处理的数据集。
```

## 4. 字段清洗与时间解析工具

原始位置：`scripts/experiments/run_patent_preprocessing.py:60-142`

```python
60  def choose_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:  # 从真实 CSV 列名中选择第一个存在的候选字段。
61      lowered = {c.lower(): c for c in columns}  # 建立小写列名到原始列名的映射，避免大小写差异。
62      for cand in candidates:  # 按配置中的优先顺序逐个尝试。
63          if cand.lower() in lowered:  # 如果候选字段存在。
64              return lowered[cand.lower()]  # 返回 CSV 中的真实字段名。
65      return None  # 所有候选都不存在时返回 None，调用侧会用空字符串或跳过。

80  def split_multi_value(value) -> List[str]:  # 解析 orgID/codeIIIId 这类多值字段。
81      if value is None:  # 缺失值直接视为空列表。
82          return []  # 没有可用 ID。
83      if isinstance(value, float) and math.isnan(value):  # pandas 读到的 NaN。
84          return []  # 缺失值。
85      if isinstance(value, (list, tuple, set)):  # 如果已经是集合/列表。
86          out = []  # 累积展开后的结果。
87          for item in value:  # 递归处理嵌套项。
88              out.extend(split_multi_value(item))  # 支持列表里还有分号字符串。
89          return out  # 返回展平后的多值列表。
91      text = str(value).strip()  # 统一转字符串并去掉首尾空白。
92      if not text or text.lower() in {"nan", "none", "null"}:  # 空值文本也跳过。
93          return []  # 没有有效 ID。
95      if text.startswith("[") and text.endswith("]"):  # 兼容 "['a','b']" 形式。
96          try:  # 尝试按 Python literal 解析。
97              parsed = ast.literal_eval(text)  # 安全解析列表文本。
98              return split_multi_value(parsed)  # 继续递归展开。
99          except Exception:  # 解析失败则回退到分隔符切分。
100             pass  # 保留原字符串继续处理。
102     text = text.replace("|", ";")  # 把竖线分隔统一成分号。
103     parts = [p.strip().strip("'").strip('"') for p in text.split(";")]  # 按分号切分并清理引号。
104     return [p for p in parts if p and p.lower() not in {"nan", "none", "null"}]  # 过滤空项和缺失占位符。

112 def parse_publication_ts(value) -> Tuple[float, str]:  # 把 publicationDate 等字段转为时间戳和规范日期字符串。
113     if value is None:  # 缺失时间。
114         return 0.0, ""  # 时间戳置 0，日期文本为空。
115     if isinstance(value, float) and math.isnan(value):  # NaN 时间。
116         return 0.0, ""  # 同样置空。
118     raw = str(value).strip()  # 转为字符串。
119     if not raw:  # 空字符串。
120         return 0.0, ""  # 无时间。
122     dt = None  # 初始化解析结果。
123     for candidate in [raw]:  # 当前只尝试原始字符串，保留可扩展结构。
124         try:  # 首轮严格解析。
125             dt = pd.to_datetime(candidate, errors="raise")  # pandas 解析日期。
126             break  # 成功即停止。
127         except Exception:  # 严格解析失败。
128             dt = None  # 回到未解析状态。
129     if dt is None and raw.isdigit() and len(raw) == 8:  # 兼容 YYYYMMDD 数字日期。
130         try:  # 尝试固定格式解析。
131             dt = datetime.strptime(raw, "%Y%m%d")  # 例如 20200131。
132         except Exception:  # 失败则继续 fallback。
133             dt = None  # 仍视作未解析。
134     if dt is None:  # 还没解析出来。
135         try:  # 使用宽松解析。
136             dt = pd.to_datetime(raw, errors="coerce")  # 失败会变 NaT。
137         except Exception:  # 极端异常。
138             dt = None  # 视为失败。
139     if dt is None or pd.isna(dt):  # 最终仍不可解析。
140         return 0.0, raw  # 保留原始文本，时间戳为 0。
141     py_dt = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt  # 转为 Python datetime。
142     return float(py_dt.timestamp()), py_dt.strftime("%Y-%m-%d")  # 返回秒级时间戳和 YYYY-MM-DD。
```

## 5. 核心函数：构造 Patent-IN unified 异质图

原始位置：`scripts/experiments/run_patent_preprocessing.py:521-939`

### 5.1 读取原始文件和映射表

```python
521 def build_unified_dataset(spec: Dict, dataset_root: Path, feature_dim: int) -> Tuple[Dict, Dict]:  # 根据 spec 构造统一动态图对象和统计计数。
522     ds_dir = dataset_root / spec["name"]  # patent_in 时目录为 /home/xuxj/2026/datasets/patent_in。
523     print(f"[prepare] start {spec['name']}", flush=True)  # 打印开始处理的数据集名。
524     df = pd.read_csv(ds_dir / spec["csv"])  # 读取 Patent-IN 主表 all-in.csv。
525     cls_df = pd.read_csv(ds_dir / spec["cls"])  # 读取 clsIII-reInd.csv，提供 codeIIIIdreId 到 codeIIIId/codeIII 的映射。
526     ipc_df = load_ipc_table(ds_dir / spec["ipc"])  # 读取 IPCSubClass_US.csv，补充 codeIII 文本描述。
528     title_col = choose_first(df.columns, spec["title_cols"])  # 定位标题列，Patent-IN 通常是 title。
529     summary_col = choose_first(df.columns, spec["summary_cols"])  # 定位摘要列，通常是 summary。
530     date_col = choose_first(df.columns, spec["date_cols"])  # 在 publicationDate/date_time/date_parsed 中选择可用时间列。
531     org_name_col = choose_first(df.columns, spec["org_name_cols"])  # 定位组织名称列 orgname。
533     ipc_desc_by_code: Dict[str, str] = {}  # codeIII 到 IPC 描述文本的字典。
534     if {"Class", "Name"}.issubset(ipc_df.columns):  # 标准两列表格式。
535         for _, row in ipc_df.iterrows():  # 遍历 IPC 表。
536             code = str(row["Class"]).strip()  # 取 IPC subclass code。
537             desc = str(row["Name"]).strip()  # 取 IPC 描述。
538             ipc_desc_by_code[code] = desc  # 建立 code -> desc。
539     else:  # 兼容只有一列、每行包含 code 和描述的格式。
540         col = ipc_df.columns[0]  # 使用第一列。
541         for _, row in ipc_df.iterrows():  # 遍历每一行。
542             text = str(row[col]).strip()  # 原始文本。
543             code = text.split()[0].strip().strip('"')  # 第一个 token 当作 code。
544             desc = text[len(code):].strip().strip('"')  # 剩余部分当作描述。
545             ipc_desc_by_code[code] = desc or text  # 若描述为空则保留整行文本。
547     code_meta_by_reid: Dict[str, Dict] = {}  # re-index 后的 codeIIIIdreId 到完整 IPC 元信息。
548     for _, row in cls_df.iterrows():  # 遍历 code 映射表。
549         codeiiiid = str(row["codeIIIId"]).strip()  # 原始 codeIIIId，用作多标签类别 ID。
550         codeiii = str(row["codeIII"]).strip()  # IPC subclass 字符串，如 A01B。
551         reid = str(row["codeIIIIdreId"]).strip()  # 重新编号后的 code ID，主表 codeIIIId 会用这个值。
552         desc = ipc_desc_by_code.get(codeiii, "")  # 找 IPC 描述。
553         code_meta_by_reid[reid] = {  # 后续只接受存在于该映射中的 code。
554             "codeIIIId": codeiiiid,  # 原始类别 ID。
555             "codeIIIIdreId": reid,  # 重编号 ID。
556             "codeIII": codeiii,  # IPC subclass 名称。
557             "description": desc,  # IPC 描述文本。
558         }
```

### 5.2 节点注册器

```python
560     node_key_to_id: Dict[str, int] = {}  # 全局节点 key 到连续 node id。
561     node_type: List[int] = []  # 每个节点的类型：0 patent，1 org，2 codeIII。
562     node_text: List[str] = []  # 用于 TF-IDF 的节点文本。
563     node_name: List[str] = []  # 可读节点名。
564     patent_nodes: List[int] = []  # patent 类型节点 id 列表。
565     patent_raw_ids: List[str] = []  # 原始 patentId。
566     patent_titles: List[str] = []  # 专利标题。
567     patent_summaries: List[str] = []  # 专利摘要。
568     patent_dates: List[str] = []  # 规范化发布日期。
569     patent_ts: List[float] = []  # 发布日期时间戳。
570     patent_code_ids: Dict[int, List[str]] = {}  # patent node -> codeIIIIdreId 列表。
571     patent_label_ids: Dict[int, List[str]] = {}  # patent node -> 原始 codeIIIId 标签列表。
572     patent_node_to_idx: Dict[int, int] = {}  # patent node id 到 patent_nodes 内部行号。
573     org_nodes: List[int] = []  # org 节点 id。
574     org_raw_ids: List[str] = []  # 原始 orgID。
575     org_names: List[str] = []  # 组织名称。
576     code_nodes: List[int] = []  # codeIII 节点 id。
577     code_raw_ids: List[str] = []  # 原始 codeIIIId。
578     code_reindexed_ids: List[str] = []  # codeIIIIdreId。
579     code_names: List[str] = []  # codeIII 名称。
580     code_descs: List[str] = []  # codeIII 描述。
582     def get_node(key: str, ntype: int, text: str, display_name: str, registry: str, raw_id: str, extra_text: str = "") -> int:  # 创建或复用节点。
583         if key in node_key_to_id:  # 节点已存在。
584             return node_key_to_id[key]  # 直接返回已有 node id。
585         nid = len(node_type)  # 新节点 id 等于当前节点数。
586         node_key_to_id[key] = nid  # 注册 key。
587         node_type.append(ntype)  # 写入节点类型。
588         full_text = " ".join([t for t in [text, extra_text] if t]).strip()  # 合并主文本和补充文本。
589         node_text.append(full_text)  # 保存用于特征抽取的文本。
590         node_name.append(display_name)  # 保存显示名。
591         if registry == "patent":  # patent 节点分支。
592             patent_nodes.append(nid)  # 记录 patent node id。
593             patent_node_to_idx[nid] = len(patent_nodes) - 1  # 记录在 patent 属性表中的行号。
594             patent_raw_ids.append(raw_id)  # 保存 patentId。
595             patent_titles.append(text)  # 保存标题。
596             patent_summaries.append(extra_text)  # 保存摘要。
597             patent_dates.append("")  # 日期先占位，读到行时间后再回填。
598             patent_ts.append(0.0)  # 时间戳先占位。
599             patent_code_ids[nid] = []  # 初始化该专利连接的 code 列表。
600             patent_label_ids[nid] = []  # 初始化该专利的多标签类别列表。
601         elif registry == "org":  # org 节点分支。
602             org_nodes.append(nid)  # 记录 org node id。
603             org_raw_ids.append(raw_id)  # 保存 orgID。
604             org_names.append(display_name)  # 保存组织名称。
605         elif registry == "code":  # codeIII 节点分支。
606             code_nodes.append(nid)  # 记录 code node id。
607             code_raw_ids.append(raw_id)  # 保存原始 codeIIIId。
608             code_reindexed_ids.append(key.split(":", 1)[1])  # 从 code:{reid} 中取 reId。
609             code_names.append(display_name)  # 保存 codeIII。
610             code_descs.append(extra_text)  # 保存 IPC 描述。
611         return nid  # 返回新建 node id。
```

### 5.3 异质边构造

```python
613     rel_map = {"patent_org": 0, "patent_codeIII": 1, "org_codeIII": 2, "org_coorg": 3}  # 四种关系的全局 ID。
614     src_list: List[int] = []  # 边源节点。
615     dst_list: List[int] = []  # 边目标节点。
616     ts_list: List[float] = []  # 边时间，全部来自专利发布时间。
617     rel_list: List[int] = []  # 边关系类型。
619     row_stats = Counter()  # 行级统计。
620     org_pair_counter = Counter()  # co-assignee 组织对去重统计。
621     org_code_counter = Counter()  # org-codeIII 唯一对统计。
622     for _, row in df.iterrows():  # 遍历 all-in.csv 每一条专利记录。
623         patent_raw = str(row["patentId"]).strip()  # 原始 patentId。
624         patent_title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""  # 标题。
625         patent_summary = str(row[summary_col]).strip() if summary_col and pd.notna(row[summary_col]) else ""  # 摘要。
626         patent_ts_val, patent_date_text = parse_publication_ts(row[date_col] if date_col else "")  # 发布时间。
627         org_ids = split_multi_value(row["orgID"])  # 拆分多 assignee/orgID。
628         code_ids = split_multi_value(row["codeIIIId"])  # 拆分多 IPC/codeIIIIdreId。
629         org_names_row = split_multi_value(row[org_name_col]) if org_name_col else []  # 拆分组织名。
630         class_codes_row = split_multi_value(row["classCode"]) if "classCode" in df.columns else []  # 仅用于统计非空。
632         org_ids = [x for x in org_ids if x]  # 清理空 org。
633         code_ids = [x for x in code_ids if x in code_meta_by_reid]  # 只保留映射表中存在的 code。
634         if not org_ids or not code_ids:  # 缺 org 或缺 code 的专利无法形成目标边。
635             row_stats["rows_skipped_missing_org_or_code"] += 1  # 计入跳过。
636             continue  # 跳过该行。
638         patent_node = get_node(  # 注册 patent 节点。
639             key=f"patent:{patent_raw}",  # patent 节点唯一 key。
640             ntype=0,  # patent 类型 ID。
641             text=patent_title,  # 标题作为主文本。
642             display_name=f"patent:{patent_raw}",  # 显示名。
643             registry="patent",  # 写入 patent 属性表。
644             raw_id=patent_raw,  # 原始 ID。
645             extra_text=patent_summary,  # 摘要参与文本特征。
646         )
647         pat_idx = patent_node_to_idx[patent_node]  # 找到 patent 属性行。
648         patent_dates[pat_idx] = patent_date_text  # 回填发布日期文本。
649         patent_ts[pat_idx] = patent_ts_val  # 回填时间戳。
651         org_node_ids = []  # 当前专利关联的 org 节点。
652         for i, org_raw in enumerate(dict.fromkeys(org_ids)):  # 去重并保留原始顺序。
653             org_name = normalize_org_name(org_names_row[i]) if i < len(org_names_row) else f"org:{org_raw}"  # 对齐 orgname。
654             org_node = get_node(  # 注册 org 节点。
655                 key=f"org:{org_raw}",  # org 唯一 key。
656                 ntype=1,  # org 类型 ID。
657                 text=org_name,  # org 名称作为文本。
658                 display_name=org_name,  # 显示名。
659                 registry="org",  # 写入 org 属性表。
660                 raw_id=str(org_raw),  # 原始 orgID。
661             )
662             org_node_ids.append(org_node)  # 记录当前行的 org node。
664         code_node_ids = []  # 当前专利关联的 codeIII 节点。
665         for code_reid in dict.fromkeys(code_ids):  # 去重并保留顺序。
666             meta = code_meta_by_reid[code_reid]  # 取 code 元信息。
667             code_name = meta["codeIII"]  # IPC subclass 名称。
668             code_desc = meta["description"]  # IPC 描述。
669             code_node = get_node(  # 注册 codeIII 节点。
670                 key=f"code:{code_reid}",  # 使用 reId 作为节点 key。
671                 ntype=2,  # codeIII 类型 ID。
672                 text=code_name,  # code 名作为主文本。
673                 display_name=code_name,  # 显示名。
674                 registry="code",  # 写入 code 属性表。
675                 raw_id=str(meta["codeIIIId"]),  # 原始 codeIIIId。
676                 extra_text=code_desc,  # IPC 描述参与文本特征。
677             )
678             code_node_ids.append(code_node)  # 记录当前行 code node。
679             patent_code_ids[patent_node].append(code_reid)  # 保存 reId 连接信息。
680             patent_label_ids[patent_node].append(meta["codeIIIId"])  # 保存多标签类别 ID。
682         patent_code_ids[patent_node] = sorted(set(patent_code_ids[patent_node]))  # 去重 code reId。
683         patent_label_ids[patent_node] = sorted(set(patent_label_ids[patent_node]), key=lambda x: int(x))  # 去重并按数值排序标签。
685         for org_node in org_node_ids:  # 对每个 assignee。
686             src_list.append(patent_node)  # patent -> org。
687             dst_list.append(org_node)  # 目标 org。
688             ts_list.append(patent_ts_val)  # 使用专利发布时间。
689             rel_list.append(rel_map["patent_org"])  # 关系类型 patent_org。
690         row_stats["patent_org_edges"] += len(org_node_ids)  # 统计 patent-org 边数。
692         for code_node in code_node_ids:  # 对每个 IPC code。
693             src_list.append(patent_node)  # patent -> codeIII。
694             dst_list.append(code_node)  # 目标 codeIII。
695             ts_list.append(patent_ts_val)  # 使用专利发布时间。
696             rel_list.append(rel_map["patent_codeIII"])  # 关系类型 patent_codeIII。
697         row_stats["patent_codeIII_edges"] += len(code_node_ids)  # 统计 patent-code 边数。
699         for org_node in org_node_ids:  # 对每个 org。
700             for code_node in code_node_ids:  # 与每个 code 组合。
701                 src_list.append(org_node)  # org -> codeIII 是边预测主任务关系。
702                 dst_list.append(code_node)  # 目标 IPC code。
703                 ts_list.append(patent_ts_val)  # 时间仍是专利发布时间。
704                 rel_list.append(rel_map["org_codeIII"])  # 关系类型 org_codeIII。
705                 org_code_counter[(org_node, code_node)] += 1  # 统计唯一 org-code 对。
706         row_stats["org_codeIII_edges"] += len(org_node_ids) * len(code_node_ids)  # 统计主任务边数。
708         for i in range(len(org_node_ids)):  # 多 assignee 专利产生共申请关系。
709             for j in range(i + 1, len(org_node_ids)):  # 枚举无向 pair 的一半。
710                 a, b = org_node_ids[i], org_node_ids[j]  # 取两个 org。
711                 src_list.extend([a, b])  # 加双向边 a->b 和 b->a。
712                 dst_list.extend([b, a])  # 双向目标。
713                 ts_list.extend([patent_ts_val, patent_ts_val])  # 两条边同一时间。
714                 rel_list.extend([rel_map["org_coorg"], rel_map["org_coorg"]])  # 关系类型 org_coorg。
715                 org_pair_counter[(a, b)] += 1  # 唯一 org pair 计数。
716         row_stats["org_coorg_edges"] += max(0, len(org_node_ids) * (len(org_node_ids) - 1))  # 双向边数量。
717         row_stats["rows_processed"] += 1  # 当前专利成功处理。
```

### 5.4 特征、标签和 unified 对象

```python
720     src = np.asarray(src_list, dtype=np.int64)  # 边源列表转 numpy。
721     dst = np.asarray(dst_list, dtype=np.int64)  # 边目标列表转 numpy。
722     ts = np.asarray(ts_list, dtype=np.float32)  # 时间列表转 numpy。
723     rel = np.asarray(rel_list, dtype=np.int64)  # 关系类型列表转 numpy。
724     node_type_arr = np.asarray(node_type, dtype=np.int64)  # 节点类型转 numpy。
726     if ts.size == 0:  # 如果没有任何边。
727         raise RuntimeError(f"{spec['name']} produced no edges")  # 直接报错，避免保存空图。
729     order = np.lexsort((np.arange(ts.size), ts))  # 按时间排序，时间相同按原始边序稳定排序。
730     src = src[order]  # 重排源节点。
731     dst = dst[order]  # 重排目标节点。
732     ts = ts[order]  # 重排时间。
733     rel = rel[order]  # 重排关系类型。
735     in_deg = np.bincount(dst, minlength=len(node_type))  # 节点入度。
736     out_deg = np.bincount(src, minlength=len(node_type))  # 节点出度。
737     deg = in_deg + out_deg  # 总度。
738     first_ts = np.full(len(node_type), ts.max(), dtype=np.float32)  # 每个节点首次出现时间初始化为最大时间。
739     last_ts = np.zeros(len(node_type), dtype=np.float32)  # 每个节点最后出现时间初始化为 0。
740     for s, d, t in zip(src, dst, ts):  # 遍历排序后的边。
741         first_ts[s] = min(first_ts[s], t)  # 更新源节点首次时间。
742         first_ts[d] = min(first_ts[d], t)  # 更新目标节点首次时间。
743         last_ts[s] = max(last_ts[s], t)  # 更新源节点最后时间。
744         last_ts[d] = max(last_ts[d], t)  # 更新目标节点最后时间。
745     ts_min, ts_max = float(ts.min()), float(ts.max())  # 全图时间范围。
746     ts_span = max(ts_max - ts_min, 1.0)  # 防止除 0。
747     first_ts_norm = ((first_ts - ts_min) / ts_span).astype(np.float32)  # 归一化首次时间。
748     last_ts_norm = ((last_ts - ts_min) / ts_span).astype(np.float32)  # 归一化最后时间。
750     texts = [t if t else n for t, n in zip(node_text, node_name)]  # 节点文本为空时使用节点名兜底。
751     text_dim = max(8, feature_dim - 8)  # 总 feature_dim 中预留 8 维结构特征。
752     vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=text_dim, lowercase=False)  # 字符 ngram TF-IDF。
753     text_feat = vectorizer.fit_transform(texts).toarray().astype(np.float32)  # 得到节点文本特征。
754     if text_feat.shape[1] < text_dim:  # 实际词表维度不足。
755         pad = np.zeros((text_feat.shape[0], text_dim - text_feat.shape[1]), dtype=np.float32)  # 右侧补零。
756         text_feat = np.concatenate([text_feat, pad], axis=1)  # 对齐固定维度。
758     struct_feat = np.stack(  # 8 维结构/类型/时间特征。
760             np.log1p(in_deg) / max(np.log1p(in_deg).max(), 1.0),  # 归一化 log 入度。
761             np.log1p(out_deg) / max(np.log1p(out_deg).max(), 1.0),  # 归一化 log 出度。
762             np.log1p(deg) / max(np.log1p(deg).max(), 1.0),  # 归一化 log 总度。
763             first_ts_norm,  # 首次出现时间。
764             last_ts_norm,  # 最后出现时间。
765             (node_type_arr == 0).astype(np.float32),  # patent 类型 one-hot。
766             (node_type_arr == 1).astype(np.float32),  # org 类型 one-hot。
767             (node_type_arr == 2).astype(np.float32),  # codeIII 类型 one-hot。
771     node_feat = np.concatenate([text_feat, struct_feat], axis=1).astype(np.float32)  # 拼成最终节点特征。
773     patent_node_arr = np.asarray(patent_nodes, dtype=np.int64)  # patent 节点 ID 数组。
774     label_id_order = sorted({cid for labels in patent_label_ids.values() for cid in labels}, key=lambda x: int(x))  # 全部 codeIIIId 标签集合。
781     label_id_to_col = {cid: i for i, cid in enumerate(label_id_order)}  # 标签 ID 到矩阵列号。
782     label_matrix = np.zeros((len(patent_nodes), len(label_id_order)), dtype=np.float32)  # patent 多标签矩阵。
783     for i, pnode in enumerate(patent_nodes):  # 遍历 patent 节点。
784         labels = [label_id_to_col[cid] for cid in patent_label_ids.get(pnode, []) if cid in label_id_to_col]  # 找该 patent 的标签列。
785         if labels:  # 如果有标签。
786             label_matrix[i, labels] = 1.0  # 置为多热编码。
788     rel_norm = rel.astype(np.float32) / max(int(rel.max()), 1)  # 关系 ID 归一化。
789     ts_norm = ((ts - ts_min) / ts_span).astype(np.float32)  # 边时间归一化。
790     edge_feat = np.stack(  # 构造 4 维边特征。
792             ts_norm,  # 归一化时间。
793             rel_norm,  # 归一化关系 ID。
794             node_type_arr[src].astype(np.float32) / 2.0,  # 源节点类型归一化。
795             node_type_arr[dst].astype(np.float32) / 2.0,  # 目标节点类型归一化。
798     ).astype(np.float32)  # 转 float32。
836     unified = {  # 最终统一数据对象。
837         "num_nodes": int(len(node_type)),  # 节点总数。
838         "num_rels": int(len(rel_map)),  # 关系类型数，固定 4。
839         "num_node_types": 3,  # 节点类型数，patent/org/codeIII。
840         "node_feat": torch.from_numpy(node_feat.astype(np.float32)),  # 节点特征 tensor。
841         "node_type": torch.from_numpy(node_type_arr.astype(np.int64)),  # 节点类型 tensor。
842         "edges": {  # 边表。
843             "src": torch.from_numpy(src.astype(np.int64)),  # 源节点。
844             "dst": torch.from_numpy(dst.astype(np.int64)),  # 目标节点。
845             "ts": torch.from_numpy(ts.astype(np.float32)),  # 时间戳。
846             "rel": torch.from_numpy(rel.astype(np.int64)),  # 关系 ID。
847             "edge_feat": torch.from_numpy(edge_feat.astype(np.float32)),  # 边特征。
848             "edge_label": torch.ones(src.shape[0], dtype=torch.float32),  # unified 中所有边默认为正例。
850         "split": {  # unified 默认 split，按 transductive 时间窗切全图上下文。
851             "train": torch.from_numpy(base_tr),  # train 边索引。
852             "val": torch.from_numpy(base_va),  # val 边索引。
853             "test": torch.from_numpy(base_te),  # test 边索引。
856         "node_multilabel": {  # patent 多标签任务数据。
857             "node_ids": torch.from_numpy(patent_node_arr.astype(np.int64)),  # patent 节点 ID。
858             "labels": torch.from_numpy(label_matrix.astype(np.float32)),  # 多标签矩阵。
859             "label_names": [  # 标签元信息。
861                     "codeIIIId": cid,  # 标签原始 ID。
862                     "codeIII": label_meta_by_id[cid]["codeIII"],  # 标签 IPC 名。
863                     "description": label_meta_by_id[cid]["description"],  # 标签描述。
868         "meta": {  # 元数据。
869             "dataset_name": spec["name"],  # patent_in。
871                 "relation_map": rel_map,  # 关系名到 ID。
872                 "node_type_map": {"patent": 0, "org": 1, "codeIII": 2},  # 节点类型映射。
881                 "patent_attributes": {  # patent 属性表。
882                     "node_ids": patent_nodes,  # patent 节点。
883                     "patent_id": patent_raw_ids,  # 原始 patentId。
884                     "title": patent_titles,  # 标题。
885                     "summary": patent_summaries,  # 摘要。
886                     "publication_date": patent_dates,  # 日期文本。
887                     "publication_ts": patent_ts,  # 时间戳。
889                 "org_attributes": {  # org 属性表。
890                     "node_ids": org_nodes,  # org 节点。
891                     "org_id": org_raw_ids,  # 原始 orgID。
892                     "name": org_names,  # org 名称。
894                 "codeIII_attributes": {  # codeIII 属性表。
895                     "node_ids": code_nodes,  # code 节点。
896                     "codeIIIId": code_raw_ids,  # 原始 codeIIIId。
897                     "codeIIIIdreId": code_reindexed_ids,  # reId。
898                     "codeIII": code_names,  # IPC 名称。
899                     "description": code_descs,  # IPC 描述。
901                 "task_splits": {  # 记录 edge task 的原始监督边 split。
902                     "org_codeIII_transductive": {  # transductive 监督边 split。
903                         "edge_indices_train": trans_tr.tolist(),  # 训练监督边。
904                         "edge_indices_val": trans_va.tolist(),  # 验证监督边。
905                         "edge_indices_test": trans_te.tolist(),  # 测试监督边。
907                     "org_codeIII_inductive": {  # inductive 监督边 split。
908                         "edge_indices_train": ind_train,  # seen org 训练边。
909                         "edge_indices_val": ind_val,  # unseen/val org 验证边。
910                         "edge_indices_test": ind_test,  # unseen/test org 测试边。
911                         "unseen_org_node_ids": np.concatenate([val_orgs, test_orgs], axis=0).astype(np.int64).tolist(),  # inductive 未见 org。
919     counts = {  # 返回给 process md 的统计。
920         "rows_total": int(len(df)),  # 原始行数。
921         "rows_processed": int(row_stats["rows_processed"]),  # 成功构图行数。
922         "rows_skipped_missing_org_or_code": int(row_stats["rows_skipped_missing_org_or_code"]),  # 跳过行数。
923         "num_patent_nodes": int(len(patent_nodes)),  # patent 节点数。
924         "num_org_nodes": int(len(org_nodes)),  # org 节点数。
925         "num_codeIII_nodes": int(len(code_nodes)),  # codeIII 节点数。
926         "num_edges_patent_org": int((rel == rel_map["patent_org"]).sum()),  # patent_org 边数。
927         "num_edges_patent_codeIII": int((rel == rel_map["patent_codeIII"]).sum()),  # patent_codeIII 边数。
928         "num_edges_org_codeIII": int((rel == rel_map["org_codeIII"]).sum()),  # 主任务 org_codeIII 边数。
929         "num_edges_org_coorg": int((rel == rel_map["org_coorg"]).sum()),  # co-assignee 边数。
930         "num_unique_org_code_pairs": int(len(org_code_counter)),  # 唯一 org-code 对数。
931         "num_unique_org_org_pairs": int(len(org_pair_counter)),  # 唯一 org-org 对数。
932         "num_multilabel_classes": int(len(label_id_order)),  # 多标签类别数。
933         "num_labeled_patents": int(label_matrix.shape[0]),  # 有标签矩阵行的 patent 数。
939     return unified, counts  # 返回统一图和统计。
```

## 6. 后续 task view 生成入口

### 6.1 通用 task view 裁剪函数

原始位置：`scripts/experiments/run_patent_preprocessing.py:343-408`

```python
343 def build_task_view(  # 根据原始 unified 图和指定旧边索引生成一个 task-specific view。
344     obj: Dict,  # 输入 unified 数据对象。
345     out_name: str,  # 输出 view 的 dataset_name。
346     train_old: np.ndarray,  # 原 unified 中属于训练窗口的边索引。
347     val_old: np.ndarray,  # 原 unified 中属于验证窗口的边索引。
348     test_old: np.ndarray,  # 原 unified 中属于测试窗口的边索引。
349     supervised_old: np.ndarray,  # 原 unified 中真正参与监督的边索引。
350     extra_info: Dict,  # 写入 meta.extra 的任务说明。
351 ) -> Dict:  # 返回新 view。
352     src_all = obj["edges"]["src"].numpy().astype(np.int64)  # 取全图源节点。
353     dst_all = obj["edges"]["dst"].numpy().astype(np.int64)  # 取全图目标节点。
354     ts_all = obj["edges"]["ts"].numpy().astype(np.float32)  # 取全图时间戳。
355     rel_all = obj["edges"]["rel"].numpy().astype(np.int64)  # 取全图关系 ID。
356     edge_feat_all = obj["edges"]["edge_feat"].numpy().astype(np.float32)  # 取全图边特征。
358     selected_old = np.concatenate([train_old, val_old, test_old], axis=0).astype(np.int64)  # 合并上下文窗口边。
359     selected_old = sort_old_indices(selected_old, ts_all)  # 按时间重排并去重。
361     old_to_new = np.full((ts_all.shape[0],), -1, dtype=np.int64)  # 旧边索引到新 view 边索引的映射。
362     old_to_new[selected_old] = np.arange(selected_old.shape[0], dtype=np.int64)  # 被选中的边获得新编号。
364     train_new = map_old_to_new(train_old, old_to_new)  # train split 转为 view 内部索引。
365     val_new = map_old_to_new(val_old, old_to_new)  # val split 转为 view 内部索引。
366     test_new = map_old_to_new(test_old, old_to_new)  # test split 转为 view 内部索引。
367     edge_label = np.isin(selected_old, supervised_old).astype(np.float32)  # 只有 supervised_old 边标为 1，其它上下文边为 0。
369     meta = dict(obj.get("meta", {}))  # 复制 meta。
370     extra = dict(meta.get("extra", {}))  # 复制 meta.extra。
371     extra["task_view_name"] = out_name  # 记录 view 名称。
372     extra["context_protocol"] = {  # 记录上下文协议。
373         "enabled": True,  # 表示这是保留上下文边的 view。
374         "supervision_mode": "edge_label_mask",  # 用 edge_label 区分监督边和上下文边。
375         "preserve_global_rel_ids": True,  # 关系 ID 不重映射，仍是 patent_org/patent_codeIII/org_codeIII/org_coorg。
376         "selected_edges": int(selected_old.size),  # view 中边数。
377         "supervised_edges": int(edge_label.sum()),  # 监督边数。
378         "train_window_edges": int(train_new.size),  # train 窗口边数。
379         "val_window_edges": int(val_new.size),  # val 窗口边数。
380         "test_window_edges": int(test_new.size),  # test 窗口边数。
381     }
382     extra.update(extra_info)  # 写入具体任务信息。
383     meta["dataset_name"] = out_name  # view 的 dataset_name 改为 task 名。
384     meta["extra"] = extra  # 回写 meta.extra。
386     return {  # 返回 task view 对象。
387         "num_nodes": int(obj["num_nodes"]),  # 节点全集不裁剪。
388         "num_rels": int(obj["num_rels"]),  # 关系全集不裁剪。
389         "num_node_types": int(obj["num_node_types"]),  # 节点类型全集不裁剪。
390         "node_feat": obj["node_feat"],  # 节点特征复用 unified。
391         "node_type": obj["node_type"],  # 节点类型复用 unified。
392         "edges": {  # 仅边表按 selected_old 裁剪。
393             "src": torch.from_numpy(src_all[selected_old].astype(np.int64)),  # view 源节点。
394             "dst": torch.from_numpy(dst_all[selected_old].astype(np.int64)),  # view 目标节点。
395             "ts": torch.from_numpy(ts_all[selected_old].astype(np.float32)),  # view 时间戳。
396             "rel": torch.from_numpy(rel_all[selected_old].astype(np.int64)),  # view 关系。
397             "edge_feat": torch.from_numpy(edge_feat_all[selected_old].astype(np.float32)),  # view 边特征。
398             "edge_label": torch.from_numpy(edge_label.astype(np.float32)),  # 监督 mask。
400         "split": {  # view 内部 split。
401             "train": torch.from_numpy(train_new),  # train 新索引。
402             "val": torch.from_numpy(val_new),  # val 新索引。
403             "test": torch.from_numpy(test_new),  # test 新索引。
405         "node_multiclass": obj.get("node_multiclass"),  # 透传 multiclass 数据，当前为 None。
406         "node_multilabel": obj.get("node_multilabel"),  # 透传 patent 多标签数据。
407         "meta": meta,  # 写入更新后的 meta。
408     }
```

### 6.2 Patent 专用 task view 生成入口

原始位置：`scripts/experiments/run_patent_preprocessing.py:411-518`

```python
411 def build_task_views(unified: Dict) -> Dict[str, Dict]:  # Patent 数据集的 task view 生成总入口。
412     rel_map = unified["meta"]["extra"]["relation_map"]  # 取关系映射。
413     target_rel = int(rel_map["org_codeIII"])  # 目标预测关系固定为 org -> codeIII。
414     src = unified["edges"]["src"].numpy().astype(np.int64)  # 源节点数组。
415     dst = unified["edges"]["dst"].numpy().astype(np.int64)  # 目标节点数组。
416     ts = unified["edges"]["ts"].numpy().astype(np.float32)  # 时间数组。
417     rel = unified["edges"]["rel"].numpy().astype(np.int64)  # 关系数组。
418     n_edges = rel.shape[0]  # 全图边数。
420     all_sorted = sort_old_indices(np.arange(n_edges, dtype=np.int64), ts)  # 所有边按时间排序。
421     target_old = sort_old_indices(np.where(rel == target_rel)[0], ts)  # 只取 org_codeIII 监督候选边。
422     tr_tgt, va_tgt, te_tgt = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # 目标边 70/15/15 时间切分。
424     train_cut = float(ts[tr_tgt[-1]]) if tr_tgt.size > 0 else float(ts[all_sorted[-1]])  # train 截止时间。
425     val_cut = float(ts[va_tgt[-1]]) if va_tgt.size > 0 else train_cut  # val 截止时间。
427     tr_old = all_sorted[ts[all_sorted] <= train_cut]  # transductive train 上下文：截止 train 的所有关系边。
428     va_old = all_sorted[(ts[all_sorted] > train_cut) & (ts[all_sorted] <= val_cut)]  # transductive val 上下文。
429     te_old = all_sorted[ts[all_sorted] > val_cut]  # transductive test 上下文。
430     trans_view = build_task_view(  # 构建 transductive 边预测 view。
432         out_name=f"{unified['meta']['dataset_name']}_edge_org_codeIII_transductive",  # patent_in_edge_org_codeIII_transductive。
433         train_old=tr_old,  # train 上下文边。
434         val_old=va_old,  # val 上下文边。
435         test_old=te_old,  # test 上下文边。
436         supervised_old=target_old,  # 只有 org_codeIII 边是监督边。
438             "target_original_rel_id": target_rel,  # 目标关系 ID。
439             "target_relation_name": "org_codeIII",  # 目标关系名。
440             "protocol": "full_hetero_context_transductive",  # 保留全异质上下文的 transductive 协议。
444     org_first_ts = {}  # 记录每个 org 首次出现在目标关系中的时间。
445     org_nodes = src[target_old]  # org_codeIII 边的源节点就是 org。
446     for edge_id, org_node in zip(target_old.tolist(), org_nodes.tolist()):  # 遍历目标边。
447         org_first_ts.setdefault(int(org_node), float(ts[int(edge_id)]))  # 保留首次出现时间。
448     org_sorted = sorted(org_first_ts.items(), key=lambda x: (x[1], x[0]))  # 按首次时间排序 org。
449     org_ids = np.array([k for k, _ in org_sorted], dtype=np.int64)  # 得到 org 序列。
450     if org_ids.size >= 3:  # 至少 3 个 org 才能做 inductive 组织级切分。
451         n_org = org_ids.size  # org 数。
452         n_train_org = max(1, int(n_org * 0.70))  # 70% org 作为 seen/train。
453         n_val_org = max(1, int(n_org * 0.15))  # 15% org 作为 val unseen。
454         if n_train_org + n_val_org >= n_org:  # 防止 test 没有 org。
455             n_train_org = max(1, n_org - 2)  # 至少留两个非训练 org。
456             n_val_org = max(1, n_org - n_train_org - 1)  # 至少留一个 val。
457         train_orgs = org_ids[:n_train_org]  # seen org。
458         val_orgs = org_ids[n_train_org : n_train_org + n_val_org]  # val unseen org。
459         test_orgs = org_ids[n_train_org + n_val_org :]  # test unseen org。
468     target_src = src[target_old]  # 目标边源节点。
469     ind_train_target = target_old[np.isin(target_src, train_orgs)]  # seen org 的监督边为 train。
470     ind_val_target = target_old[np.isin(target_src, val_orgs)]  # val unseen org 的监督边。
471     ind_test_target = target_old[np.isin(target_src, test_orgs)]  # test unseen org 的监督边。
472     if ind_val_target.size == 0 or ind_test_target.size == 0:  # 若 org 切分导致 val/test 为空。
473         ind_train_target, ind_val_target, ind_test_target = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # 回退为时间切分。
474         unseen_orgs = np.array([], dtype=np.int64)  # 回退时不记录 unseen org。
476         unseen_orgs = np.concatenate([val_orgs, test_orgs], axis=0).astype(np.int64)  # 正常 inductive 下记录未见 org。
478     train_cut_ind = float(ts[ind_train_target[-1]]) if ind_train_target.size > 0 else float(ts[all_sorted[-1]])  # inductive train 时间截止。
479     val_cut_ind = float(ts[ind_val_target[-1]]) if ind_val_target.size > 0 else train_cut_ind  # inductive val 时间截止。
480     involve_unseen = np.isin(src, unseen_orgs) | np.isin(dst, unseen_orgs) if unseen_orgs.size > 0 else np.zeros(n_edges, dtype=bool)  # 标记涉及 unseen org 的上下文边。
481     ind_tr_old = all_sorted[(ts[all_sorted] <= train_cut_ind) & (~involve_unseen[all_sorted])]  # 训练上下文排除 unseen org。
482     ind_va_old = all_sorted[(ts[all_sorted] > train_cut_ind) & (ts[all_sorted] <= val_cut_ind)]  # inductive val 上下文。
483     ind_te_old = all_sorted[ts[all_sorted] > val_cut_ind]  # inductive test 上下文。
490     ind_view = build_task_view(  # 构建 inductive 边预测 view。
492         out_name=f"{unified['meta']['dataset_name']}_edge_org_codeIII_inductive",  # patent_in_edge_org_codeIII_inductive。
493         train_old=ind_tr_old,  # 训练上下文边。
494         val_old=ind_va_old,  # 验证上下文边。
495         test_old=ind_te_old,  # 测试上下文边。
496         supervised_old=np.concatenate([ind_train_target, ind_val_target, ind_test_target], axis=0),  # 三段目标边均作为监督 mask。
498             "target_original_rel_id": target_rel,  # 目标关系 ID。
499             "target_relation_name": "org_codeIII",  # 目标关系名。
500             "protocol": "full_hetero_context_inductive",  # 保留全异质上下文的 inductive 协议。
501             "unseen_org_node_ids": unseen_orgs.tolist(),  # 未见 org 列表。
505     node_view = build_task_view(  # 构建 patent 多标签节点任务 view。
507         out_name=f"{unified['meta']['dataset_name']}_node_patent_multilabel",  # patent_in_node_patent_multilabel。
508         train_old=tr_old,  # 沿用 transductive 时间窗口。
509         val_old=va_old,  # 沿用 transductive val。
510         test_old=te_old,  # 沿用 transductive test。
511         supervised_old=np.array([], dtype=np.int64),  # 节点任务不靠 edge_label 监督，故监督边为空。
512         extra_info={"node_task": "patent_multilabel_codeIII", "node_task_view": True},  # 标记节点多标签任务。
514     return {  # 返回三个 task view。
515         "edge_org_codeIII_transductive": trans_view,  # 对应输出 patent_in_edge_org_codeIII_transductive.pt。
516         "edge_org_codeIII_inductive": ind_view,  # 对应输出 patent_in_edge_org_codeIII_inductive.pt。
517         "node_patent_multilabel": node_view,  # 对应输出 patent_in_node_patent_multilabel.pt。
518     }
```

## 7. CLI 主入口、落盘和 adapter 生成

原始位置：`scripts/experiments/run_patent_preprocessing.py:1024-1145`

```python
1024 def main() -> None:  # 脚本主入口。
1025     ap = argparse.ArgumentParser()  # 创建 CLI parser。
1026     ap.add_argument("--project-root", default="/home/xuxj/2026/dyn_relation_memory")  # 项目根目录。
1027     ap.add_argument("--dataset-root", default="/home/xuxj/2026/datasets")  # 原始数据根目录。
1028     ap.add_argument("--python-bin", default="/home/xuxj/.conda/envs/pytorch/bin/python")  # 记录 Python 路径。
1029     ap.add_argument("--feature-dim", type=int, default=128)  # 节点特征维度，默认 128。
1030     ap.add_argument("--gpu-summary", default="GPU 0 preferred after nvidia-smi check")  # 仅写入 process 文档的说明。
1031     ap.add_argument("--datasets", default="", help="Comma-separated dataset names, e.g. patent_us,patent_cn")  # 可筛选数据集。
1032     args = ap.parse_args()  # 解析参数。
1034     project_root = Path(args.project_root).resolve()  # 规范化项目根目录。
1035     dataset_root = Path(args.dataset_root).resolve()  # 规范化原始数据根目录。
1036     processed_root = ensure_dir(project_root / "data" / "processed")  # 创建/定位 processed 目录。
1037     task_root = ensure_dir(processed_root / "tasks")  # 创建/定位 task view 目录。
1038     adapter_root = ensure_dir(project_root / "data" / "interim" / "baseline_adapters")  # 创建/定位 baseline adapter 目录。
1039     report_root = ensure_dir(project_root / "data" / "reports")  # 创建/定位报告目录。
1048     selected = {x.strip() for x in str(args.datasets).split(",") if x.strip()}  # 解析 --datasets 过滤集合。
1050     for spec in DATASET_SPECS:  # 依次遍历 us/cn/in。
1051         if selected and spec["name"] not in selected:  # 如果指定了过滤且当前不是目标。
1052             continue  # 跳过该数据集。
1053         unified, counts = build_unified_dataset(spec, dataset_root, args.feature_dim)  # 构造 unified 图；patent_in 在这里进入核心函数。
1054         unified_path = processed_root / f"{spec['name']}_unified.pt"  # patent_in_unified.pt 输出路径。
1055         torch.save(unified, unified_path)  # 保存 unified 对象。
1056         print(f"[save] unified {unified_path}", flush=True)  # 打印保存路径。
1058         task_views = build_task_views(unified)  # 生成三个 patent task view。
1059         task_paths = []  # 收集 task 文件路径。
1060         for suffix, task_obj in task_views.items():  # 遍历三个 view。
1061             out_path = task_root / f"{spec['name']}_{suffix}.pt"  # 拼出 task pt 路径。
1062             torch.save(task_obj, out_path)  # 保存 task view。
1063             export_baseline_bundle(task_obj, adapter_root / out_path.stem, out_path.stem)  # 导出 TGN 与 TGB2/STHN adapter。
1064             task_paths.append(out_path)  # 记录路径。
1065             print(f"[save] task {out_path}", flush=True)  # 打印保存路径。
1067         analyze_stats = [analyze_one(unified, report_root)]  # 对 unified 生成统计图和 stats.json。
1068         for task_obj in task_views.values():  # 遍历 task view。
1069             analyze_stats.append(analyze_one(task_obj, report_root))  # 也生成 task 级统计。
1070         stats_rows.extend(analyze_stats)  # 汇总统计行。
1071         print(f"[analyze] completed {spec['name']}", flush=True)  # 打印分析完成。
1124         outputs = [unified_path] + task_paths + [  # 组织 process md 的输出列表。
1125             adapter_root / p.stem / "tgn" / f"ml_{p.stem}.csv" for p in task_paths  # TGN adapter CSV。
1126         ] + [
1127             adapter_root / p.stem / "tgb2" / f"{p.stem}_edgelist.csv" for p in task_paths  # TGB2/STHN edgelist。
1128         ]
1129         write_process_md(  # 写单数据集 process log。
1130             path=project_root / f"{spec['name']}_process.md",  # patent_in_process.md。
1131             dataset_name=spec["name"],  # patent_in。
1135             counts=counts,  # 核心统计。
1136             outputs=outputs,  # 输出文件列表。
1137         )
1139     pd.DataFrame(stats_rows).drop_duplicates(subset=["dataset"]).to_csv(report_root / "dataset_stats_summary.csv", index=False)  # 汇总统计 CSV。
1140     write_summary_md(project_root, dataset_summaries)  # 更新 patent.md。
1141     print("[done] patent preprocessing pipeline completed", flush=True)  # 全流程结束。
1144 if __name__ == "__main__":  # Python 脚本直接运行入口。
1145     main()  # 调用 main。
```

注意：源码第 `1144` 行是 `if __name__ == "__main__":`，第 `1145` 行调用 `main()`。

## 8. Baseline adapter 导出片段

原始位置：`scripts/experiments/run_patent_preprocessing.py:199-238`

```python
199 def export_tgn(obj: Dict, out_dir: Path, name: str) -> None:  # 导出 TGN 风格输入。
200     ensure_dir(out_dir)  # 创建输出目录。
201     src = obj["edges"]["src"].numpy()  # 源节点。
202     dst = obj["edges"]["dst"].numpy()  # 目标节点。
203     ts = obj["edges"]["ts"].numpy()  # 时间。
204     edge_feat = obj["edges"]["edge_feat"].numpy()  # 边特征。
206     with (out_dir / f"{name}.csv").open("w", encoding="utf-8") as f:  # 写一个简化 CSV。
207         f.write("u,i,ts,label,f0,f1,f2,f3\n")  # 表头。
208         for i in range(len(src)):  # 遍历每条边。
209             f0, f1, f2, f3 = [float(x) for x in edge_feat[i, :4]]  # 取 4 维边特征。
210             f.write(f"{int(src[i])},{int(dst[i])},{float(ts[i])},1,{f0},{f1},{f2},{f3}\n")  # 写正例边。
212     df = pd.DataFrame({"u": src + 1, "i": dst + 1, "ts": ts, "label": 1, "idx": range(1, len(src) + 1)})  # TGN 常用 1-based 节点和边索引。
213     df.to_csv(out_dir / f"ml_{name}.csv", index=False)  # 写 ml_*.csv。
214     np.save(out_dir / f"ml_{name}.npy", np.vstack([np.zeros((1, edge_feat.shape[1]), dtype=np.float32), edge_feat]))  # 写边特征 npy，首行补零。
215     node_feat = obj["node_feat"].numpy().astype(np.float32)  # 节点特征。
216     np.save(out_dir / f"ml_{name}_node.npy", np.vstack([np.zeros((1, node_feat.shape[1]), dtype=np.float32), node_feat]))  # 写节点特征 npy，首行补零。

219 def export_tgb2(obj: Dict, out_dir: Path, name: str) -> None:  # 导出 TGB2/STHN 风格输入。
220     ensure_dir(out_dir)  # 创建输出目录。
221     pd.DataFrame(  # 构建 edgelist。
223             "u": obj["edges"]["src"].numpy(),  # 源节点。
224             "i": obj["edges"]["dst"].numpy(),  # 目标节点。
225             "ts": obj["edges"]["ts"].numpy(),  # 时间戳。
226             "edge_type": obj["edges"]["rel"].numpy(),  # 异质关系类型。
227             "label": 1,  # 导出文件中边均为正例。
228             "idx": range(obj["edges"]["src"].shape[0]),  # 0-based 边索引。
230     ).to_csv(out_dir / f"{name}_edgelist.csv", index=False)  # 写 edgelist CSV。
231     pd.DataFrame({"node_id": range(obj["num_nodes"]), "node_type": obj["node_type"].numpy()}).to_csv(  # 写节点类型表。
232         out_dir / f"{name}_nodetype.csv", index=False  # 输出 nodetype CSV。
233     )

236 def export_baseline_bundle(obj: Dict, out_root: Path, name: str) -> None:  # baseline 导出总入口。
237     export_tgn(obj, out_root / "tgn", name)  # 生成 tgn 子目录。
238     export_tgb2(obj, out_root / "tgb2", name)  # 生成 tgb2 子目录。
```

## 9. 重点数据流

- 原始输入：
  - `/home/xuxj/2026/datasets/patent_in/all-in.csv`
  - `/home/xuxj/2026/datasets/patent_in/clsIII-reInd.csv`
  - `/home/xuxj/2026/datasets/patent_in/IPCSubClass_US.csv`
- 主字段：
  - `patentId` -> patent 节点 key `patent:{patentId}`
  - `orgID` -> org 节点 key `org:{orgID}`
  - `codeIIIId` -> 先按 `clsIII-reInd.csv` 校验为 `codeIIIIdreId`，再生成 code 节点 key `code:{codeIIIIdreId}`
  - `title + summary` -> patent 文本特征
  - `orgname` -> org 文本特征
  - `codeIII + IPC description` -> codeIII 文本特征
  - `publicationDate/date_time/date_parsed` -> 所有该专利派生边的时间戳
- 节点类型：
  - `0`: `patent`
  - `1`: `org`
  - `2`: `codeIII`
- 关系类型：
  - `0`: `patent_org`
  - `1`: `patent_codeIII`
  - `2`: `org_codeIII`
  - `3`: `org_coorg`
- 监督任务：
  - 边预测目标关系：`org_codeIII`
  - transductive：按 `org_codeIII` 目标边时间切 70/15/15，并保留对应时间窗内的所有异质上下文边。
  - inductive：按 org 首次出现时间切 seen/unseen org，训练上下文排除 unseen org；若 val/test 目标边为空则回退到时间切分。
  - 节点多标签：`patent` 节点预测 `codeIIIId` 多标签，标签矩阵来自每个 patent 关联的 codeIIIId 集合。
- 输出：
  - `data/processed/patent_in_unified.pt`
  - `data/processed/tasks/patent_in_edge_org_codeIII_transductive.pt`
  - `data/processed/tasks/patent_in_edge_org_codeIII_inductive.pt`
  - `data/processed/tasks/patent_in_node_patent_multilabel.pt`
  - `data/interim/baseline_adapters/patent_in_*`

## 10. Patent-IN 已记录统计

来自已有 `patent_in_process.md`，本次未重新计算：

- `rows_total`: `160718`
- `rows_processed`: `160514`
- `rows_skipped_missing_org_or_code`: `204`
- `num_patent_nodes`: `160514`
- `num_org_nodes`: `3166`
- `num_codeIII_nodes`: `627`
- `num_edges_patent_org`: `186660`
- `num_edges_patent_codeIII`: `411184`
- `num_edges_org_codeIII`: `483440`
- `num_edges_org_coorg`: `73520`
- `num_unique_org_code_pairs`: `139748`
- `num_unique_org_org_pairs`: `8953`
- `num_multilabel_classes`: `627`
- `num_labeled_patents`: `160514`
