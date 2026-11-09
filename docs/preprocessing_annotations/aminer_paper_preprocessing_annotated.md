# AMiner-Paper 预处理源码注释

本文档只注释 AMiner-Paper 数据集预处理和 paper task view 生成的相关代码路径。

## 原始源码位置

- 统一图预处理入口：`scripts/datasets/prepare_dataset.py`
- AMiner paper 记录迭代器：`scripts/datasets/prepare_dataset.py:980`
- AMiner author 记录迭代器：`scripts/datasets/prepare_dataset.py:1057`
- AMiner Author2Paper 行迭代器：`scripts/datasets/prepare_dataset.py:1124`
- AMiner Coauthor 行迭代器：`scripts/datasets/prepare_dataset.py:1150`
- AMiner 主解析函数：`scripts/datasets/prepare_dataset.py:1173`
- CLI 主分支：`scripts/datasets/prepare_dataset.py:1642`
- paper task view 生成入口：`scripts/datasets/build_paper_task_views.py:272`
- paper task CLI 入口：`scripts/datasets/build_paper_task_views.py:332`

## 执行链路

1. 命令行运行 `prepare_dataset.py --dataset aminer_paper --archive AMiner-Paper.rar ...`。
2. `main()` 解析参数，把全局文本编码配置写入 `TEXT_ENCODER/TEXT_DIM/TEXT_MODEL_NAME/TEXT_BATCH_SIZE`。
3. `main()` 在 `args.dataset == "aminer_paper"` 分支调用 `parse_aminer_paper(...)`。
4. `parse_aminer_paper(...)` 检查 `unrar` 和三个同目录 zip：`AMiner-Author.zip`、`AMiner-Author2Paper.zip`、`AMiner-Coauthor.zip`。
5. `iter_aminer_paper_records(...)` 从 rar 流式读取论文记录，解析 `#index/#*/#@/#t/#c/#!/#%` 字段。
6. `parse_aminer_paper(...)` 过滤年份窗口，按年份配额采样引用边，建立 paper 节点和 `paper_cite_paper` 边。
7. `iter_aminer_author2paper_rows(...)` 读取作者-论文关系，只保留 active paper 涉及的行，得到 paper 的作者列表。
8. `iter_aminer_author_records(...)` 读取 active author 的元信息，用于 author 节点文本特征。
9. `iter_aminer_coauthor_rows(...)` 读取合作者边，只保留 active author 之间的 pair。
10. `parse_aminer_paper(...)` 生成 `author_write_paper`、`paper_publish_venue`、`author_coauthor_author`，拼接成统一异构时序边表。
11. `parse_aminer_paper(...)` 构造结构特征、文本 hash 特征、paper topic 标签和 `meta_extra`。
12. `main()` 调用 `to_unified(...)` 排序并保存统一 `.pt`；若设置 `--export-baselines`，再导出 baseline adapter。
13. 后续命令运行 `build_paper_task_views.py --input aminer_paper_unified.pt --name-prefix ... --target-rel ...`。
14. `build(...)` 读统一图，按 `target_rel` 选监督关系，生成 transductive、inductive 和 node_multilabel 三类 task view。

## 逐行注释版源码片段

### 1. `iter_aminer_paper_records`

原始位置：`scripts/datasets/prepare_dataset.py:980-1054`

```python
def iter_aminer_paper_records(archive_path: Path):
    # 定义论文 rar 的流式迭代器；输入是 AMiner-Paper.rar 的路径。
    proc = subprocess.Popen(
        # 用 `unrar p` 把 rar 内文件内容直接打印到 stdout，不解压落盘。
        ["unrar", "p", "-inul", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1,
    )
    # 如果 stdout 没有成功建立，说明无法从 unrar 管道读数据。
    if proc.stdout is None:
        raise RuntimeError("Failed to open unrar pipe for AMiner paper data")

    # cur 保存当前正在解析的一篇论文记录。
    cur = None

    def _emit_current():
        # 内部函数：把 cur 清洗成标准 dict；没有有效论文时返回 None。
        if cur is None:
            return None
        # AMiner 论文 id 来自 #index 行，必须非空。
        pid = str(cur.get("id", "")).strip()
        if not pid:
            return None
        return {
            # 统一输出字段：id/title/authors/year/venue/abstract/references。
            "id": pid,
            "title": str(cur.get("title", "")).strip(),
            "authors": str(cur.get("authors", "")).strip(),
            "year": str(cur.get("year", "")).strip(),
            "venue": str(cur.get("venue", "")).strip(),
            "abstract": str(cur.get("abstract", "")).strip(),
            # 引用列表来自多个 #% 行，过滤空引用 id。
            "references": [str(x).strip() for x in cur.get("references", []) if str(x).strip()],
        }

    # early_stop 用来区分调用方主动提前结束和 unrar 真失败。
    early_stop = False
    try:
        # 逐行读取 rar 输出，避免把大文件一次性载入内存。
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("#index"):
                # 遇到新论文开头时，先 yield 上一篇已经收集完的记录。
                item = _emit_current()
                if item is not None:
                    yield item
                # 新建当前论文记录；references 默认空列表。
                cur = {"id": line[len("#index") :].strip(), "references": []}
                continue
            if cur is None:
                # 文件开头若出现非 #index 行，无法归属论文，跳过。
                continue
            if line.startswith("#*"):
                # #* 是论文标题。
                cur["title"] = line[2:].strip()
            elif line.startswith("#@"):
                # #@ 是作者字符串，后续用于 paper 文本补充，不直接生成 author 节点。
                cur["authors"] = line[2:].strip()
            elif line.startswith("#t"):
                # #t 是年份字符串，后续用正则抽取 4 位年份。
                cur["year"] = line[2:].strip()
            elif line.startswith("#c"):
                # #c 是 venue 名称，后续生成 venue 节点和 paper_publish_venue 边。
                cur["venue"] = line[2:].strip()
            elif line.startswith("#!"):
                # #! 是摘要；若出现多行摘要，就拼接到已有摘要后面。
                old = str(cur.get("abstract", "")).strip()
                cur["abstract"] = (old + " " + line[2:].strip()).strip() if old else line[2:].strip()
            elif line.startswith("#%"):
                # #% 是一条引用论文 id；一篇论文可有多条引用。
                cur.setdefault("references", []).append(line[2:].strip())

        # 文件结束后补发最后一篇论文。
        item = _emit_current()
        if item is not None:
            yield item
    except GeneratorExit:
        # 调用方 break 结束生成器时会进入这里，不应把 unrar 管道关闭当错误处理。
        early_stop = True
        raise
    finally:
        # 关闭 stdout/stderr，等待 unrar 子进程退出。
        if proc.stdout is not None:
            proc.stdout.close()
        stderr = ""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
            proc.stderr.close()
        ret = proc.wait()
        # 只有非主动提前结束且 unrar 返回非 0 时才报错。
        if ret != 0 and not early_stop:
            raise RuntimeError(f"unrar failed for {archive_path}: {stderr.strip()}")
```

### 2. `iter_aminer_author_records`

原始位置：`scripts/datasets/prepare_dataset.py:1057-1121`

```python
def iter_aminer_author_records(archive_path: Path, wanted_ids: Optional[set] = None):
    # 定义作者元信息 zip 的流式迭代器；wanted_ids 可限制只读取 active author。
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        if not names:
            return
        # 代码假设 zip 中第一个文件就是作者文本数据。
        with z.open(names[0]) as f:
            cur = None
            # wanted 是局部副本，避免修改调用方传入的 set。
            wanted = set(wanted_ids) if wanted_ids is not None else None

            def _emit_current():
                # 把当前作者记录清洗成标准 dict。
                if cur is None:
                    return None
                aid = str(cur.get("id", "")).strip()
                if not aid:
                    return None
                # 若指定 wanted，只输出目标作者。
                if wanted is not None and aid not in wanted:
                    return None
                # 输出后从 wanted 删除，用于提前停止扫描。
                if wanted is not None:
                    wanted.discard(aid)
                return {
                    "id": aid,
                    "name": str(cur.get("name", "")).strip(),
                    "affiliation": str(cur.get("affiliation", "")).strip(),
                    "paper_count": str(cur.get("paper_count", "")).strip(),
                    "citation_count": str(cur.get("citation_count", "")).strip(),
                    "h_index": str(cur.get("h_index", "")).strip(),
                    "pi": str(cur.get("pi", "")).strip(),
                    "upi": str(cur.get("upi", "")).strip(),
                    "topics": str(cur.get("topics", "")).strip(),
                }

            for raw_line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith("#index"):
                    # 新作者开始前，先发出上一位作者。
                    item = _emit_current()
                    if item is not None:
                        yield item
                        # 所有 wanted author 都读到后提前结束，减少全量扫描成本。
                        if wanted is not None and len(wanted) == 0:
                            return
                    cur = {"id": line[len("#index") :].strip()}
                    continue
                if cur is None:
                    continue
                if line.startswith("#n"):
                    # 作者姓名。
                    cur["name"] = line[2:].strip()
                elif line.startswith("#a"):
                    # 作者机构。
                    cur["affiliation"] = line[2:].strip()
                elif line.startswith("#pc"):
                    # paper count，当前预处理只保存在 meta dict 中，不直接入图。
                    cur["paper_count"] = line[3:].strip()
                elif line.startswith("#cn"):
                    # citation count，当前预处理不直接作为数值特征。
                    cur["citation_count"] = line[3:].strip()
                elif line.startswith("#hi"):
                    # h-index，当前预处理不直接作为数值特征。
                    cur["h_index"] = line[3:].strip()
                elif line.startswith("#pi"):
                    cur["pi"] = line[3:].strip()
                elif line.startswith("#upi"):
                    cur["upi"] = line[4:].strip()
                elif line.startswith("#t"):
                    # 作者研究主题，用于 author 节点文本。
                    cur["topics"] = line[2:].strip()

            # 文件结束后补发最后一位作者。
            item = _emit_current()
            if item is not None:
                yield item
```

### 3. `iter_aminer_author2paper_rows`

原始位置：`scripts/datasets/prepare_dataset.py:1124-1147`

```python
def iter_aminer_author2paper_rows(archive_path: Path):
    # 定义 Author2Paper zip 的行迭代器。
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        if not names:
            return
        with z.open(names[0]) as f:
            for raw_line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                line = raw_line.strip()
                if not line:
                    continue
                # 源文件是空白分隔格式。
                parts = re.split(r"\s+", line)
                if len(parts) < 4:
                    continue
                try:
                    # parts[1] 是 author_id，parts[2] 是 paper_id，parts[3] 是作者顺序。
                    author_id = str(int(parts[1]))
                    paper_id = str(int(parts[2]))
                    author_order = int(parts[3])
                except Exception:
                    # 非法行直接跳过，避免单行脏数据中断全流程。
                    continue
                yield {
                    "author_id": author_id,
                    "paper_id": paper_id,
                    "author_order": author_order,
                }
```

### 4. `iter_aminer_coauthor_rows`

原始位置：`scripts/datasets/prepare_dataset.py:1150-1170`

```python
def iter_aminer_coauthor_rows(archive_path: Path):
    # 定义 Coauthor zip 的行迭代器。
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        if not names:
            return
        with z.open(names[0]) as f:
            for raw_line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                line = raw_line.strip()
                if not line:
                    continue
                # 部分行可能以 # 开头；这里去掉前导 #。
                line = line.lstrip("#")
                parts = re.split(r"\s+", line)
                if len(parts) < 3:
                    continue
                try:
                    # 三列分别是两个 author id 和合著权重。
                    src = str(int(parts[0]))
                    dst = str(int(parts[1]))
                    weight = int(parts[2])
                except Exception:
                    continue
                yield {"src_author_id": src, "dst_author_id": dst, "weight": weight}
```

### 5. `parse_aminer_paper`

原始位置：`scripts/datasets/prepare_dataset.py:1173-1530`

```python
def parse_aminer_paper(archive_path: Path, max_edges: int, min_year: int = 2002, max_year: int = 2012):
    # AMiner-Paper 是 rar，必须依赖系统 unrar 流式读取。
    if not shutil.which("unrar"):
        raise RuntimeError("unrar is required to parse AMiner-Paper.rar")

    # 三个辅助 zip 被约定放在 AMiner-Paper.rar 同级目录。
    aminer_dir = archive_path.parent
    author_zip = aminer_dir / "AMiner-Author.zip"
    author2paper_zip = aminer_dir / "AMiner-Author2Paper.zip"
    coauthor_zip = aminer_dir / "AMiner-Coauthor.zip"
    missing = [str(p.name) for p in [author_zip, author2paper_zip, coauthor_zip] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"AMiner heterogeneous parsing requires sibling archives next to {archive_path.name}: {', '.join(missing)}"
        )

    # paper_meta 保存被采样源论文的标题、摘要、年份、venue、引用等元信息。
    paper_meta: Dict[str, Dict[str, Any]] = {}
    # active_paper_ids 包括源论文和被引用论文，用于过滤 Author2Paper。
    active_paper_ids: set = set()
    # selected_source_paper_ids 只包括经过年份和引用采样后保留的源论文。
    selected_source_paper_ids: set = set()
    # source_paper_nodes 记录源论文节点 id，后续只对这些 paper 生成 topic 标签。
    source_paper_nodes: List[int] = []

    def get_node(k: str):
        # 用字符串 key 管理全局节点表，例如 paper:123、author:456、venue:KDD。
        if k not in node_map:
            node_map[k] = len(node_map)
        return node_map[k]

    def merge_text(nid: int, text: str) -> None:
        # 把多来源文本合并到同一节点，供 append_text_features 使用。
        t = str(text).strip()
        if not t:
            return
        old = node_text_map.get(int(nid), "")
        if old:
            # 简单去重：完全相同片段不重复拼接。
            if t not in old:
                node_text_map[int(nid)] = (old + " " + t)[:512]
        else:
            node_text_map[int(nid)] = t[:512]

    # 全局节点 id 映射和节点文本缓存。
    node_map: Dict[str, int] = {}
    node_text_map: Dict[int, str] = {}
    # AMiner 统一图包含 paper/author/venue 三种节点类型。
    node_type_map = {"paper": 0, "author": 1, "venue": 2}
    paper_year: Dict[int, float] = {}
    paper_venue: Dict[int, str] = {}
    paper_doc_text: Dict[int, str] = {}

    # citation_* 暂存 paper_cite_paper 边。
    citation_src: List[int] = []
    citation_dst: List[int] = []
    citation_rel: List[int] = []
    citation_ts: List[float] = []
    # 关系 id 在统一图中的定义；task view 后续按这些 id 指定监督关系。
    rel_map = {
        "paper_cite_paper": 0,
        "author_write_paper": 1,
        "paper_publish_venue": 2,
        "author_coauthor_author": 3,
    }
    papers_with_edges = 0
    # 按年份窗口构造平衡采样配额，避免 citation 边集中在少数年份。
    year_values = list(range(int(min_year), int(max_year) + 1))
    per_year_cap = max(1, int(np.ceil(float(max_edges) / max(1, len(year_values))))) if max_edges > 0 else 0
    year_edge_used: Dict[int, int] = {y: 0 for y in year_values}

    for obj in iter_aminer_paper_records(archive_path):
        # 从论文记录的 year 字符串中提取 4 位年份。
        year_raw = obj.get("year", "")
        year_match = re.search(r"\d{4}", year_raw)
        if year_match is None:
            continue
        year_i = int(year_match.group(0))
        year = float(year_i)
        # 只保留配置年份窗口内的论文。
        if year_i < int(min_year):
            continue
        if year_i > int(max_year):
            continue
        refs = obj.get("references", [])
        # 没有引用的论文不产生 citation 监督边，因此跳过。
        if not refs:
            continue
        if max_edges > 0:
            # 同时受总 citation 边数和单年配额限制。
            remaining_total = int(max_edges - len(citation_src))
            if remaining_total <= 0:
                break
            remaining_year = int(per_year_cap - year_edge_used.get(year_i, 0))
            if remaining_year <= 0:
                continue
            refs = refs[: min(len(refs), remaining_total, remaining_year)]
            if not refs:
                continue

        # 保存源论文元信息；注意这里只保存采样后保留的 refs。
        pid = str(obj.get("id", ""))
        paper_meta[pid] = {
            "title": str(obj.get("title", "")).strip(),
            "abstract": str(obj.get("abstract", "")).strip(),
            "authors": str(obj.get("authors", "")).strip(),
            "year": year,
            "venue": str(obj.get("venue", "")).strip(),
            "references": [str(x).strip() for x in refs if str(x).strip()],
        }
        selected_source_paper_ids.add(pid)
        active_paper_ids.add(pid)
        active_paper_ids.update(refs)

        # 源论文进入 node_map；被引用论文稍后在 citation 循环中进入 node_map。
        p = get_node(f"paper:{pid}")
        paper_year[p] = year
        venue = str(obj.get("venue", "")).strip()
        if venue:
            paper_venue[p] = venue
        title = str(obj.get("title", "")).strip()
        abstract = str(obj.get("abstract", "")).strip()
        authors = str(obj.get("authors", "")).strip().replace(";", " ")
        doc_text = f"{title} {venue} {abstract}".strip()
        paper_doc_text[p] = doc_text
        source_paper_nodes.append(p)
        merge_text(p, f"{doc_text} authors {authors} year_{int(year)}")

        for ref in refs:
            # 每个引用 id 作为目标 paper 节点，构造有向 citation 边：source paper -> referenced paper。
            rp = get_node(f"paper:{ref}")
            citation_src.append(p)
            citation_dst.append(rp)
            citation_rel.append(rel_map["paper_cite_paper"])
            citation_ts.append(year)
        year_edge_used[year_i] = int(year_edge_used.get(year_i, 0) + len(refs))

        papers_with_edges += 1
        if max_edges > 0 and len(citation_src) >= max_edges:
            break

    # 读取 Author2Paper 后得到 active author、paper->author 列表和作者首次出现年份。
    active_author_ids: set = set()
    paper_authors: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    author_first_year: Dict[str, float] = {}
    author_topic_meta: Dict[str, str] = {}
    author_name_meta: Dict[str, str] = {}
    author_affil_meta: Dict[str, str] = {}
    venue_node_ids: List[int] = []

    write_src: List[int] = []
    write_dst: List[int] = []
    write_rel: List[int] = []
    write_ts: List[float] = []

    venue_src: List[int] = []
    venue_dst: List[int] = []
    venue_rel: List[int] = []
    venue_ts: List[float] = []

    for row in iter_aminer_author2paper_rows(author2paper_zip):
        paper_id = str(row["paper_id"])
        if paper_id not in active_paper_ids:
            continue
        author_id = str(row["author_id"])
        active_author_ids.add(author_id)
        year = float(paper_meta.get(paper_id, {}).get("year", 0.0))
        if year > 0:
            prev = author_first_year.get(author_id, np.inf)
            author_first_year[author_id] = float(min(prev, year))
        paper_authors[paper_id].append((author_id, int(row["author_order"])))

    for meta in iter_aminer_author_records(author_zip, wanted_ids=active_author_ids):
        # 只读取 active author 的元信息，用于后续 author 节点文本。
        author_id = str(meta["id"])
        if author_id not in active_author_ids:
            continue
        author_name_meta[author_id] = str(meta.get("name", "")).strip()
        author_affil_meta[author_id] = str(meta.get("affiliation", "")).strip()
        author_topic_meta[author_id] = str(meta.get("topics", "")).strip()

    coauthor_weight_meta: Dict[Tuple[str, str], int] = {}
    for row in iter_aminer_coauthor_rows(coauthor_zip):
        # 只保留 active author 之间的 coauthor pair；无向 pair 用排序 tuple 表示。
        a = str(row["src_author_id"])
        b = str(row["dst_author_id"])
        if a not in active_author_ids or b not in active_author_ids or a == b:
            continue
        key = tuple(sorted((a, b)))
        coauthor_weight_meta[key] = int(row["weight"])

    dynamic_coauthor_seen: set = set()
    co_src: List[int] = []
    co_dst: List[int] = []
    co_rel: List[int] = []
    co_ts: List[float] = []

    for paper_id, author_rows in paper_authors.items():
        # 对每篇 active paper，根据作者列表生成 write、venue、coauthor 动态边。
        p_node = get_node(f"paper:{paper_id}")
        meta = paper_meta.get(paper_id, {})
        year = float(meta.get("year", 0.0))
        venue = str(meta.get("venue", "")).strip()

        if paper_id in selected_source_paper_ids:
            # 只有源论文生成 author_write_paper 和 paper_publish_venue 边。
            for author_id, _ in sorted(author_rows, key=lambda x: (x[1], x[0])):
                a_node = get_node(f"author:{author_id}")
                write_src.append(a_node)
                write_dst.append(p_node)
                write_rel.append(rel_map["author_write_paper"])
                write_ts.append(year)

            if venue:
                v_node = get_node(f"venue:{venue}")
                venue_node_ids.append(v_node)
                venue_src.append(p_node)
                venue_dst.append(v_node)
                venue_rel.append(rel_map["paper_publish_venue"])
                venue_ts.append(year)

        sorted_authors = [aid for aid, _ in sorted(author_rows, key=lambda x: (x[1], x[0]))]
        for i in range(len(sorted_authors)):
            for j in range(i + 1, len(sorted_authors)):
                # 同一篇论文内任意作者两两组成 coauthor pair。
                a = sorted_authors[i]
                b = sorted_authors[j]
                key = tuple(sorted((a, b)))
                # 若 Coauthor zip 提供了 pair 白名单，则只保留其中存在的 pair。
                if coauthor_weight_meta and key not in coauthor_weight_meta:
                    continue
                # 同一 author pair 同一年只生成一次动态 coauthor 事件。
                edge_key = (key[0], key[1], int(year))
                if edge_key in dynamic_coauthor_seen:
                    continue
                dynamic_coauthor_seen.add(edge_key)
                a_node = get_node(f"author:{a}")
                b_node = get_node(f"author:{b}")
                # coauthor 是无向语义，但统一图用两条有向边表示双向关系。
                co_src.append(a_node)
                co_dst.append(b_node)
                co_rel.append(rel_map["author_coauthor_author"])
                co_ts.append(year)
                co_src.append(b_node)
                co_dst.append(a_node)
                co_rel.append(rel_map["author_coauthor_author"])
                co_ts.append(year)

    n = len(node_map)
    node_type = np.zeros(n, dtype=np.int64)
    for key, nid in node_map.items():
        # 根据节点 key 前缀写入 node_type。
        if key.startswith("author:"):
            node_type[nid] = node_type_map["author"]
        elif key.startswith("venue:"):
            node_type[nid] = node_type_map["venue"]
        else:
            node_type[nid] = node_type_map["paper"]

    for key, nid in node_map.items():
        # 第二遍遍历节点，补全 paper/author/venue 文本。
        if key.startswith("paper:"):
            pid = key.split(":", 1)[1]
            meta = paper_meta.get(pid, {})
            title = str(meta.get("title", "")).strip()
            abstract = str(meta.get("abstract", "")).strip()
            venue = str(meta.get("venue", "")).strip()
            authors = str(meta.get("authors", "")).strip().replace(";", " ")
            year = float(meta.get("year", 0.0))
            if year > 0:
                paper_year[nid] = year
            if venue:
                paper_venue[nid] = venue
            doc_text = f"{title} {venue} {abstract}".strip()
            if doc_text:
                paper_doc_text[nid] = doc_text
            merge_text(nid, f"{title} {venue} {abstract} authors {authors} year_{int(year) if year > 0 else 0}")
        elif key.startswith("author:"):
            aid = key.split(":", 1)[1]
            name = author_name_meta.get(aid, aid)
            aff = author_affil_meta.get(aid, "")
            topics = author_topic_meta.get(aid, "")
            first_year = int(author_first_year.get(aid, 0.0)) if author_first_year.get(aid, 0.0) > 0 else 0
            merge_text(
                nid,
                f"author {name} affiliation {aff} topics {topics} first_year_{first_year}",
            )
        elif key.startswith("venue:"):
            venue = key.split(":", 1)[1]
            merge_text(nid, f"venue {venue}")

    # 按固定关系块拼接统一边表：citation -> write -> venue -> coauthor。
    src = citation_src + write_src + venue_src + co_src
    dst = citation_dst + write_dst + venue_dst + co_dst
    rel = citation_rel + write_rel + venue_rel + co_rel
    ts = citation_ts + write_ts + venue_ts + co_ts

    src_np = np.array(src, dtype=np.int64)
    dst_np = np.array(dst, dtype=np.int64)
    rel_np = np.array(rel, dtype=np.int64) if rel else np.zeros((0,), dtype=np.int64)
    ts_np = np.array(ts, dtype=np.float32) if ts else np.zeros((0,), dtype=np.float32)

    # 先构造结构/时间/类型特征，再拼接文本特征。
    feat = build_degree_features(n, src_np, dst_np, ts_np, node_type, dim=64)
    node_text = []
    for i in range(n):
        txt = node_text_map.get(i, "")
        if not txt:
            # 对缺文本的 paper 节点生成兜底文本，避免文本特征为空。
            venue = paper_venue.get(i, "")
            year = int(paper_year.get(i, 0.0)) if paper_year.get(i, 0.0) > 0 else 0
            parts = ["paper", f"node_{i}"]
            if venue:
                parts.append(venue)
            if year > 0:
                parts.append(f"year_{year}")
            txt = " ".join(parts)
        node_text.append(txt)
    feat = append_text_features(feat, node_text, text_dim=TEXT_DIM)

    # topic_rules 是 paper 节点分类标签的关键词规则。
    topic_rules = [
        ("ai_ml", ["artificial intelligence", "machine learning", "neural", "data mining", "pattern recognition", "knowledge discovery", "aaai", "ijcai", "kdd", "icml", "nips", "neurips"]),
        ("theory_algorithms", ["algorithm", "complexity", "combinator", "discrete mathematics", "theoretical computer science", "symbolic computation", "approximation"]),
        ("db_ir_web", ["database", "data management", "sigmod", "vldb", "icde", "information retrieval", "web", "search", "semantic web"]),
        ("systems_software", ["software", "program", "programming", "compiler", "operating system", "testing", "design automation", "embedded"]),
        ("networks_security", ["network", "wireless", "sensor", "communication", "security", "cryptograph", "distributed systems"]),
        ("vision_graphics_hci", ["vision", "image", "graphics", "multimedia", "human factors", "chi", "computer animation"]),
        ("bio_med", ["bio", "medical", "genome", "health", "clinical", "protein"]),
        ("robotics_control", ["robot", "control", "autonomous", "navigation", "planning"]),
    ]
    label_names = [name for name, _ in topic_rules]
    ml_node_ids: List[int] = []
    ml_rows: List[List[float]] = []
    mc_node_ids: List[int] = []
    mc_labels: List[int] = []
    for node_id in sorted(set(int(x) for x in source_paper_nodes)):
        # 只对源论文节点打标签；被引用但没有元信息的目标论文不参与 paper label。
        text = str(paper_doc_text.get(node_id, "")).lower()
        if not text:
            continue
        row = [1.0 if any(tok in text for tok in toks) else 0.0 for _, toks in topic_rules]
        if sum(row) <= 0:
            continue
        ml_node_ids.append(int(node_id))
        ml_rows.append(row)
        mc_node_ids.append(int(node_id))
        # multiclass 标签取命中的第一个最大值位置。
        mc_labels.append(int(np.argmax(np.asarray(row, dtype=np.float32))))

    if ml_rows:
        # 有关键词标签时，保存稀疏的 labeled node ids 和对应标签矩阵。
        ml = np.asarray(ml_rows, dtype=np.float32)
        ml_node_ids_np = np.asarray(ml_node_ids, dtype=np.int64)
        mc = np.asarray(mc_labels, dtype=np.int64)
        mc_node_ids_np = np.asarray(mc_node_ids, dtype=np.int64)
    else:
        # 若规则没有命中任何论文，则退回默认结构标签。
        ml = build_default_multilabel(n, src_np, dst_np, ts_np)
        ml_node_ids_np = np.arange(n, dtype=np.int64)
        mc = np.zeros(n, dtype=np.int64)
        mc_node_ids_np = np.arange(n, dtype=np.int64)

    venue_counts = Counter(v for v in paper_venue.values() if v)
    top_venues = [v for v, _ in venue_counts.most_common(16)]

    return {
        # 返回的是中间 dataset_obj，之后由 to_unified(...) 统一排序和保存。
        "name": "aminer_paper",
        "src": src_np,
        "dst": dst_np,
        "ts": ts_np,
        "rel": rel_np,
        "node_type": node_type,
        "node_feat": feat,
        "node_multiclass": mc,
        "node_multiclass_node_ids": mc_node_ids_np,
        "node_multilabel": ml,
        "node_multilabel_node_ids": ml_node_ids_np,
        "meta_extra": {
            **text_encoder_meta(),
            "hetero_main_benchmark_ok": True,
            "sampling_policy": "year_window_with_balanced_edge_quota",
            "year_window": {"min_year": int(min_year), "max_year": int(max_year)},
            "per_year_edge_cap": int(per_year_cap) if max_edges > 0 else 0,
            "per_year_edge_used": {str(k): int(v) for k, v in year_edge_used.items() if int(v) > 0},
            "papers_with_edges": int(papers_with_edges),
            "active_papers": int(len(active_paper_ids)),
            "active_authors": int(len(active_author_ids)),
            "venues": int(len(set(x for x in paper_venue.values() if x))),
            "coauthor_pairs_from_zip": int(len(coauthor_weight_meta)),
            "dynamic_coauthor_edges": int(len(co_src)),
            "relation_map": rel_map,
            "node_type_map": node_type_map,
            "top_venues": top_venues,
            "node_label_names": label_names,
            "node_label_source": "paper_text_keyword_rules",
        },
    }
```

### 6. `prepare_dataset.py` 的 AMiner CLI 分支

原始位置：`scripts/datasets/prepare_dataset.py:1642-1705`

```python
def main():
    # 使用全局变量承接 CLI 文本编码配置，供 append_text_features/text_encoder_meta 读取。
    global TEXT_ENCODER, TEXT_DIM, TEXT_MODEL_NAME, TEXT_BATCH_SIZE
    p = argparse.ArgumentParser("Prepare dataset into unified .pt format")
    # --dataset 包含 aminer_paper；当前源码 default 也是 aminer_paper，但 required=True 仍要求命令行显式传入。
    p.add_argument("--dataset", default='aminer_paper', required=True, choices=["amazon_movies", "raw_cn", "raw_us", "dblp", "aminer_paper"]) #"kg4ai"
    # AMiner 场景下 --archive 指向 AMiner-Paper.rar。
    p.add_argument("--archive", required=True, type=str, default="/home/xuxj/2026/datasets/Aminer/AMiner-Paper.rar")
    # max_edges 对 AMiner 表示 citation 边的总采样上限。
    p.add_argument("--max-edges", type=int, default=200000)
    # 输出统一图 .pt。
    p.add_argument("--out-pt", required=True, type=str, default="data/processed/aminer_paper_unified_debug.pt")
    p.add_argument(
        "--text-encoder",
        type=str,
        default="hybrid_hash",
        choices=["hybrid_hash", "hash_word", "sentence_transformer", "none"],
        help="Text encoder backend. hybrid_hash is Chinese-friendly (word+char ngram).",
    )
    p.add_argument("--text-dim", type=int, default=64, help="Output dimension for text features.")
    p.add_argument(
        "--text-model-name",
        type=str,
        default="",
        help="Model name/path for sentence_transformer backend.",
    )
    p.add_argument("--text-batch-size", type=int, default=128, help="Batch size for sentence_transformer encoding.")
    # AMiner 专属年份窗口参数，传入 parse_aminer_paper。
    p.add_argument("--aminer-min-year", type=int, default=2002)
    p.add_argument("--aminer-max-year", type=int, default=2012)
    # 可选导出 baseline adapter。
    p.add_argument("--export-baselines", action="store_true")
    p.add_argument("--baseline-out-dir", type=str, default="")
    args = p.parse_args()

    # 固化文本编码参数。
    TEXT_ENCODER = str(args.text_encoder).strip().lower()
    TEXT_DIM = int(max(8, args.text_dim))
    TEXT_MODEL_NAME = str(args.text_model_name).strip()
    TEXT_BATCH_SIZE = int(max(1, args.text_batch_size))

    archive = Path(args.archive)
    out_pt = Path(args.out_pt)

    # 其他 dataset 分支省略；AMiner 进入 parse_aminer_paper。
    if args.dataset == "amazon_movies":
        d = parse_amazon_movies(archive, args.max_edges)
    elif args.dataset == "raw_cn":
        d = parse_raw_patent(archive, "CN", args.max_edges)
    elif args.dataset == "raw_us":
        d = parse_raw_patent(archive, "US", args.max_edges)
    elif args.dataset == "dblp":
        d = parse_dblp(archive, args.max_edges)
    elif args.dataset == "aminer_paper":
        d = parse_aminer_paper(archive, args.max_edges, min_year=args.aminer_min_year, max_year=args.aminer_max_year)
    else:
        raise ValueError(args.dataset)

    # to_unified 会按时间排序边、构造 split、meta，并写出 .pt。
    obj = to_unified(d, out_pt)
    print(f"saved unified dataset to {out_pt}")
    print(f"nodes={obj['num_nodes']} edges={obj['edges']['src'].shape[0]} rels={obj['num_rels']}")

    if args.export_baselines:
        # baseline adapter 输出目录必须显式指定。
        if not args.baseline_out_dir:
            raise ValueError("--baseline-out-dir is required when --export-baselines is set")
        bdir = Path(args.baseline_out_dir)
        # 导出 TGN 和 TGB2-like CSV 格式。
        export_tgn_format(obj, bdir / "tgn", d["name"])
        export_tgb2_like_csv(obj, bdir / "tgb2", d["name"])
        print(f"baseline adapters exported to {bdir}")


if __name__ == "__main__":
    main()
```

### 7. `build_paper_task_views.py` 的 task view 入口

原始位置：`scripts/datasets/build_paper_task_views.py:262-352`

```python
def choose_target_relation(rel: np.ndarray, target_rel: int = None) -> int:
    # 如果命令行显式传入 --target-rel，则用它作为监督关系。
    if target_rel is not None and int(target_rel) >= 0:
        return int(target_rel)
    # 否则默认选边数最多的关系；AMiner 中通常会选到 coauthor，因此 citation/coauthor 应显式传参。
    vals, cnt = np.unique(rel, return_counts=True)
    if vals.size == 0:
        return 0
    best_rel = int(vals[np.argmax(cnt)])
    return best_rel


def build(input_pt: Path, out_dir: Path, prefix: str, target_rel: int = None):
    # 读取 prepare_dataset.py 生成的统一异构图。
    obj = torch.load(input_pt, map_location="cpu")
    rel = obj["edges"]["rel"].numpy().astype(np.int64)
    ts = obj["edges"]["ts"].numpy().astype(np.float32)
    src = obj["edges"]["src"].numpy().astype(np.int64)
    dst = obj["edges"]["dst"].numpy().astype(np.int64)

    # 确定监督目标关系，例如 citation 用 0，coauthor 用 3。
    target_rel_id = choose_target_relation(rel, target_rel=target_rel)
    # target_old 是统一图原始边索引中属于目标关系的边，按时间排序。
    target_old = sort_old_indices(np.where(rel == int(target_rel_id))[0].astype(np.int64), ts)
    if target_old.size == 0:
        raise RuntimeError(f"no edges found for target relation id={target_rel_id}")

    # transductive：目标关系边按时间 70/15/15 切分。
    trans_tr_old, trans_va_old, trans_te_old = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)
    trans_view = build_edge_view(
        obj=obj,
        out_name=f"{prefix}_edge_transductive",
        target_old=target_old,
        train_target_old=trans_tr_old,
        val_target_old=trans_va_old,
        test_target_old=trans_te_old,
        extra_info={
            "target_original_rel_id": int(target_rel_id),
            "protocol": "full_hetero_context_transductive",
        },
    )

    # inductive：按目标边的 source node 抽 unseen 节点，构造源节点归纳测试。
    ind_tr_old, ind_va_old, ind_te_old, unseen_nodes = inductive_partition_target(
        target_old=target_old,
        src=src,
        ts=ts,
        rng=np.random.RandomState(42),
    )
    # context_keep_mask 过滤掉触及 unseen 节点的上下文边，避免训练/验证泄露。
    keep_mask = ~(np.isin(src, unseen_nodes) | np.isin(dst, unseen_nodes))
    ind_view = build_edge_view(
        obj=obj,
        out_name=f"{prefix}_edge_inductive",
        target_old=target_old,
        train_target_old=ind_tr_old,
        val_target_old=ind_va_old,
        test_target_old=ind_te_old,
        context_keep_mask=keep_mask,
        extra_info={
            "target_original_rel_id": int(target_rel_id),
            "protocol": "full_hetero_context_inductive",
            "unseen_source_node_ids": unseen_nodes.tolist(),
        },
    )

    # node view 不限制单一目标关系，而是保留完整异构时间流，用于 node_multilabel。
    node_view = build_sorted_full_view(obj, out_name=f"{prefix}_node_multilabel")

    # 写出三个 task view。
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"{prefix}_edge_transductive.pt"
    p2 = out_dir / f"{prefix}_edge_inductive.pt"
    p3 = out_dir / f"{prefix}_node_multilabel.pt"
    torch.save(trans_view, p1)
    torch.save(ind_view, p2)
    torch.save(node_view, p3)
    return [p1, p2, p3]


def main():
    # paper task view CLI。
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--name-prefix", required=True, type=str)
    ap.add_argument("--target-rel", type=int, default=-1)
    args = ap.parse_args()

    outs = build(
        input_pt=Path(args.input),
        out_dir=Path(args.out_dir),
        prefix=args.name_prefix,
        target_rel=None if args.target_rel < 0 else int(args.target_rel),
    )
    print("generated:")
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()
```

## 重点数据流

### 输入文件

- `AMiner-Paper.rar`：论文 id、title、authors 字符串、year、venue、abstract、reference ids。
- `AMiner-Author2Paper.zip`：author id、paper id、作者顺序。
- `AMiner-Author.zip`：author name、affiliation、topics 等文本元信息。
- `AMiner-Coauthor.zip`：author pair 和 coauthor weight，作为合著 pair 白名单。

### 节点数据流

- `paper:{pid}` -> `node_map` -> `node_type=0`
- `author:{author_id}` -> `node_map` -> `node_type=1`
- `venue:{venue}` -> `node_map` -> `node_type=2`
- `node_text_map` 汇总 paper title/venue/abstract/authors/year、author name/affiliation/topics/first_year、venue name。
- `build_degree_features(...)` 产生结构/时间/类型特征。
- `append_text_features(...)` 把文本特征拼到结构特征后，默认结构 64 维 + 文本 64 维 = 128 维。

### 边数据流

- `paper_cite_paper`：从 `iter_aminer_paper_records` 的 `references` 生成，关系 id 为 `0`，时间戳是源论文年份。
- `author_write_paper`：从 `Author2Paper` 生成，关系 id 为 `1`，只对 `selected_source_paper_ids` 生成。
- `paper_publish_venue`：由源论文 venue 生成，关系 id 为 `2`。
- `author_coauthor_author`：由同论文作者两两组合生成，并用 `Coauthor.zip` pair 过滤，关系 id 为 `3`，用双向有向边表示。
- 四类边最终按 `citation + write + venue + coauthor` 拼接，再由 `to_unified(...)` 按时间排序保存。

### 标签数据流

- 只对 `source_paper_nodes` 尝试生成 paper topic 标签。
- `topic_rules` 用关键词命中得到 multi-label 行。
- multi-class 标签取 multi-label 行中 `argmax`。
- 若没有任何规则命中，则退回 `build_default_multilabel(...)` 的结构标签；AMiner 正常路径使用 `paper_text_keyword_rules`。

### task view 数据流

- `build_paper_task_views.py --target-rel 0` 生成 citation 监督任务：
  - `{prefix}_edge_transductive.pt`
  - `{prefix}_edge_inductive.pt`
  - `{prefix}_node_multilabel.pt`
- `build_paper_task_views.py --target-rel 3` 生成 coauthor 监督任务。
- `build_edge_view(...)` 保留完整异构上下文边，并用 `edge_label` 标记监督目标边。
- transductive 视图按目标关系边时间切分。
- inductive 视图按目标边 source node 抽 unseen 节点，并过滤触及 unseen 节点的上下文边。
- node view 使用 `build_sorted_full_view(...)`，目标是完整异构时序流上的 `node_multilabel`。

## AMiner 常用命令形态

```bash
/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/prepare_dataset.py \
  --dataset aminer_paper \
  --archive /home/xuxj/2026/datasets/Aminer/AMiner-Paper.rar \
  --out-pt /home/xuxj/2026/dyn_relation_memory/data/processed/aminer_paper_unified.pt \
  --max-edges 200000 \
  --aminer-min-year 2002 \
  --aminer-max-year 2012 \
  --export-baselines \
  --baseline-out-dir /home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/aminer_paper
```

```bash
/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/build_paper_task_views.py \
  --input data/processed/aminer_paper_unified.pt \
  --out-dir data/processed/paper_tasks \
  --name-prefix aminer_paper_citation \
  --target-rel 0
```

```bash
/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/build_paper_task_views.py \
  --input data/processed/aminer_paper_unified.pt \
  --out-dir data/processed/paper_tasks \
  --name-prefix aminer_paper_coauthor \
  --target-rel 3
```

## 核查要点

- `target_rel` 必须显式指定：AMiner 的最大关系通常是 `author_coauthor_author`，不传 `--target-rel` 会触发最大关系默认策略，citation 任务会被误导。
- citation 边采样受 `max_edges` 和 `per_year_cap` 双重限制。
- 被引用 paper 会作为节点保留，但不一定有完整 paper_meta，因此 topic 标签只从源论文节点生成。
- coauthor 边是从同论文作者组合动态生成，不是直接把 Coauthor.zip 的静态边原样入图。
- task view 中的训练损失监督由 `edge_label` 限定目标关系，但上下文仍可包含其他关系边。
