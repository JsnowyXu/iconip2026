# thgl-software 预处理代码注释

## 1. 定位结论

thgl-software 的预处理分两段：

1. **TGB2 原始 CSV 导入 unified `.pt`**
   - 源码：`/home/xuxj/2026/dyn_relation_memory/scripts/datasets/import_tgb2_thg.py`
   - 核心函数：`load_thg_dataset(...)`
   - 输出函数：`prepare_dataset.to_unified(...)`
   - 日志命令见：`/home/xuxj/2026/dyn_relation_memory/thgl_software_process.md:3-6`

2. **从 unified `.pt` 生成 edge/node task views**
   - 源码：`/home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py`
   - 核心函数：`build(...)`、`build_edge_view(...)`、`inductive_partition_target(...)`
   - 输出文件：`thgl_software_edge_transductive.pt`、`thgl_software_edge_inductive.pt`、`thgl_software_node_multilabel.pt`
   - 日志命令见：`/home/xuxj/2026/dyn_relation_memory/thgl_software_process.md:8-10`，修复后重建见 `:144-155`

实验流水线入口：

- `scripts/experiments/edge_benchmark_common.py:126-196` 的 `ensure_thgl_tasks(...)`：如果 unified 或 task view 不存在，则调用上述两个脚本。
- `scripts/experiments/edge_benchmark_common.py:318-333` 的 `TaskSpec`：把后续实验任务绑定到 `data/processed/tgb2_tasks/thgl_software_edge_*.pt`。
- 平台后端通用入口是 `platform/backend/core/services/job_builder.py:64-81`，但当前 `registry.py` 未注册 `thgl_software`，所以 thgl-software 实际主要由实验流水线入口处理。

## 2. 实际执行链路

来自 `thgl_software_process.md` 的已执行命令：

```bash
/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/import_tgb2_thg.py \
  --dataset-name thgl_software \
  --edgelist-csv /home/xuxj/2026/datasets/tgb2/thgl-software/thgl-software_edgelist.csv \
  --nodetype-csv /home/xuxj/2026/datasets/tgb2/thgl-software/thgl-software_nodetype.csv \
  --out-pt /home/xuxj/2026/dyn_relation_memory/data/processed/thgl_software_unified.pt \
  --max-edges -1
```

```bash
/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py \
  --input /home/xuxj/2026/dyn_relation_memory/data/processed/thgl_software_unified.pt \
  --out-dir /home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks \
  --name-prefix thgl_software
```

数据流：

```text
TGB2 raw CSV
  thgl-software_edgelist.csv: timestamp, head, tail, relation_type
  thgl-software_nodetype.csv: node_id, type
        |
        v
import_tgb2_thg.py::load_thg_dataset()
  读取 CSV -> 按 timestamp 稳定排序 -> 节点重编号 -> 关系重编号
  -> node_type 对齐到新节点 id -> 构造 node_feat/node labels
        |
        v
prepare_dataset.py::to_unified()
  保存统一图 schema: num_nodes, num_rels, node_feat, node_type,
  edges{src,dst,ts,rel,edge_feat,edge_label}, split, node labels, meta
        |
        v
build_paper_task_views.py::build()
  选择目标关系 rel_6/出现次数最多关系 -> 生成 transductive split
  -> 生成 source-node holdout inductive split -> 调 build_edge_view()
        |
        v
data/processed/tgb2_tasks/
  thgl_software_edge_transductive.pt
  thgl_software_edge_inductive.pt
  thgl_software_node_multilabel.pt
```

## 3. TGB2 导入脚本逐行注释

原始位置：`/home/xuxj/2026/dyn_relation_memory/scripts/datasets/import_tgb2_thg.py:1-134`

```python
#!/usr/bin/env python3  # 使用环境中的 python3 直接执行该脚本。
import argparse  # 解析命令行参数。
from pathlib import Path  # 用 Path 处理输入输出路径。
from typing import Dict  # 标注 load_thg_dataset 返回字典。

import numpy as np  # 用于数组、类型转换、排序和唯一值处理。
import pandas as pd  # 用于读取 TGB2 CSV。

from prepare_dataset import build_default_multilabel, build_degree_features, to_unified  # 复用统一图特征、标签和保存逻辑。


def maybe_factorize(series: pd.Series) -> np.ndarray:  # 通用离散值编码工具；当前脚本里未被调用。
    if np.issubdtype(series.dtype, np.integer):  # 如果原始列已经是整数类型。
        return series.to_numpy(dtype=np.int64)  # 直接转成 int64 numpy 数组。
    codes, _ = pd.factorize(series.astype(str), sort=True)  # 否则按字符串排序后 factorize，得到稳定编码。
    return codes.astype(np.int64)  # 返回 int64 编码。


def load_thg_dataset(edgelist_csv: Path, nodetype_csv: Path, dataset_name: str, max_edges: int) -> Dict:  # 核心导入函数。
    edges = pd.read_csv(edgelist_csv)  # 读取 thgl-software_edgelist.csv。
    required_edge_cols = {"timestamp", "head", "tail", "relation_type"}  # TGB2 THG 边表必须包含的列。
    missing = required_edge_cols.difference(edges.columns)  # 检查缺失列。
    if missing:  # 如果缺列。
        raise ValueError(f"missing columns in {edgelist_csv}: {sorted(missing)}")  # 立即报错，避免生成错误图。

    edges = edges.sort_values("timestamp", kind="mergesort").reset_index(drop=True)  # 按时间稳定排序；mergesort 保持同时间原始相对顺序。
    if max_edges > 0 and len(edges) > max_edges:  # max_edges=-1 表示不过滤；正数才截断。
        edges = edges.iloc[:max_edges].copy()  # 只保留最早的 max_edges 条边。

    head_raw = edges["head"]  # 原始源节点列。
    tail_raw = edges["tail"]  # 原始目标节点列。
    rel_raw = edges["relation_type"]  # 原始关系类型列。
    ts = edges["timestamp"].to_numpy(dtype=np.float32)  # 时间戳转 float32；后续统一图也按 float32 保存。

    if np.issubdtype(head_raw.dtype, np.integer) and np.issubdtype(tail_raw.dtype, np.integer):  # 如果头尾节点都是整数 id。
        all_nodes = pd.Index(
            np.unique(np.concatenate([head_raw.to_numpy(dtype=np.int64), tail_raw.to_numpy(dtype=np.int64)]))  # 合并头尾节点并去重排序。
        )
        node_to_new = {int(node_id): idx for idx, node_id in enumerate(all_nodes.tolist())}  # 原始节点 id -> 连续新 id。
        src = head_raw.map(node_to_new).to_numpy(dtype=np.int64)  # head 映射成连续 src。
        dst = tail_raw.map(node_to_new).to_numpy(dtype=np.int64)  # tail 映射成连续 dst。
        original_node_ids = all_nodes.to_numpy(dtype=np.int64)  # 保存原始节点 id 样例到 meta。
    else:  # 如果原始节点 id 不是纯整数。
        all_nodes = pd.Index(pd.concat([head_raw.astype(str), tail_raw.astype(str)], axis=0).unique())  # 转字符串后按出现顺序去重。
        node_to_new = {str(node_id): idx for idx, node_id in enumerate(all_nodes.tolist())}  # 字符串节点 id -> 连续新 id。
        src = head_raw.astype(str).map(node_to_new).to_numpy(dtype=np.int64)  # head 映射成连续 src。
        dst = tail_raw.astype(str).map(node_to_new).to_numpy(dtype=np.int64)  # tail 映射成连续 dst。
        original_node_ids = np.arange(len(all_nodes), dtype=np.int64)  # 非整数场景用新 id 样例代替原始 id 样例。

    if np.issubdtype(rel_raw.dtype, np.integer):  # 如果关系列是整数。
        rel_vals = np.sort(rel_raw.astype(np.int64).unique())  # 取所有关系原始值并排序。
        rel_map = {int(r): idx for idx, r in enumerate(rel_vals.tolist())}  # 原始关系 id -> 0..R-1 连续 id。
        rel = rel_raw.map(rel_map).to_numpy(dtype=np.int64)  # 映射每条边的关系 id。
        rel_original = rel_vals.astype(np.int64)  # 保存原始关系值样例。
    else:  # 如果关系列不是整数。
        rel_codes, rel_uniques = pd.factorize(rel_raw.astype(str), sort=True)  # 按字符串排序后编码，保证稳定关系 id。
        rel = rel_codes.astype(np.int64)  # 得到连续关系 id。
        rel_original = rel_uniques.to_numpy()  # 保存原始关系值。

    node_types = pd.read_csv(nodetype_csv)  # 读取 thgl-software_nodetype.csv。
    required_node_cols = {"node_id", "type"}  # 节点类型表必须包含的列。
    missing = required_node_cols.difference(node_types.columns)  # 检查缺失列。
    if missing:  # 如果缺列。
        raise ValueError(f"missing columns in {nodetype_csv}: {sorted(missing)}")  # 立即报错。

    type_lookup = {}  # 原始节点 id -> 节点类型 的查找表。
    for row in node_types.itertuples(index=False):  # 遍历节点类型表。
        type_lookup[row.node_id] = int(row.type)  # 按原始类型保存一次。
        type_lookup[str(row.node_id)] = int(row.type)  # 按字符串形式再保存一次，兼容整数/字符串 id。

    node_type = np.zeros(len(all_nodes), dtype=np.int64)  # 为每个新节点准备类型数组；默认 0。
    missing_type_count = 0  # 统计找不到类型的节点数。
    for new_id, original_id in enumerate(all_nodes.tolist()):  # 遍历连续新 id 和对应原始 id。
        if original_id in type_lookup:  # 优先用原始 id 匹配。
            node_type[new_id] = int(type_lookup[original_id])  # 写入该新节点类型。
        elif str(original_id) in type_lookup:  # 再尝试字符串 id 匹配。
            node_type[new_id] = int(type_lookup[str(original_id)])  # 写入该新节点类型。
        else:  # 如果节点类型表没有该节点。
            missing_type_count += 1  # 累计缺失数量。
            node_type[new_id] = 0  # 缺失类型回退为 0。

    node_feat = build_degree_features(len(all_nodes), src, dst, ts, node_type, dim=128)  # 构造节点结构特征：度、时间范围、类型 one-hot，补到 128 维。
    node_multilabel = build_default_multilabel(len(all_nodes), src, dst, ts)  # 构造默认 4 维节点多标签，用于 node_multilabel 任务。
    node_multiclass = node_type.copy()  # 节点多分类标签直接采用节点类型。

    return {  # 返回 prepare_dataset.to_unified 所需的中间字典。
        "name": dataset_name,  # 数据集名：thgl_software。
        "src": src,  # 连续源节点 id。
        "dst": dst,  # 连续目标节点 id。
        "ts": ts,  # float32 时间戳。
        "rel": rel,  # 连续关系 id。
        "node_type": node_type,  # 每个连续节点 id 的类型。
        "node_feat": node_feat,  # 节点特征矩阵。
        "node_multiclass": node_multiclass,  # 节点类型分类标签。
        "node_multiclass_node_ids": np.arange(len(all_nodes), dtype=np.int64),  # 多分类标签覆盖所有节点。
        "node_multilabel": node_multilabel,  # 默认节点多标签。
        "node_multilabel_node_ids": np.arange(len(all_nodes), dtype=np.int64),  # 多标签覆盖所有节点。
        "meta_extra": {  # 写入统一图 meta.extra，便于追溯。
            "source": "TGB2.0",  # 标记来源为 TGB2.0。
            "original_edgelist_csv": str(edgelist_csv),  # 记录原始边 CSV 路径。
            "original_nodetype_csv": str(nodetype_csv),  # 记录原始节点类型 CSV 路径。
            "max_edges_applied": int(max_edges),  # 记录是否截断；thgl-software 使用 -1。
            "num_original_rows_loaded": int(len(edges)),  # 实际导入的边行数。
            "num_unique_relations": int(len(np.unique(rel))),  # 连续关系 id 数量。
            "num_unique_node_types": int(len(np.unique(node_type))),  # 节点类型数量。
            "missing_node_type_count": int(missing_type_count),  # 缺失节点类型数量。
            "original_node_ids_sample": original_node_ids[:10].tolist(),  # 原始节点 id 样例。
            "original_relation_values_sample": rel_original[:10].tolist() if hasattr(rel_original, "tolist") else list(rel_original[:10]),  # 原始关系值样例。
        },
    }


def main() -> None:  # 命令行入口。
    ap = argparse.ArgumentParser()  # 创建参数解析器。
    ap.add_argument("--dataset-name", required=True, type=str)  # 数据集名，例如 thgl_software。
    ap.add_argument("--edgelist-csv", required=True, type=str)  # TGB2 边 CSV 路径。
    ap.add_argument("--nodetype-csv", required=True, type=str)  # TGB2 节点类型 CSV 路径。
    ap.add_argument("--out-pt", required=True, type=str)  # 输出 unified .pt 路径。
    ap.add_argument("--max-edges", type=int, default=120000)  # 默认截断 12 万；传 -1 则全量导入。
    args = ap.parse_args()  # 解析命令行参数。

    dataset_obj = load_thg_dataset(  # 读取 CSV 并转换成中间统一字典。
        edgelist_csv=Path(args.edgelist_csv),  # 边表 Path。
        nodetype_csv=Path(args.nodetype_csv),  # 节点类型表 Path。
        dataset_name=args.dataset_name,  # 数据集名称。
        max_edges=int(args.max_edges),  # 截断参数。
    )
    out_pt = Path(args.out_pt)  # 输出路径转 Path。
    to_unified(dataset_obj, out_pt)  # 转成项目统一 `.pt` schema 并保存。
    print(out_pt)  # 打印输出路径，供日志记录。


if __name__ == "__main__":  # 只有直接执行脚本时才进入 main。
    main()  # 启动导入流程。
```

## 4. unified 保存函数与字段承接

原始位置：`/home/xuxj/2026/dyn_relation_memory/scripts/datasets/prepare_dataset.py:1533-1597`

```python
def to_unified(dataset_obj: Dict, out_pt: Path):  # 把 import_tgb2_thg.py 返回的中间字典保存成统一图对象。
    src = dataset_obj["src"].astype(np.int64)  # 源节点数组转 int64。
    dst = dataset_obj["dst"].astype(np.int64)  # 目标节点数组转 int64。
    ts = dataset_obj["ts"].astype(np.float32)  # 时间戳数组转 float32。
    rel = dataset_obj["rel"].astype(np.int64)  # 关系 id 数组转 int64。

    order = np.argsort(ts)  # 按时间排序，确保统一图边序是时间升序。
    src = src[order]  # 同步重排 src。
    dst = dst[order]  # 同步重排 dst。
    ts = ts[order]  # 同步重排时间。
    rel = rel[order]  # 同步重排关系。

    num_nodes = dataset_obj["node_feat"].shape[0]  # 节点数来自节点特征矩阵行数。
    num_rels = int(rel.max()) + 1 if rel.size > 0 else 1  # 关系数为最大连续 id + 1。

    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)  # 归一化时间作为边特征第 1 维。
    rel_norm = rel / max(1, num_rels - 1)  # 归一化关系 id 作为边特征第 2 维。
    edge_feat = np.stack([ts_norm, rel_norm], axis=1).astype(np.float32)  # 组合成 [E,2] 边特征。

    tr, va, te = temporal_split(ts)  # 生成一个全图时间 split；task view 阶段会重新按目标关系构建 split。

    mc_node_ids = dataset_obj.get("node_multiclass_node_ids", np.arange(num_nodes, dtype=np.int64))  # 多分类节点 id，默认全节点。
    mc_labels = dataset_obj["node_multiclass"]  # 多分类标签。
    ml_node_ids = dataset_obj.get("node_multilabel_node_ids", np.arange(num_nodes, dtype=np.int64))  # 多标签节点 id，默认全节点。
    ml_labels = dataset_obj["node_multilabel"]  # 多标签标签矩阵。

    obj = {  # 统一图对象；后续 build_paper_task_views.py 用这个 schema。
        "num_nodes": int(num_nodes),  # 节点总数。
        "num_rels": int(num_rels),  # 关系总数。
        "num_node_types": int(dataset_obj["node_type"].max()) + 1,  # 节点类型数。
        "node_feat": torch.from_numpy(dataset_obj["node_feat"].astype(np.float32)),  # 节点特征 tensor。
        "node_type": torch.from_numpy(dataset_obj["node_type"].astype(np.int64)),  # 节点类型 tensor。
        "edges": {  # 全部边字段。
            "src": torch.from_numpy(src),  # 源节点 tensor。
            "dst": torch.from_numpy(dst),  # 目标节点 tensor。
            "ts": torch.from_numpy(ts),  # 时间戳 tensor。
            "rel": torch.from_numpy(rel),  # 关系 tensor。
            "edge_feat": torch.from_numpy(edge_feat),  # 边特征 tensor。
            "edge_label": torch.ones(src.shape[0], dtype=torch.float32),  # unified 阶段所有边都是正样本。
        },
        "split": {  # 全图时间 split。
            "train": torch.from_numpy(tr.astype(np.int64)),  # train 边索引。
            "val": torch.from_numpy(va.astype(np.int64)),  # val 边索引。
            "test": torch.from_numpy(te.astype(np.int64)),  # test 边索引。
        },
        "node_multiclass": {  # 节点多分类任务标签。
            "node_ids": torch.from_numpy(np.asarray(mc_node_ids, dtype=np.int64)),  # 有标签节点。
            "labels": torch.from_numpy(np.asarray(mc_labels, dtype=np.int64)),  # 分类标签。
        },
        "node_multilabel": {  # 节点多标签任务标签。
            "node_ids": torch.from_numpy(np.asarray(ml_node_ids, dtype=np.int64)),  # 有标签节点。
            "labels": torch.from_numpy(np.asarray(ml_labels, dtype=np.float32)),  # 多标签矩阵。
        },
        "meta": {  # 元数据。
            "dataset_name": dataset_obj["name"],  # thgl_software。
            "num_edges": int(src.shape[0]),  # 边数。
            "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},  # 默认全图 split 比例。
            "extra": dataset_obj.get("meta_extra", {}),  # 导入阶段写入的 TGB2 路径和统计信息。
        },
    }

    out_pt.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    torch.save(obj, out_pt)  # 保存为 unified `.pt`。
    return obj  # 返回对象，便于调用方继续使用。
```

## 5. task view 生成核心函数逐行注释

原始位置：`/home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py:10-352`

### 5.1 排序与切分工具

```python
def sort_old_indices(indices: np.ndarray, ts: np.ndarray) -> np.ndarray:  # 把旧边索引清洗并按时间排序。
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)  # 输入转成一维 int64。
    idx = idx[(idx >= 0) & (idx < ts.shape[0])]  # 丢弃越界索引。
    if idx.size == 0:  # 空输入直接返回空数组。
        return np.array([], dtype=np.int64)
    idx = np.unique(idx)  # 去重。
    order = np.lexsort((idx, ts[idx]))  # 先按 ts 排序，同时间按旧索引排序，保证稳定。
    return idx[order].astype(np.int64)  # 返回排序后的旧边索引。


def temporal_partition_sorted(sorted_old: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15):  # 对已排序边索引按比例切 train/val/test。
    sorted_old = np.asarray(sorted_old, dtype=np.int64).reshape(-1)  # 保证一维 int64。
    if sorted_old.size == 0:  # 空目标边。
        z = np.array([], dtype=np.int64)  # 统一空数组。
        return z, z, z  # 三个 split 都为空。
    n = sorted_old.size  # 目标边数量。
    n_train = max(1, int(n * train_ratio))  # 至少 1 条 train。
    n_val = max(1, int(n * val_ratio))  # 至少 1 条 val。
    if n_train + n_val >= n:  # 如果 train+val 挤占 test。
        n_train = max(1, n - 2)  # 给 val/test 留空间。
        n_val = max(1, n - n_train - 1)  # 保证 test 至少可有 1 条。
    tr = sorted_old[:n_train]  # 前 70% 作为 train。
    va = sorted_old[n_train : n_train + n_val]  # 中间 15% 作为 val。
    te = sorted_old[n_train + n_val :]  # 后面作为 test。
    if te.size == 0 and va.size > 0:  # 如果 test 为空。
        te = va[-1:]  # 从 val 尾部挪一条到 test。
        va = va[:-1]  # val 删除被挪走的边。
    if va.size == 0 and tr.size > 1:  # 如果 val 为空。
        va = tr[-1:]  # 从 train 尾部挪一条到 val。
        tr = tr[:-1]  # train 删除被挪走的边。
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)  # 返回三个 split。
```

### 5.2 inductive 目标关系切分

```python
def inductive_partition_target(target_old: np.ndarray, src: np.ndarray, ts: np.ndarray, rng: np.random.RandomState):  # 为目标关系构造 source-node holdout 归纳切分。
    target_old = sort_old_indices(target_old, ts)  # 目标关系边按时间排序。
    if target_old.size == 0:  # 没有目标边。
        z = np.array([], dtype=np.int64)
        return z, z, z, np.array([], dtype=np.int64)  # 返回空 split 和空 unseen 节点。

    target_src = src[target_old]  # 取目标边的源节点。
    src_nodes, src_counts = np.unique(target_src, return_counts=True)  # 统计每个源节点贡献多少目标边。
    if src_nodes.size == 0:  # 极端空场景。
        z = np.array([], dtype=np.int64)
        return z, z, z, np.array([], dtype=np.int64)

    target_holdout_edges = max(1, int(0.10 * target_old.size))  # 目标是 holdout 约 10% 目标边。
    min_base_edges = max(64, int(0.10 * target_old.size))  # 剩余 base 至少保留足够监督边。
    perm = rng.permutation(src_nodes.size)  # 随机打乱源节点候选；调用处固定 RandomState(42)。
    unseen_nodes = []  # 将被归纳测试持出的源节点。
    heldout_edges = 0  # 当前已持出的目标边数。
    for idx in perm.tolist():  # 遍历随机节点顺序。
        if heldout_edges >= target_holdout_edges:  # 持出边数达到目标。
            break  # 停止选节点。
        node = int(src_nodes[idx])  # 候选源节点 id。
        count = int(src_counts[idx])  # 该源节点关联的目标边数。
        remain_after = int(target_old.size - (heldout_edges + count))  # 如果持出该节点，还剩多少目标边给训练/验证。
        if remain_after < min_base_edges:  # 如果剩余监督太少。
            continue  # 跳过该源节点。
        unseen_nodes.append(node)  # 加入 unseen 源节点。
        heldout_edges += count  # 累加持出边数。

    if len(unseen_nodes) == 0:  # 如果随机选择没有选出合法 unseen 节点。
        smallest = int(src_nodes[np.argmin(src_counts)])  # 选目标边最少的源节点作为 fallback。
        if int(target_old.size - src_counts.min()) >= min_base_edges:  # 仍需保证 base 边足够。
            unseen_nodes = [smallest]  # 使用这个节点作为 unseen。

    unseen = np.sort(np.asarray(unseen_nodes, dtype=np.int64))  # unseen 节点排序。
    test_mask = np.isin(target_src, unseen)  # 源节点属于 unseen 的目标边进入 test。
    test_old = target_old[test_mask]  # 归纳测试目标边。
    base_old = target_old[~test_mask]  # 剩余目标边用于 train/val。
    if test_old.size == 0 or base_old.size < min_base_edges:  # 如果归纳持出失败或 base 监督过少。
        base_old = target_old  # 回退为全部目标边参与时间切分。
        test_old = np.array([], dtype=np.int64)  # 清空归纳 test。
        unseen = np.array([], dtype=np.int64)  # 清空 unseen 节点。
    base_old = sort_old_indices(base_old, ts)  # base 目标边按时间排序。
    cut = min(max(1, int(0.80 * base_old.size)), max(1, base_old.size - 1))  # base 前 80% 训练，后 20% 验证。
    train_old = base_old[:cut]  # 归纳 train 目标边。
    val_old = base_old[cut:]  # 归纳 val 目标边。
    if val_old.size < max(1, min(64, base_old.size // 20)):  # 如果验证窗口太小。
        train_old, val_old, test_old = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # 回退到稳定时间切分。
        unseen = np.array([], dtype=np.int64)  # 因为回退为时间切分，所以没有 unseen 源节点。
    return train_old, val_old, test_old.astype(np.int64), unseen  # 返回目标关系上的归纳 split 和 unseen 节点。
```

这段正对应 README 和日志中提到的修复：早期 inductive split 可能在过滤 unseen 节点后得到空验证集；现在当验证窗口过小，会回退到稳定 temporal split。

### 5.3 task view 组装函数

```python
def build_edge_view(
    obj: Dict,  # unified `.pt` 读出的统一图对象。
    out_name: str,  # 输出 task view 名称，例如 thgl_software_edge_transductive。
    target_old: np.ndarray,  # 目标关系的旧边索引全集。
    train_target_old: np.ndarray,  # 目标关系 train 旧边索引。
    val_target_old: np.ndarray,  # 目标关系 val 旧边索引。
    test_target_old: np.ndarray,  # 目标关系 test 旧边索引。
    context_keep_mask: np.ndarray = None,  # 上下文边过滤 mask；inductive 用它去除 touching unseen nodes 的边。
    extra_info: Dict = None,  # 写入 meta.extra.context_protocol 的附加信息。
) -> Dict:
    src_all = obj["edges"]["src"].numpy().astype(np.int64)  # 取全图 src。
    dst_all = obj["edges"]["dst"].numpy().astype(np.int64)  # 取全图 dst。
    ts_all = obj["edges"]["ts"].numpy().astype(np.float32)  # 取全图时间。
    rel_all = obj["edges"]["rel"].numpy().astype(np.int64)  # 取全图关系。
    raw_num_rels = int(obj.get("num_rels", int(rel_all.max()) + 1 if rel_all.size > 0 else 1))  # 保留原始关系总数，避免 task view 压缩关系空间。

    all_old_sorted = sort_old_indices(np.arange(ts_all.shape[0], dtype=np.int64), ts_all)  # 所有旧边按时间排序。
    keep_mask = np.ones(ts_all.shape[0], dtype=bool) if context_keep_mask is None else np.asarray(context_keep_mask, dtype=bool)  # 默认保留全部上下文边。
    keep_mask = keep_mask.reshape(-1)  # 保证一维。
    if keep_mask.shape[0] != ts_all.shape[0]:  # mask 长度必须等于全图边数。
        raise ValueError("context_keep_mask length mismatch")  # 不一致则报错。

    train_cut = float(ts_all[train_target_old[-1]]) if train_target_old.size > 0 else float(ts_all[all_old_sorted[0]])  # 训练窗口截止时间取目标 train 最后一条边时间。
    val_cut = float(ts_all[val_target_old[-1]]) if val_target_old.size > 0 else train_cut  # 验证窗口截止时间取目标 val 最后一条边时间。

    train_old = all_old_sorted[keep_mask[all_old_sorted] & (ts_all[all_old_sorted] <= train_cut)]  # train 窗口包含所有关系的上下文边。
    val_old = all_old_sorted[keep_mask[all_old_sorted] & (ts_all[all_old_sorted] > train_cut) & (ts_all[all_old_sorted] <= val_cut)]  # val 窗口包含所有关系上下文边。
    assigned = np.zeros(ts_all.shape[0], dtype=bool)  # 标记已分配到 train/val 的旧边。
    assigned[train_old] = True  # 标记 train。
    assigned[val_old] = True  # 标记 val。
    test_old = all_old_sorted[~assigned[all_old_sorted]]  # 其它边进入 test 窗口。

    explicit_val_old = sort_old_indices(val_target_old, ts_all)  # 显式目标 val 边。
    explicit_test_old = sort_old_indices(test_target_old, ts_all)  # 显式目标 test 边。
    if explicit_val_old.size > 0:  # 如果有显式 val 目标边。
        train_old = remove_indices(train_old, explicit_val_old, ts_all)  # 从 train 中移除，避免泄漏。
        test_old = remove_indices(test_old, explicit_val_old, ts_all)  # 从 test 中移除，避免重复。
        val_old = sort_old_indices(np.concatenate([val_old, explicit_val_old]), ts_all)  # 强制加入 val。
    if explicit_test_old.size > 0:  # 如果有显式 test 目标边。
        train_old = remove_indices(train_old, explicit_test_old, ts_all)  # 从 train 中移除。
        val_old = remove_indices(val_old, explicit_test_old, ts_all)  # 从 val 中移除。
        test_old = sort_old_indices(np.concatenate([test_old, explicit_test_old]), ts_all)  # 强制加入 test。

    if val_old.size == 0 and np.asarray(val_target_old).size > 0:  # inductive 上下文过滤后 val 为空时触发保护。
        val_old = sort_old_indices(val_target_old, ts_all)  # 回退到显式目标 val。
        train_old = remove_indices(train_old, val_old, ts_all)  # 避免 val 与 train 重叠。
        test_old = remove_indices(test_old, val_old, ts_all)  # 避免 val 与 test 重叠。
    if test_old.size == 0 and np.asarray(test_target_old).size > 0:  # 如果 test 为空但显式目标 test 存在。
        test_old = sort_old_indices(test_target_old, ts_all)  # 回退到显式目标 test。
        train_old = remove_indices(train_old, test_old, ts_all)  # 避免 test 与 train 重叠。
        val_old = remove_indices(val_old, test_old, ts_all)  # 避免 test 与 val 重叠。

    selected_old = np.concatenate([train_old, val_old, test_old], axis=0).astype(np.int64)  # 当前 task view 保留的旧边全集。
    selected_old = sort_old_indices(selected_old, ts_all)  # 统一按时间排序。

    old_to_new = np.full((ts_all.shape[0],), -1, dtype=np.int64)  # 旧边索引 -> task view 新边索引。
    old_to_new[selected_old] = np.arange(selected_old.shape[0], dtype=np.int64)  # 给被选中的边分配新索引。

    src = src_all[selected_old]  # task view src。
    dst = dst_all[selected_old]  # task view dst。
    ts = ts_all[selected_old]  # task view 时间。
    rel = rel_all[selected_old]  # task view 关系，保留全局关系 id。
    rel_vals = sorted(np.unique(rel).tolist()) if rel.size > 0 else [0]  # 记录该 view 中出现的关系。
    num_rels = max(1, raw_num_rels)  # 关系总数保持 unified 全局关系数。

    supervised_old = sort_old_indices(np.concatenate([train_target_old, val_target_old, test_target_old], axis=0), ts_all)  # 只有目标关系边是监督边。
    supervised_mask = np.isin(selected_old, supervised_old)  # 标记 selected_old 中哪些是目标监督边。
    edge_label = supervised_mask.astype(np.float32)  # 监督边 label=1，上下文边 label=0。

    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6) if ts.size > 0 else np.array([], dtype=np.float32)  # 重新归一化当前 view 时间。
    rel_norm = rel / max(1, num_rels - 1) if rel.size > 0 else np.array([], dtype=np.float32)  # 关系 id 归一化但不重编号。
    edge_feat = np.stack([ts_norm, rel_norm], axis=1).astype(np.float32) if ts.size > 0 else np.zeros((0, 2), dtype=np.float32)  # 当前 view 边特征。

    train_new = map_old_to_new(train_old, old_to_new)  # train 旧边索引映射成新边索引。
    val_new = map_old_to_new(val_old, old_to_new)  # val 旧边索引映射成新边索引。
    test_new = map_old_to_new(test_old, old_to_new)  # test 旧边索引映射成新边索引。

    meta = dict(obj.get("meta", {}))  # 复制 unified meta。
    extra = dict(meta.get("extra", {}))  # 复制 meta.extra。
    extra.pop("task_splits", None)  # 删除旧 task_splits，避免污染新 view。
    extra["task_view_name"] = out_name  # 写入当前 view 名。
    extra["source_rel_ids"] = rel_vals  # 写入当前 view 出现的关系 id。
    extra["context_protocol"] = {  # 记录上下文协议。
        "enabled": True,  # task view 使用 full heterogeneous context。
        "supervision_mode": "edge_label_mask",  # 通过 edge_label 区分监督边和上下文边。
        "preserve_global_rel_ids": True,  # 关系 id 保持全局编号。
        "selected_edges": int(selected_old.size),  # view 总边数。
        "supervised_edges": int(edge_label.sum()),  # 监督目标边数。
        "train_window_edges": int(train_new.size),  # train 窗口边数，含上下文。
        "val_window_edges": int(val_new.size),  # val 窗口边数，含上下文。
        "test_window_edges": int(test_new.size),  # test 窗口边数，含上下文。
        "target_original_rel_ids": sorted(np.unique(rel_all[supervised_old]).tolist()) if supervised_old.size > 0 else [],  # 目标关系 id。
        "train_supervised_edges": int(np.isin(train_old, supervised_old).sum()),  # train 监督边数。
        "val_supervised_edges": int(np.isin(val_old, supervised_old).sum()),  # val 监督边数。
        "test_supervised_edges": int(np.isin(test_old, supervised_old).sum()),  # test 监督边数。
    }
    if isinstance(extra_info, dict):  # 如果调用方传入额外协议信息。
        extra["context_protocol"].update(extra_info)  # 合并到 context_protocol。
    meta["dataset_name"] = out_name  # 当前 task view 的数据集名。
    meta["extra"] = extra  # 写回 meta.extra。

    return {  # 返回 task view 对象。
        "num_nodes": int(obj["num_nodes"]),  # 节点数保持全图节点数。
        "num_rels": int(num_rels),  # 关系数保持全局关系数。
        "num_node_types": int(obj.get("num_node_types", 1)),  # 节点类型数。
        "node_feat": obj["node_feat"],  # 复用全图节点特征。
        "node_type": obj["node_type"],  # 复用全图节点类型。
        "edges": {
            "src": torch.from_numpy(src),  # view 边 src。
            "dst": torch.from_numpy(dst),  # view 边 dst。
            "ts": torch.from_numpy(ts),  # view 边时间。
            "rel": torch.from_numpy(rel),  # view 边关系。
            "edge_feat": torch.from_numpy(edge_feat),  # view 边特征。
            "edge_label": torch.from_numpy(edge_label),  # 监督 mask label。
        },
        "split": {
            "train": torch.from_numpy(train_new),  # train 新边索引。
            "val": torch.from_numpy(val_new),  # val 新边索引。
            "test": torch.from_numpy(test_new),  # test 新边索引。
        },
        "node_multiclass": obj.get("node_multiclass"),  # 保留节点分类标签。
        "node_multilabel": obj.get("node_multilabel"),  # 保留节点多标签。
        "meta": meta,  # 写入 task view meta。
    }
```

### 5.4 `build()` 总入口和保存逻辑

```python
def choose_target_relation(rel: np.ndarray, target_rel: int = None) -> int:  # 选择边预测目标关系。
    if target_rel is not None and int(target_rel) >= 0:  # 如果命令行显式传入 --target-rel。
        return int(target_rel)  # 使用显式关系 id。
    vals, cnt = np.unique(rel, return_counts=True)  # 否则统计每个关系出现次数。
    if vals.size == 0:  # 没有关系时的保护。
        return 0
    best_rel = int(vals[np.argmax(cnt)])  # 默认选择出现次数最多的关系；日志里 thgl_software 对应 rel_6。
    return best_rel  # 返回目标关系 id。


def build(input_pt: Path, out_dir: Path, prefix: str, target_rel: int = None):  # task view 生成总入口。
    obj = torch.load(input_pt, map_location="cpu")  # 读取 unified `.pt` 到 CPU。
    rel = obj["edges"]["rel"].numpy().astype(np.int64)  # 取全图关系数组。
    ts = obj["edges"]["ts"].numpy().astype(np.float32)  # 取全图时间数组。
    src = obj["edges"]["src"].numpy().astype(np.int64)  # 取全图源节点数组。
    dst = obj["edges"]["dst"].numpy().astype(np.int64)  # 取全图目标节点数组。

    target_rel_id = choose_target_relation(rel, target_rel=target_rel)  # 选择目标关系。
    target_old = sort_old_indices(np.where(rel == int(target_rel_id))[0].astype(np.int64), ts)  # 找出目标关系边旧索引并按时间排序。
    if target_old.size == 0:  # 如果该关系没有边。
        raise RuntimeError(f"no edges found for target relation id={target_rel_id}")  # 报错阻止生成空任务。

    trans_tr_old, trans_va_old, trans_te_old = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)  # transductive 直接按目标关系时间切分。
    trans_view = build_edge_view(  # 组装 transductive task view。
        obj=obj,  # unified 图对象。
        out_name=f"{prefix}_edge_transductive",  # thgl_software_edge_transductive。
        target_old=target_old,  # 目标关系全集。
        train_target_old=trans_tr_old,  # transductive train 目标边。
        val_target_old=trans_va_old,  # transductive val 目标边。
        test_target_old=trans_te_old,  # transductive test 目标边。
        extra_info={
            "target_original_rel_id": int(target_rel_id),  # 写入目标关系 id。
            "protocol": "full_hetero_context_transductive",  # 标记 transductive 协议。
        },
    )

    ind_tr_old, ind_va_old, ind_te_old, unseen_nodes = inductive_partition_target(  # 为 inductive 生成目标边 split。
        target_old=target_old,  # 目标关系边。
        src=src,  # 全图源节点。
        ts=ts,  # 全图时间。
        rng=np.random.RandomState(42),  # 固定随机种子，保证可复现。
    )
    keep_mask = ~(np.isin(src, unseen_nodes) | np.isin(dst, unseen_nodes))  # inductive 上下文过滤：去掉触及 unseen 节点的所有边。
    ind_view = build_edge_view(  # 组装 inductive task view。
        obj=obj,  # unified 图对象。
        out_name=f"{prefix}_edge_inductive",  # thgl_software_edge_inductive。
        target_old=target_old,  # 目标关系全集。
        train_target_old=ind_tr_old,  # inductive train 目标边。
        val_target_old=ind_va_old,  # inductive val 目标边。
        test_target_old=ind_te_old,  # inductive test 目标边。
        context_keep_mask=keep_mask,  # 应用 unseen 节点上下文过滤。
        extra_info={
            "target_original_rel_id": int(target_rel_id),  # 写入目标关系 id。
            "protocol": "full_hetero_context_inductive",  # 标记 inductive 协议。
            "unseen_source_node_ids": unseen_nodes.tolist(),  # 记录 unseen 源节点 id。
        },
    )

    node_view = build_sorted_full_view(obj, out_name=f"{prefix}_node_multilabel")  # 额外生成节点多标签 view，使用全图时间切分。

    out_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    p1 = out_dir / f"{prefix}_edge_transductive.pt"  # transductive 输出路径。
    p2 = out_dir / f"{prefix}_edge_inductive.pt"  # inductive 输出路径。
    p3 = out_dir / f"{prefix}_node_multilabel.pt"  # node multilabel 输出路径。
    torch.save(trans_view, p1)  # 保存 transductive task view。
    torch.save(ind_view, p2)  # 保存 inductive task view。
    torch.save(node_view, p3)  # 保存 node task view。
    return [p1, p2, p3]  # 返回生成路径列表。


def main():  # 命令行入口。
    ap = argparse.ArgumentParser()  # 创建参数解析器。
    ap.add_argument("--input", required=True, type=str)  # unified `.pt` 输入。
    ap.add_argument("--out-dir", required=True, type=str)  # task view 输出目录。
    ap.add_argument("--name-prefix", required=True, type=str)  # 输出文件名前缀：thgl_software。
    ap.add_argument("--target-rel", type=int, default=-1)  # 可选目标关系；默认 -1 表示自动选最高频关系。
    args = ap.parse_args()  # 解析命令行。

    outs = build(  # 调用核心 build。
        input_pt=Path(args.input),  # 输入路径。
        out_dir=Path(args.out_dir),  # 输出目录。
        prefix=args.name_prefix,  # 文件名前缀。
        target_rel=None if args.target_rel < 0 else int(args.target_rel),  # -1 转成 None，触发自动目标关系选择。
    )
    print("generated:")  # 打印生成提示。
    for p in outs:  # 遍历输出文件。
        print(p)  # 打印每个输出路径。


if __name__ == "__main__":  # 直接执行脚本时。
    main()  # 启动 task view 构建。
```

## 6. 后续 task view 生成入口

原始位置：`/home/xuxj/2026/dyn_relation_memory/scripts/experiments/edge_benchmark_common.py:126-196`

```python
def ensure_thgl_tasks(project_root: Path, repo_root: Path, python_bin: str, env: Dict[str, str]) -> None:  # 实验流水线确保 THGL 任务存在。
    processed_root = resolve_processed_root(project_root)  # 通常是 project_root/data/processed。
    task_root = processed_root / "tgb2_tasks"  # THGL task view 输出目录。
    task_root.mkdir(parents=True, exist_ok=True)  # 确保目录存在。
    import_script = project_root / "scripts" / "datasets" / "import_tgb2_thg.py"  # TGB2 CSV 导入脚本。
    task_script = project_root / "scripts" / "datasets" / "build_paper_task_views.py"  # task view 构建脚本。

    specs = {  # THGL 数据集配置。
        "thgl_github": { ... },  # 另一套 THGL 数据集，此处省略。
        "thgl_software": {  # thgl-software 的实际预处理配置。
            "dataset_name": "thgl_software",  # unified 内部数据集名。
            "edgelist": repo_root / "datasets" / "tgb2" / "thgl-software" / "thgl-software_edgelist.csv",  # 原始边 CSV。
            "nodetype": repo_root / "datasets" / "tgb2" / "thgl-software" / "thgl-software_nodetype.csv",  # 原始节点类型 CSV。
            "unified": processed_root / "thgl_software_unified.pt",  # unified 输出路径。
        },
    }

    for dataset, spec in specs.items():  # 遍历 thgl_github 和 thgl_software。
        trans_pt = task_root / f"{dataset}_edge_transductive.pt"  # 期望的 transductive 文件。
        ind_pt = task_root / f"{dataset}_edge_inductive.pt"  # 期望的 inductive 文件。
        if not spec["unified"].exists():  # 如果 unified 不存在。
            cmd = [
                python_bin, str(import_script), "--dataset-name", str(spec["dataset_name"]),
                "--edgelist-csv", str(spec["edgelist"]), "--nodetype-csv", str(spec["nodetype"]),
                "--out-pt", str(spec["unified"]), "--max-edges", "-1",
            ]  # 拼出 import_tgb2_thg.py 全量导入命令。
            run_sync(cmd, cwd=project_root, env=env)  # 同步执行导入命令；这是实际会跑任务的地方。
            append_dataset_log(project_root, dataset, [...])  # 把命令和输出写入对应 process log。
        if not (trans_pt.exists() and ind_pt.exists()):  # 如果两个 edge task view 不齐。
            cmd = [
                python_bin, str(task_script), "--input", str(spec["unified"]),
                "--out-dir", str(task_root), "--name-prefix", dataset,
            ]  # 拼出 build_paper_task_views.py 命令。
            run_sync(cmd, cwd=project_root, env=env)  # 同步执行 task view 生成。
            append_dataset_log(project_root, dataset, [...])  # 记录 task view 生成命令和输出。
```

原始位置：`/home/xuxj/2026/dyn_relation_memory/scripts/experiments/edge_benchmark_common.py:318-333`

```python
TaskSpec(
    dataset="thgl_software",  # 实验系统里的数据集 key。
    task_name="thgl_software_edge_inductive",  # 归纳边预测任务名。
    task_type="edge_inductive",  # 任务类型。
    relation_target="dominant_relation",  # 目标关系描述：最高频关系。
    data_path=thgl_root / "thgl_software_edge_inductive.pt",  # 后续训练读取的 inductive task view。
    sthn_dataset_slot="thgl-software",  # 对接 TGB2/STHN 官方槽位名。
),
TaskSpec(
    dataset="thgl_software",  # 同一数据集。
    task_name="thgl_software_edge_transductive",  # 传导边预测任务名。
    task_type="edge_transductive",  # 任务类型。
    relation_target="dominant_relation",  # 目标关系仍是最高频关系。
    data_path=thgl_root / "thgl_software_edge_transductive.pt",  # 后续训练读取的 transductive task view。
    sthn_dataset_slot="thgl-software",  # 对接官方 TGB2/STHN 名称。
),
```

平台后端入口补充，原始位置：`/home/xuxj/2026/dyn_relation_memory/platform/backend/core/services/job_builder.py:64-81`

```python
if mode in {'build_tasks', 'prepare_and_build_tasks'}:  # 平台预处理 job 需要生成 task views。
    builder = dataset_cfg['builder']  # 从 DATASET_CONFIG 读取 builder 类型。
    if builder == 'paper':  # paper 类数据集走 build_paper_task_views.py。
        cmd = [
            'python', 'scripts/datasets/build_paper_task_views.py',
            '--input', str(unified_path),
            '--out-dir', str(task_dir),
            '--name-prefix', dataset_key,
            '--target-rel', str(int(payload.get('target_rel', -1))),
        ]  # 生成 paper-style edge/node task views 的命令。
    else:  # 非 paper 类数据集。
        cmd = [
            'python', 'scripts/datasets/build_task_views.py',
            '--input', str(unified_path),
            '--out-dir', str(task_dir),
            '--name-prefix', dataset_key,
        ]  # 走 patent/general task builder。
    commands.append(cmd)  # 加入 job 脚本。
```

注意：`platform/backend/core/services/registry.py` 当前只注册了 `arxiv`、`dblp`、`amazon_movies`、`kg4ai`、`raw_cn`、`raw_us`，没有 `thgl_software`。因此这个平台入口说明的是通用 task view 生成逻辑，不是当前 thgl-software 已执行日志里的入口。

## 7. 重点数据流和语义

- `head/tail` -> `src/dst`：原始节点 id 被重编号为连续整数，新 id 覆盖所有出现在边表中的节点。
- `relation_type` -> `rel`：原始关系值被重编号为连续关系 id；默认目标关系由 `choose_target_relation()` 选择最高频关系。thgl_software 运行日志里的 `relation_target: rel_6` 说明该任务目标关系为 `6`。
- `nodetype.csv` -> `node_type`：节点类型按原始 id 查表后对齐到新节点 id；缺失类型回退为 `0` 并记录 `missing_node_type_count`。
- `node_feat`：由入度、出度、总度、首次/末次出现时间、时间跨度、节点类型 one-hot 组成，不足 128 维补零。
- `unified edge_label`：导入阶段全部为 `1`，表示全图正边。
- `task view edge_label`：只有目标关系 split 中的监督边为 `1`，保留下来的其他异构上下文边为 `0`。
- `transductive view`：目标关系边按时间 70/15/15 切分；上下文保留所有关系，按目标边窗口时间划入 train/val/test。
- `inductive view`：优先按源节点 holdout 构造 unseen source nodes；上下文边会过滤掉触及 unseen 节点的边。若导致验证窗口太小或为空，则回退到稳定时间切分。
- `num_rels`：task view 保留 unified 的全局关系数，不把当前 view 中出现的关系重新压缩编号。
- `split.train/val/test`：保存的是 task view 内的新边索引，不是 unified 旧边索引。

## 8. 输出文件

最终预处理产物：

- `/home/xuxj/2026/dyn_relation_memory/data/processed/thgl_software_unified.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_software_edge_transductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_software_edge_inductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_software_node_multilabel.pt`

本注释文件只记录源码定位与数据流，没有运行任何预处理或训练任务。
