#!/usr/bin/env python3
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
try:
    from sklearn.feature_extraction.text import HashingVectorizer
except Exception:  # pragma: no cover
    HashingVectorizer = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

# Workaround for OpenMP runtime conflicts in mixed pandas/torch environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

TEXT_ENCODER = "hybrid_hash"
TEXT_DIM = 64
TEXT_MODEL_NAME = ""
TEXT_BATCH_SIZE = 128
_TEXT_MODEL_CACHE: Dict[str, Any] = {}


def text_encoder_meta() -> Dict[str, Any]:
    return {
        "text_encoder": str(TEXT_ENCODER),
        "text_dim": int(TEXT_DIM),
        "text_model_name": str(TEXT_MODEL_NAME),
        "text_batch_size": int(TEXT_BATCH_SIZE),
    }


def parse_id_list(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\d+", text)


def safe_ts(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.number):
        return series.to_numpy(dtype=np.float32)
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().sum() > 0:
        return (dt.view("int64") // 10**9).fillna(0).to_numpy(dtype=np.float32)
    return np.arange(len(series), dtype=np.float32)


def build_degree_features(num_nodes: int, src: np.ndarray, dst: np.ndarray, ts: np.ndarray, node_type: np.ndarray, dim: int = 64):
    if num_nodes == 0:
        return np.zeros((0, dim), dtype=np.float32)
    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float32)
    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float32)
    deg = in_deg + out_deg

    first_ts = np.full(num_nodes, ts.max() if ts.size > 0 else 0, dtype=np.float32)
    last_ts = np.zeros(num_nodes, dtype=np.float32)
    for u, v, t in zip(src, dst, ts):
        if t < first_ts[u]:
            first_ts[u] = t
        if t < first_ts[v]:
            first_ts[v] = t
        if t > last_ts[u]:
            last_ts[u] = t
        if t > last_ts[v]:
            last_ts[v] = t

    t_span = last_ts - first_ts
    base = np.stack([in_deg, out_deg, deg, first_ts, last_ts, t_span], axis=1)
    base = np.nan_to_num(base)
    base = (base - base.mean(axis=0, keepdims=True)) / (base.std(axis=0, keepdims=True) + 1e-6)

    type_onehot = np.zeros((num_nodes, int(node_type.max()) + 1), dtype=np.float32)
    type_onehot[np.arange(num_nodes), node_type] = 1.0

    feat = np.concatenate([base, type_onehot], axis=1)
    if feat.shape[1] >= dim:
        return feat[:, :dim].astype(np.float32)
    pad = np.zeros((num_nodes, dim - feat.shape[1]), dtype=np.float32)
    return np.concatenate([feat, pad], axis=1).astype(np.float32)


def temporal_split(ts: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    if ts.size < 3:
        idx = np.arange(ts.shape[0])
        if ts.size == 0:
            return idx, idx, idx
        return idx, idx[-1:], idx[-1:]
    val_q = train_ratio
    test_q = train_ratio + val_ratio
    t1 = np.quantile(ts, val_q)
    t2 = np.quantile(ts, test_q)
    idx = np.arange(ts.shape[0])
    train = idx[ts <= t1]
    val = idx[(ts > t1) & (ts <= t2)]
    test = idx[ts > t2]
    if train.size == 0 or val.size == 0 or test.size == 0:
        # Fallback to chronological index-based split for discrete timestamps.
        order = np.argsort(ts)
        n = len(order)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, n - n_train - n_val)
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        train = order[:n_train]
        val = order[n_train : n_train + n_val]
        test = order[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def build_default_multilabel(num_nodes: int, src: np.ndarray, dst: np.ndarray, ts: np.ndarray) -> np.ndarray:
    in_deg = np.bincount(dst, minlength=num_nodes)
    out_deg = np.bincount(src, minlength=num_nodes)
    total = in_deg + out_deg

    def positive_quantile_mask(values: np.ndarray, q: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        mask = np.zeros(values.shape[0], dtype=np.float32)
        active = values > 0
        if not np.any(active):
            return mask
        thr = np.quantile(values[active], q)
        mask[active] = (values[active] >= thr).astype(np.float32)
        return mask

    labels = np.zeros((num_nodes, 4), dtype=np.float32)
    # Use positive-only quantiles so sparse graphs with many isolated nodes do
    # not collapse synthetic labels into all-ones columns.
    labels[:, 0] = positive_quantile_mask(total, 0.75)
    labels[:, 1] = positive_quantile_mask(in_deg, 0.75)
    labels[:, 2] = positive_quantile_mask(out_deg, 0.75)

    node_first = np.full(num_nodes, np.inf, dtype=np.float32)
    for u, v, t in zip(src, dst, ts):
        node_first[u] = min(node_first[u], t)
        node_first[v] = min(node_first[v], t)

    active = np.isfinite(node_first)
    if np.any(active):
        thr = np.quantile(node_first[active], 0.25)
        labels[active, 3] = (node_first[active] <= thr).astype(np.float32)
    return labels


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        l = col.lower()
        for cand in candidates:
            if cand.lower() in l:
                return col
    return None


def split_multi_values(text: Any) -> List[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    s = str(text).strip()
    if not s:
        return []
    parts = re.split(r"[;|,\n\t]+", s)
    out = []
    for p in parts:
        p = p.strip().strip('"').strip("'")
        if p:
            out.append(p)
    return out


def norm_assignee_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "").replace("'", "")
    return s


def parse_publication_ts(value: Any, fallback: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    s = str(value).strip()
    if not s:
        return fallback
    if re.fullmatch(r"\d{8}", s):
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    else:
        dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return fallback
    return float(int(dt.value // 10**9))


def extract_ipc_levels(raw_code: str) -> Tuple[List[str], List[str]]:
    """Return (subgroup_codes, subclass_codes)."""
    subgroup_codes, subclass_codes = [], []
    if not raw_code:
        return subgroup_codes, subclass_codes
    for token in split_multi_values(raw_code):
        t = token.upper().replace(" ", "")
        t = t.replace("－", "-")
        # subgroup pattern: A01B33/00, A01B-1/02, etc.
        m_subgroup = re.search(r"([A-H]\d{2}[A-Z]-?\d+(?:/\d+)?)", t)
        if m_subgroup is None:
            continue
        subgroup = m_subgroup.group(1).replace("-", "")
        subgroup_codes.append(subgroup)
        m_subclass = re.match(r"([A-H]\d{2}[A-Z])", subgroup)
        if m_subclass is not None:
            subclass_codes.append(m_subclass.group(1))
    subgroup_codes = sorted(set(subgroup_codes))
    subclass_codes = sorted(set(subclass_codes))
    return subgroup_codes, subclass_codes


def load_ipc_subclass_names(country: str) -> Dict[str, str]:
    base = Path(__file__).resolve().parents[2] / "vendor/ref/MENPC/PreProcess/IPC"
    names: Dict[str, str] = {}
    if country.upper() == "US":
        fp = base / "US" / "IPCSubClass.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            class_col = find_col(df, ["Class"])
            name_col = find_col(df, ["Name"])
            if class_col and name_col:
                for _, row in df.iterrows():
                    code = str(row[class_col]).strip().upper()
                    desc = str(row[name_col]).strip()
                    if code:
                        names[code] = desc
    else:
        fp = base / "IPCSubClass.csv"
        if fp.exists():
            try:
                df = pd.read_csv(fp, engine="python")
                cn_col = find_col(df, ["Class Name", "ClassName", "Name"])
                if cn_col:
                    for _, row in df.iterrows():
                        raw = str(row[cn_col]).strip()
                        m = re.match(r"([A-H]\d{2}[A-Z])\s+\"?(.*)\"?$", raw)
                        if m is None:
                            m = re.match(r"([A-H]\d{2}[A-Z])\s+(.*)$", raw)
                        if m is not None:
                            names[m.group(1).upper()] = m.group(2).strip()
            except Exception:
                for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    raw = line.strip().strip('"')
                    if not raw or raw.lower().startswith("class"):
                        continue
                    # Format1: A01B "name..."
                    m = re.match(r"([A-H]\d{2}[A-Z])\s+\"?(.*)\"?$", raw)
                    if m is None:
                        # Format2: A01B,name...
                        m = re.match(r"([A-H]\d{2}[A-Z]),\s*(.*)$", raw)
                    if m is not None:
                        names[m.group(1).upper()] = m.group(2).strip().strip('"')
    return names


def _safe_texts(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        s = str(t).strip()
        out.append(s if s else "[EMPTY]")
    return out


def _project_dim(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[1] == dim:
        return x.astype(np.float32)
    if x.shape[1] < dim:
        pad = np.zeros((x.shape[0], dim - x.shape[1]), dtype=np.float32)
        return np.concatenate([x.astype(np.float32), pad], axis=1)
    rng = np.random.RandomState(42)
    w = rng.normal(0.0, 1.0 / np.sqrt(max(1, dim)), size=(x.shape[1], dim)).astype(np.float32)
    return (x.astype(np.float32) @ w).astype(np.float32)


def _hybrid_hash_features(texts: List[str], dim: int) -> np.ndarray:
    if HashingVectorizer is None:
        return np.zeros((len(texts), dim), dtype=np.float32)
    d_word = max(8, dim // 2)
    d_char = max(8, dim - d_word)

    vec_word = HashingVectorizer(
        n_features=d_word,
        alternate_sign=False,
        norm="l2",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        lowercase=True,
    )
    vec_char = HashingVectorizer(
        n_features=d_char,
        alternate_sign=False,
        norm="l2",
        analyzer="char",
        ngram_range=(2, 4),
        lowercase=False,
    )
    mw = vec_word.transform(texts).astype(np.float32).toarray()
    mc = vec_char.transform(texts).astype(np.float32).toarray()
    x = np.concatenate([mw, mc], axis=1).astype(np.float32)
    return _project_dim(x, dim)


def _sentence_transformer_features(texts: List[str], dim: int) -> np.ndarray:
    if SentenceTransformer is None:
        return _hybrid_hash_features(texts, dim)
    model_name = TEXT_MODEL_NAME.strip() or "paraphrase-multilingual-MiniLM-L12-v2"
    if model_name not in _TEXT_MODEL_CACHE:
        _TEXT_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    model = _TEXT_MODEL_CACHE[model_name]
    x = model.encode(
        texts,
        batch_size=max(1, int(TEXT_BATCH_SIZE)),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return _project_dim(x, dim)


def build_text_features(texts: List[str], dim: int = 64) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros((0, dim), dtype=np.float32)
    texts = _safe_texts(texts)
    enc = str(TEXT_ENCODER).strip().lower()
    if enc == "none":
        return np.zeros((len(texts), dim), dtype=np.float32)
    if enc == "hash_word":
        if HashingVectorizer is None:
            return np.zeros((len(texts), dim), dtype=np.float32)
        vec = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        return vec.transform(texts).astype(np.float32).toarray()
    if enc == "sentence_transformer":
        return _sentence_transformer_features(texts, dim)
    return _hybrid_hash_features(texts, dim)


def append_text_features(base_feat: np.ndarray, node_text: List[str], text_dim: int = 64) -> np.ndarray:
    if base_feat.shape[0] == 0:
        return base_feat.astype(np.float32)
    if len(node_text) != base_feat.shape[0]:
        node_text = (node_text + ["[MISSING]"] * base_feat.shape[0])[: base_feat.shape[0]]
    txt = build_text_features(node_text, dim=text_dim)
    return np.concatenate([base_feat.astype(np.float32), txt.astype(np.float32)], axis=1).astype(np.float32)


def parse_arxiv(zip_path: Path, max_edges: int):
    with zipfile.ZipFile(zip_path) as z:
        with z.open("arxiv/raw/edge.csv.gz") as ef:
            edges = pd.read_csv(ef, compression="gzip", header=None, names=["src", "dst"], nrows=max_edges)
        with z.open("arxiv/raw/node_year.csv.gz") as yf:
            years = pd.read_csv(yf, compression="gzip", header=None, names=["year"])
        with z.open("arxiv/raw/node-label.csv.gz") as lf:
            labels = pd.read_csv(lf, compression="gzip", header=None, names=["label"])

        node_feat = None
        try:
            with z.open("arxiv/raw/node-feat.csv.gz") as nf:
                node_feat = pd.read_csv(nf, compression="gzip", header=None)
        except KeyError:
            node_feat = None

    src = edges["src"].to_numpy(dtype=np.int64)
    dst = edges["dst"].to_numpy(dtype=np.int64)
    n = int(max(src.max(), dst.max(), len(years) - 1) + 1)

    yv = years["year"].to_numpy(dtype=np.float32)
    yv = np.pad(yv, (0, max(0, n - len(yv))), constant_values=float(np.median(yv)))
    ts = np.maximum(yv[src], yv[dst])

    rel = np.zeros(src.shape[0], dtype=np.int64)
    node_type = np.zeros(n, dtype=np.int64)

    mc = labels["label"].to_numpy(dtype=np.int64)
    mc = np.pad(mc, (0, max(0, n - len(mc))), constant_values=0)
    ml = build_default_multilabel(n, src, dst, ts)
    if node_feat is None:
        feat = build_degree_features(n, src, dst, ts, node_type, dim=128)
    else:
        feat_np = node_feat.to_numpy(dtype=np.float32)
        if feat_np.shape[0] < n:
            pad = np.zeros((n - feat_np.shape[0], feat_np.shape[1]), dtype=np.float32)
            feat_np = np.concatenate([feat_np, pad], axis=0)
        feat = feat_np[:n]
    node_text = [f"paper year_{int(yv[i])} label_{int(mc[i])}" for i in range(n)]
    feat = append_text_features(feat, node_text, text_dim=TEXT_DIM)

    return {
        "name": "arxiv",
        "src": src,
        "dst": dst,
        "ts": ts,
        "rel": rel,
        "node_type": node_type,
        "node_feat": feat,
        "node_multiclass": mc,
        "node_multilabel": ml,
        "meta_extra": {
            **text_encoder_meta(),
            "hetero_main_benchmark_ok": False,
            "hetero_skip_reason": "single_relation_single_node_type",
        },
    }


def parse_amazon_movies(zip_path: Path, max_edges: int):
    with zipfile.ZipFile(zip_path) as z:
        with z.open("edge_list.csv") as f:
            df = pd.read_csv(f, nrows=max_edges)

    # Columns in this dataset are commonly [index, u, r, i, ts, label, idx].
    # In the released archive, `r` is a per-review identifier with review text in
    # relation_text.csv, so treating it as a relation type collapses link tasks.
    # We therefore fall back to the discrete rating label when `r` is effectively
    # near-unique, which keeps one processed dataset usable across edge/node tasks.
    src = df["u"].to_numpy(dtype=np.int64)
    dst = df["i"].to_numpy(dtype=np.int64)
    rel_raw = df["r"].to_numpy(dtype=np.int64)
    ts = df["ts"].to_numpy(dtype=np.float32)
    rating = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else None

    rel_vals, rel_cnt = np.unique(rel_raw, return_counts=True)
    rel_is_edge_id_like = rel_vals.size > 0 and (
        rel_vals.size >= max(32, int(0.2 * max(1, rel_raw.shape[0])))
        or int(rel_cnt.max()) <= 16
    )
    if rating is not None:
        rating_vals = np.unique(rating)
    else:
        rating_vals = np.array([], dtype=np.int64)

    if rel_is_edge_id_like and 1 < rating_vals.size <= 10:
        rel_source = rating
        rel_source_name = "rating_label"
    else:
        rel_source = rel_raw
        rel_source_name = "raw_relation_id"

    rel_unique = {r: i for i, r in enumerate(sorted(np.unique(rel_source).tolist()))}
    rel = np.array([rel_unique[r] for r in rel_source], dtype=np.int64)

    n = int(max(src.max(), dst.max()) + 1)
    node_type = np.zeros(n, dtype=np.int64)
    node_type[dst] = 1
    feat = build_degree_features(n, src, dst, ts, node_type, dim=64)

    # Multiclass node label: dominant relation per source node.
    rel_counter = [Counter() for _ in range(n)]
    for u, r in zip(src, rel):
        rel_counter[u][int(r)] += 1
    mc = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(rel_counter):
        if c:
            mc[i] = c.most_common(1)[0][0]

    ml = build_default_multilabel(n, src, dst, ts)
    in_deg = np.bincount(dst, minlength=n).astype(np.int64)
    out_deg = np.bincount(src, minlength=n).astype(np.int64)
    node_text = []
    for i in range(n):
        ntype = "item" if int(node_type[i]) == 1 else "user"
        node_text.append(f"{ntype} id_{i} indeg_{int(in_deg[i])} outdeg_{int(out_deg[i])}")
    feat = append_text_features(feat, node_text, text_dim=TEXT_DIM)

    return {
        "name": "amazon_movies",
        "src": src,
        "dst": dst,
        "ts": ts,
        "rel": rel,
        "node_type": node_type,
        "node_feat": feat,
        "node_multiclass": mc,
        "node_multilabel": ml,
        "meta_extra": {
            **text_encoder_meta(),
            "relation_source": rel_source_name,
            "raw_relation_unique": int(rel_vals.size),
            "edge_relation_unique": int(np.unique(rel).size),
        },
    }


def parse_kg4ai(tar_path: Path, max_edges: int):
    node_map: Dict[str, int] = {}
    node_type_map: Dict[str, int] = {"Scholar": 0, "Paper": 1, "Specialty": 2}
    node_keys: List[str] = []

    def get_node(x: str):
        if x not in node_map:
            node_map[x] = len(node_map)
            node_keys.append(x)
        return node_map[x]

    src, dst, rel, ts = [], [], [], []
    scholar_hindex = {}
    rel_map = {"author_of": 0, "has_specialty": 1}

    with tarfile.open(tar_path, "r:gz") as tar:
        f = tar.extractfile("kg4ai_global_ai.jsondl")
        if f is None:
            raise FileNotFoundError("kg4ai_global_ai.jsondl not found")

        for li, line in enumerate(f):
            if len(src) >= max_edges:
                break
            try:
                obj = json.loads(line.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            graph = obj.get("@graph", [])
            for item in graph:
                if item.get("@type") != "Scholar":
                    continue
                sid = str(item.get("@id", f"scholar_{li}"))
                s = get_node(sid)
                scholar_hindex[s] = int(item.get("hIndex", 0) or 0)

                for p in item.get("hasPart", []) or []:
                    pid = str(p.get("@id", ""))
                    if not pid:
                        continue
                    src.append(s)
                    dst.append(get_node(pid))
                    rel.append(rel_map["author_of"])
                    ts.append(float(li))
                    if len(src) >= max_edges:
                        break
                if len(src) >= max_edges:
                    break

                for sp in item.get("specialty", []) or []:
                    spid = str(sp.get("@id", ""))
                    if not spid:
                        continue
                    src.append(s)
                    dst.append(get_node(spid))
                    rel.append(rel_map["has_specialty"])
                    ts.append(float(li))
                    if len(src) >= max_edges:
                        break
                if len(src) >= max_edges:
                    break

    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    rel = np.array(rel, dtype=np.int64)
    ts = np.array(ts, dtype=np.float32)

    n = len(node_map)
    node_type = np.zeros(n, dtype=np.int64)
    for k, i in node_map.items():
        if "/archive/" in k:
            node_type[i] = node_type_map["Paper"]
        elif "/domain/" in k:
            node_type[i] = node_type_map["Specialty"]
        else:
            node_type[i] = node_type_map["Scholar"]

    feat = build_degree_features(n, src, dst, ts, node_type, dim=64)
    node_text = []
    for i in range(n):
        key = node_keys[i] if i < len(node_keys) else f"node_{i}"
        tail = key.split("/")[-1].replace("_", " ").replace("-", " ")
        if int(node_type[i]) == node_type_map["Scholar"]:
            node_text.append(f"scholar {tail} hindex_{int(scholar_hindex.get(i, 0))}")
        elif int(node_type[i]) == node_type_map["Paper"]:
            node_text.append(f"paper {tail}")
        else:
            node_text.append(f"specialty {tail}")
    feat = append_text_features(feat, node_text, text_dim=TEXT_DIM)

    # Scholar-based class labels by h-index quantization.
    hvals = np.array([scholar_hindex.get(i, 0) for i in range(n)], dtype=np.float32)
    bins = np.quantile(hvals, [0.33, 0.66]) if hvals.size > 0 else [0.0, 0.0]
    mc = np.digitize(hvals, bins).astype(np.int64)
    ml = build_default_multilabel(n, src, dst, ts)

    return {
        "name": "kg4ai",
        "src": src,
        "dst": dst,
        "ts": ts,
        "rel": rel,
        "node_type": node_type,
        "node_feat": feat,
        "node_multiclass": mc,
        "node_multilabel": ml,
        "meta_extra": text_encoder_meta(),
    }


def parse_raw_patent(zip_path: Path, country: str, max_edges: int):
    inner = f"raw-data/{country}-all.csv"
    read_rows = None if max_edges <= 0 else max(50000, int(max_edges * 4))
    try:
        with zipfile.ZipFile(zip_path) as z:
            with z.open(inner) as f:
                df = pd.read_csv(f, nrows=read_rows)
    except (NotImplementedError, RuntimeError, KeyError):
        # Fallback for zip methods unsupported by Python's zipfile (e.g., Deflate64).
        proc = subprocess.Popen(
            ["unzip", "-p", str(zip_path), inner],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout is None:
            raise RuntimeError("Failed to open unzip pipe for raw patent data")
        df = pd.read_csv(proc.stdout, nrows=read_rows)
        proc.stdout.close()
        proc.wait()

    title_col = find_col(df, ["title", "patentTitle"])
    abstract_col = find_col(df, ["summary", "abstract", "摘要"])
    date_col = find_col(df, ["publicationDate", "publishDate", "pubDate"])
    assignee_col = find_col(df, ["applicantName", "orgname", "assignee", "assigneeName"])
    class_col = find_col(df, ["classCode", "ipcCode", "ipccode", "ipc"])

    if assignee_col is None or class_col is None:
        raise ValueError(f"missing assignee/class columns in {inner}; columns={df.columns.tolist()}")

    if date_col is not None:
        dt = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        df = df.assign(_dt=dt).sort_values(by="_dt", kind="mergesort")
    else:
        df = df.copy()

    ipc_name_map = load_ipc_subclass_names(country)

    node_map: Dict[str, int] = {}
    node_type: List[int] = []
    node_text: List[str] = []
    node_name: List[str] = []

    def get_node(key: str, ntype: int, text: str, name: str) -> int:
        if key in node_map:
            return node_map[key]
        nid = len(node_map)
        node_map[key] = nid
        node_type.append(int(ntype))
        node_text.append(str(text))
        node_name.append(str(name))
        return nid

    # node types: patent=0, assignee=1, ipc_subclass=2
    rel_map = {"patent_assignee": 0, "patent_ipc": 1, "assignee_ipc": 2}
    src, dst, rel, ts = [], [], [], []

    patent_node_ids: List[int] = []
    patent_publication_ts: List[float] = []
    patent_publication_raw: List[str] = []
    patent_titles: List[str] = []
    patent_abstracts: List[str] = []
    patent_subclass_codes: Dict[int, List[str]] = {}

    assignee_node_ids: set = set()
    ipc_code_to_node: Dict[str, int] = {}

    patent_seq = 0
    for ridx, row in df.iterrows():
        title = str(row.get(title_col, "")) if title_col is not None else ""
        abstract = str(row.get(abstract_col, "")) if abstract_col is not None else ""
        publication_raw = str(row.get(date_col, "")) if date_col is not None else ""
        t_val = parse_publication_ts(row.get(date_col, None) if date_col is not None else None, float(ridx))

        assignees = [norm_assignee_name(x) for x in split_multi_values(row.get(assignee_col, ""))]
        assignees = [x for x in assignees if x]
        _, subclass_codes = extract_ipc_levels(str(row.get(class_col, "")))
        subclass_codes = sorted(set(subclass_codes))

        if len(assignees) == 0 and len(subclass_codes) == 0:
            continue

        patent_id = patent_seq
        patent_seq += 1
        p_node = get_node(
            key=f"pat:{patent_id}",
            ntype=0,
            text=(title + " " + abstract).strip(),
            name=f"patent:{patent_id}",
        )

        patent_node_ids.append(p_node)
        patent_publication_ts.append(t_val)
        patent_publication_raw.append(publication_raw)
        patent_titles.append(title)
        patent_abstracts.append(abstract)
        patent_subclass_codes[p_node] = subclass_codes

        asg_nodes = []
        for asg in assignees:
            a_node = get_node(key=f"asg:{asg}", ntype=1, text=asg, name=asg)
            assignee_node_ids.add(a_node)
            asg_nodes.append(a_node)
            src.append(p_node)
            dst.append(a_node)
            rel.append(rel_map["patent_assignee"])
            ts.append(t_val)

        ipc_nodes = []
        for sc in subclass_codes:
            ipc_name = ipc_name_map.get(sc, "")
            c_node = get_node(key=f"ipc:{sc}", ntype=2, text=(sc + " " + ipc_name).strip(), name=sc)
            ipc_code_to_node[sc] = c_node
            ipc_nodes.append(c_node)
            src.append(p_node)
            dst.append(c_node)
            rel.append(rel_map["patent_ipc"])
            ts.append(t_val)

        for a_node in asg_nodes:
            for c_node in ipc_nodes:
                src.append(a_node)
                dst.append(c_node)
                rel.append(rel_map["assignee_ipc"])
                ts.append(t_val)

        if max_edges > 0 and len(src) >= max_edges:
            break

    if len(src) == 0:
        pa = get_node("pat:0", 0, "empty patent", "patent:0")
        asg = get_node("asg:dummy", 1, "dummy assignee", "dummy")
        ipc = get_node("ipc:A00A", 2, "A00A dummy", "A00A")
        src = [pa, pa, asg]
        dst = [asg, ipc, ipc]
        rel = [rel_map["patent_assignee"], rel_map["patent_ipc"], rel_map["assignee_ipc"]]
        ts = [0.0, 0.0, 0.0]
        patent_node_ids = [pa]
        patent_publication_ts = [0.0]
        patent_publication_raw = [""]
        patent_titles = [""]
        patent_abstracts = [""]
        patent_subclass_codes = {pa: ["A00A"]}
        assignee_node_ids = {asg}
        ipc_code_to_node = {"A00A": ipc}

    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    rel = np.array(rel, dtype=np.int64)
    ts = np.array(ts, dtype=np.float32)

    if max_edges > 0 and len(src) > max_edges:
        src = src[:max_edges]
        dst = dst[:max_edges]
        rel = rel[:max_edges]
        ts = ts[:max_edges]

    n = len(node_map)
    node_type_arr = np.array(node_type, dtype=np.int64)
    deg_feat = build_degree_features(n, src, dst, ts, node_type_arr, dim=64)
    feat = append_text_features(deg_feat, node_text, text_dim=TEXT_DIM)

    subclass_codes_sorted = sorted(ipc_code_to_node.keys())
    subclass_to_idx = {c: i for i, c in enumerate(subclass_codes_sorted)}
    mc_pat = np.zeros(len(patent_node_ids), dtype=np.int64)
    ml_pat = np.zeros((len(patent_node_ids), max(1, len(subclass_codes_sorted))), dtype=np.float32)
    for i, p_node in enumerate(patent_node_ids):
        codes = patent_subclass_codes.get(p_node, [])
        if len(codes) == 0:
            continue
        for c in codes:
            if c in subclass_to_idx:
                ml_pat[i, subclass_to_idx[c]] = 1.0
        mc_pat[i] = subclass_to_idx[codes[0]] if codes[0] in subclass_to_idx else 0

    # Extra task metadata for assignee->IPC trend prediction.
    asg_ipc_idx = np.where(rel == rel_map["assignee_ipc"])[0]
    trans_train = trans_val = trans_test = np.array([], dtype=np.int64)
    ind_train = ind_val = ind_test = np.array([], dtype=np.int64)
    unseen_assignee_ids: List[int] = []
    if asg_ipc_idx.size > 0:
        tr_i, va_i, te_i = temporal_split(ts[asg_ipc_idx])
        trans_train = asg_ipc_idx[tr_i]
        trans_val = asg_ipc_idx[va_i]
        trans_test = asg_ipc_idx[te_i]

        rng = np.random.RandomState(42)
        asg_events: Dict[int, List[int]] = defaultdict(list)
        for ei in asg_ipc_idx:
            asg_events[int(src[ei])].append(int(ei))
        candidates = [k for k, v in asg_events.items() if len(v) >= 2]
        if len(candidates) > 0:
            n_unseen = max(1, int(len(candidates) * 0.10))
            unseen_assignee_ids = sorted(rng.choice(candidates, size=n_unseen, replace=False).tolist())
        unseen_set = set(unseen_assignee_ids)
        inductive_edges = np.array([ei for ei in asg_ipc_idx if int(src[ei]) in unseen_set], dtype=np.int64)
        base_edges = np.array([ei for ei in asg_ipc_idx if int(src[ei]) not in unseen_set], dtype=np.int64)
        if base_edges.size > 0:
            btr_i, bva_i, _ = temporal_split(ts[base_edges], train_ratio=0.8, val_ratio=0.2)
            ind_train = base_edges[btr_i]
            ind_val = base_edges[bva_i]
        ind_test = inductive_edges

    # Note: patent_id is monotonically increasing and unique by construction.
    return {
        "name": f"raw_{country.lower()}",
        "src": src,
        "dst": dst,
        "ts": ts,
        "rel": rel,
        "node_type": node_type_arr,
        "node_feat": feat,
        "node_multiclass": mc_pat,
        "node_multiclass_node_ids": np.array(patent_node_ids, dtype=np.int64),
        "node_multilabel": ml_pat,
        "node_multilabel_node_ids": np.array(patent_node_ids, dtype=np.int64),
        "meta_extra": {
            **text_encoder_meta(),
            "country": country.upper(),
            "relation_map": rel_map,
            "node_type_map": {"patent": 0, "assignee": 1, "ipc_subclass": 2},
            "patent_attributes": {
                "node_ids": patent_node_ids,
                "patent_id": list(range(len(patent_node_ids))),
                "title": patent_titles,
                "abstract": patent_abstracts,
                "publication_date": patent_publication_raw,
                "publication_ts": patent_publication_ts,
            },
            "ipc_attributes": {
                "codes": subclass_codes_sorted,
                "names": [ipc_name_map.get(c, "") for c in subclass_codes_sorted],
                "node_ids": [ipc_code_to_node[c] for c in subclass_codes_sorted],
            },
            "assignee_node_ids": sorted(list(assignee_node_ids)),
            "task_splits": {
                "assignee_ipc_transductive": {
                    "edge_indices_train": trans_train.tolist(),
                    "edge_indices_val": trans_val.tolist(),
                    "edge_indices_test": trans_test.tolist(),
                },
                "assignee_ipc_inductive": {
                    "edge_indices_train": ind_train.tolist(),
                    "edge_indices_val": ind_val.tolist(),
                    "edge_indices_test": ind_test.tolist(),
                    "unseen_assignee_node_ids": unseen_assignee_ids,
                },
            },
        },
    }


def parse_dblp(zip_path: Path, max_edges: int):
    node_map = {}
    node_text_map: Dict[int, str] = {}

    def get_node(k: str):
        if k not in node_map:
            node_map[k] = len(node_map)
        return node_map[k]

    def merge_text(nid: int, text: str) -> None:
        t = str(text).strip()
        if not t:
            return
        old = node_text_map.get(int(nid), "")
        if old:
            if t not in old:
                node_text_map[int(nid)] = (old + " " + t)[:256]
        else:
            node_text_map[int(nid)] = t[:256]

    rel_map = {"author_write_paper": 0, "paper_cite_paper": 1}
    src, dst, rel, ts = [], [], [], []
    paper_year = {}

    with zipfile.ZipFile(zip_path) as z:
        with z.open("DBLP-Citation-network-V18.jsonl") as f:
            for li, line in enumerate(f):
                if len(src) >= max_edges:
                    break
                try:
                    obj = json.loads(line.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    continue

                pid = str(obj.get("id", f"paper_{li}"))
                p = get_node(f"paper:{pid}")
                year = float(obj.get("year", li))
                paper_year[p] = year
                title = str(obj.get("title", "")).strip()
                venue = str(obj.get("venue", "")).strip()
                merge_text(p, f"{title} {venue} year_{int(year)}")

                for a in obj.get("authors", []) or []:
                    if isinstance(a, dict):
                        aid = str(a.get("id", ""))
                        aname = str(a.get("name", "")).strip()
                    else:
                        aid = str(a).strip()
                        aname = str(a).strip()
                    if not aid:
                        continue
                    au = get_node(f"author:{aid}")
                    merge_text(au, f"author {aname if aname else aid}")
                    src.append(au)
                    dst.append(p)
                    rel.append(rel_map["author_write_paper"])
                    ts.append(year)
                    if len(src) >= max_edges:
                        break
                if len(src) >= max_edges:
                    break

                for ref in obj.get("references", []) or []:
                    rp = get_node(f"paper:{ref}")
                    src.append(p)
                    dst.append(rp)
                    rel.append(rel_map["paper_cite_paper"])
                    ts.append(year)
                    if len(src) >= max_edges:
                        break

    src = np.array(src[:max_edges], dtype=np.int64)
    dst = np.array(dst[:max_edges], dtype=np.int64)
    rel = np.array(rel[:max_edges], dtype=np.int64)
    ts = np.array(ts[:max_edges], dtype=np.float32)
    n = len(node_map)

    node_type = np.zeros(n, dtype=np.int64)
    for k, i in node_map.items():
        node_type[i] = 0 if k.startswith("paper:") else 1

    feat = build_degree_features(n, src, dst, ts, node_type, dim=64)
    node_text = []
    for i in range(n):
        txt = node_text_map.get(i, "")
        if not txt:
            txt = f"paper node_{i}" if int(node_type[i]) == 0 else f"author node_{i}"
        node_text.append(txt)
    feat = append_text_features(feat, node_text, text_dim=TEXT_DIM)

    years = np.array([paper_year.get(i, 0.0) for i in range(n)], dtype=np.float32)
    if np.any(years > 0):
        bins = np.quantile(years[years > 0], [0.33, 0.66])
        mc = np.digitize(years, bins).astype(np.int64)
    else:
        mc = np.zeros(n, dtype=np.int64)
    ml = build_default_multilabel(n, src, dst, ts)

    return {
        "name": "dblp_v18",
        "src": src,
        "dst": dst,
        "ts": ts,
        "rel": rel,
        "node_type": node_type,
        "node_feat": feat,
        "node_multiclass": mc,
        "node_multilabel": ml,
        "meta_extra": text_encoder_meta(),
    }


def iter_aminer_paper_records(archive_path: Path):
    proc = subprocess.Popen(
        ["unrar", "p", "-inul", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError("Failed to open unrar pipe for AMiner paper data")

    cur = None

    def _emit_current():
        if cur is None:
            return None
        pid = str(cur.get("id", "")).strip()
        if not pid:
            return None
        return {
            "id": pid,
            "title": str(cur.get("title", "")).strip(),
            "authors": str(cur.get("authors", "")).strip(),
            "year": str(cur.get("year", "")).strip(),
            "venue": str(cur.get("venue", "")).strip(),
            "abstract": str(cur.get("abstract", "")).strip(),
            "references": [str(x).strip() for x in cur.get("references", []) if str(x).strip()],
        }

    early_stop = False
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("#index"):
                item = _emit_current()
                if item is not None:
                    yield item
                cur = {"id": line[len("#index") :].strip(), "references": []}
                continue
            if cur is None:
                continue
            if line.startswith("#*"):
                cur["title"] = line[2:].strip()
            elif line.startswith("#@"):
                cur["authors"] = line[2:].strip()
            elif line.startswith("#t"):
                cur["year"] = line[2:].strip()
            elif line.startswith("#c"):
                cur["venue"] = line[2:].strip()
            elif line.startswith("#!"):
                old = str(cur.get("abstract", "")).strip()
                cur["abstract"] = (old + " " + line[2:].strip()).strip() if old else line[2:].strip()
            elif line.startswith("#%"):
                cur.setdefault("references", []).append(line[2:].strip())

        item = _emit_current()
        if item is not None:
            yield item
    except GeneratorExit:
        early_stop = True
        raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        stderr = ""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
            proc.stderr.close()
        ret = proc.wait()
        if ret != 0 and not early_stop:
            raise RuntimeError(f"unrar failed for {archive_path}: {stderr.strip()}")


def iter_aminer_author_records(archive_path: Path, wanted_ids: Optional[set] = None):
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        if not names:
            return
        with z.open(names[0]) as f:
            cur = None
            wanted = set(wanted_ids) if wanted_ids is not None else None

            def _emit_current():
                if cur is None:
                    return None
                aid = str(cur.get("id", "")).strip()
                if not aid:
                    return None
                if wanted is not None and aid not in wanted:
                    return None
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
                    item = _emit_current()
                    if item is not None:
                        yield item
                        if wanted is not None and len(wanted) == 0:
                            return
                    cur = {"id": line[len("#index") :].strip()}
                    continue
                if cur is None:
                    continue
                if line.startswith("#n"):
                    cur["name"] = line[2:].strip()
                elif line.startswith("#a"):
                    cur["affiliation"] = line[2:].strip()
                elif line.startswith("#pc"):
                    cur["paper_count"] = line[3:].strip()
                elif line.startswith("#cn"):
                    cur["citation_count"] = line[3:].strip()
                elif line.startswith("#hi"):
                    cur["h_index"] = line[3:].strip()
                elif line.startswith("#pi"):
                    cur["pi"] = line[3:].strip()
                elif line.startswith("#upi"):
                    cur["upi"] = line[4:].strip()
                elif line.startswith("#t"):
                    cur["topics"] = line[2:].strip()

            item = _emit_current()
            if item is not None:
                yield item


def iter_aminer_author2paper_rows(archive_path: Path):
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        if not names:
            return
        with z.open(names[0]) as f:
            for raw_line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                line = raw_line.strip()
                if not line:
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) < 4:
                    continue
                try:
                    author_id = str(int(parts[1]))
                    paper_id = str(int(parts[2]))
                    author_order = int(parts[3])
                except Exception:
                    continue
                yield {
                    "author_id": author_id,
                    "paper_id": paper_id,
                    "author_order": author_order,
                }


def iter_aminer_coauthor_rows(archive_path: Path):
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        if not names:
            return
        with z.open(names[0]) as f:
            for raw_line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                line = raw_line.strip()
                if not line:
                    continue
                line = line.lstrip("#")
                parts = re.split(r"\s+", line)
                if len(parts) < 3:
                    continue
                try:
                    src = str(int(parts[0]))
                    dst = str(int(parts[1]))
                    weight = int(parts[2])
                except Exception:
                    continue
                yield {"src_author_id": src, "dst_author_id": dst, "weight": weight}


def parse_aminer_paper(archive_path: Path, max_edges: int, min_year: int = 2002, max_year: int = 2012):
    if not shutil.which("unrar"):
        raise RuntimeError("unrar is required to parse AMiner-Paper.rar")

    aminer_dir = archive_path.parent
    author_zip = aminer_dir / "AMiner-Author.zip"
    author2paper_zip = aminer_dir / "AMiner-Author2Paper.zip"
    coauthor_zip = aminer_dir / "AMiner-Coauthor.zip"
    missing = [str(p.name) for p in [author_zip, author2paper_zip, coauthor_zip] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"AMiner heterogeneous parsing requires sibling archives next to {archive_path.name}: {', '.join(missing)}"
        )

    paper_meta: Dict[str, Dict[str, Any]] = {}
    active_paper_ids: set = set()
    selected_source_paper_ids: set = set()
    source_paper_nodes: List[int] = []

    def get_node(k: str):
        if k not in node_map:
            node_map[k] = len(node_map)
        return node_map[k]

    def merge_text(nid: int, text: str) -> None:
        t = str(text).strip()
        if not t:
            return
        old = node_text_map.get(int(nid), "")
        if old:
            if t not in old:
                node_text_map[int(nid)] = (old + " " + t)[:512]
        else:
            node_text_map[int(nid)] = t[:512]

    node_map: Dict[str, int] = {}
    node_text_map: Dict[int, str] = {}
    node_type_map = {"paper": 0, "author": 1, "venue": 2}
    paper_year: Dict[int, float] = {}
    paper_venue: Dict[int, str] = {}
    paper_doc_text: Dict[int, str] = {}

    citation_src: List[int] = []
    citation_dst: List[int] = []
    citation_rel: List[int] = []
    citation_ts: List[float] = []
    rel_map = {
        "paper_cite_paper": 0,
        "author_write_paper": 1,
        "paper_publish_venue": 2,
        "author_coauthor_author": 3,
    }
    papers_with_edges = 0
    year_values = list(range(int(min_year), int(max_year) + 1))
    per_year_cap = max(1, int(np.ceil(float(max_edges) / max(1, len(year_values))))) if max_edges > 0 else 0
    year_edge_used: Dict[int, int] = {y: 0 for y in year_values}

    for obj in iter_aminer_paper_records(archive_path):
        year_raw = obj.get("year", "")
        year_match = re.search(r"\d{4}", year_raw)
        if year_match is None:
            continue
        year_i = int(year_match.group(0))
        year = float(year_i)
        if year_i < int(min_year):
            continue
        if year_i > int(max_year):
            continue
        refs = obj.get("references", [])
        if not refs:
            continue
        if max_edges > 0:
            remaining_total = int(max_edges - len(citation_src))
            if remaining_total <= 0:
                break
            remaining_year = int(per_year_cap - year_edge_used.get(year_i, 0))
            if remaining_year <= 0:
                continue
            refs = refs[: min(len(refs), remaining_total, remaining_year)]
            if not refs:
                continue

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
            rp = get_node(f"paper:{ref}")
            citation_src.append(p)
            citation_dst.append(rp)
            citation_rel.append(rel_map["paper_cite_paper"])
            citation_ts.append(year)
        year_edge_used[year_i] = int(year_edge_used.get(year_i, 0) + len(refs))

        papers_with_edges += 1
        if max_edges > 0 and len(citation_src) >= max_edges:
            break

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
        author_id = str(meta["id"])
        if author_id not in active_author_ids:
            continue
        author_name_meta[author_id] = str(meta.get("name", "")).strip()
        author_affil_meta[author_id] = str(meta.get("affiliation", "")).strip()
        author_topic_meta[author_id] = str(meta.get("topics", "")).strip()

    coauthor_weight_meta: Dict[Tuple[str, str], int] = {}
    for row in iter_aminer_coauthor_rows(coauthor_zip):
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
        p_node = get_node(f"paper:{paper_id}")
        meta = paper_meta.get(paper_id, {})
        year = float(meta.get("year", 0.0))
        venue = str(meta.get("venue", "")).strip()

        if paper_id in selected_source_paper_ids:
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
                a = sorted_authors[i]
                b = sorted_authors[j]
                key = tuple(sorted((a, b)))
                if coauthor_weight_meta and key not in coauthor_weight_meta:
                    continue
                edge_key = (key[0], key[1], int(year))
                if edge_key in dynamic_coauthor_seen:
                    continue
                dynamic_coauthor_seen.add(edge_key)
                a_node = get_node(f"author:{a}")
                b_node = get_node(f"author:{b}")
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
        if key.startswith("author:"):
            node_type[nid] = node_type_map["author"]
        elif key.startswith("venue:"):
            node_type[nid] = node_type_map["venue"]
        else:
            node_type[nid] = node_type_map["paper"]

    for key, nid in node_map.items():
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

    src = citation_src + write_src + venue_src + co_src
    dst = citation_dst + write_dst + venue_dst + co_dst
    rel = citation_rel + write_rel + venue_rel + co_rel
    ts = citation_ts + write_ts + venue_ts + co_ts

    src_np = np.array(src, dtype=np.int64)
    dst_np = np.array(dst, dtype=np.int64)
    rel_np = np.array(rel, dtype=np.int64) if rel else np.zeros((0,), dtype=np.int64)
    ts_np = np.array(ts, dtype=np.float32) if ts else np.zeros((0,), dtype=np.float32)

    feat = build_degree_features(n, src_np, dst_np, ts_np, node_type, dim=64)
    node_text = []
    for i in range(n):
        txt = node_text_map.get(i, "")
        if not txt:
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
        text = str(paper_doc_text.get(node_id, "")).lower()
        if not text:
            continue
        row = [1.0 if any(tok in text for tok in toks) else 0.0 for _, toks in topic_rules]
        if sum(row) <= 0:
            continue
        ml_node_ids.append(int(node_id))
        ml_rows.append(row)
        mc_node_ids.append(int(node_id))
        mc_labels.append(int(np.argmax(np.asarray(row, dtype=np.float32))))

    if ml_rows:
        ml = np.asarray(ml_rows, dtype=np.float32)
        ml_node_ids_np = np.asarray(ml_node_ids, dtype=np.int64)
        mc = np.asarray(mc_labels, dtype=np.int64)
        mc_node_ids_np = np.asarray(mc_node_ids, dtype=np.int64)
    else:
        ml = build_default_multilabel(n, src_np, dst_np, ts_np)
        ml_node_ids_np = np.arange(n, dtype=np.int64)
        mc = np.zeros(n, dtype=np.int64)
        mc_node_ids_np = np.arange(n, dtype=np.int64)

    venue_counts = Counter(v for v in paper_venue.values() if v)
    top_venues = [v for v, _ in venue_counts.most_common(16)]

    return {
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


def to_unified(dataset_obj: Dict, out_pt: Path):
    src = dataset_obj["src"].astype(np.int64)
    dst = dataset_obj["dst"].astype(np.int64)
    ts = dataset_obj["ts"].astype(np.float32)
    rel = dataset_obj["rel"].astype(np.int64)

    order = np.argsort(ts)
    src = src[order]
    dst = dst[order]
    ts = ts[order]
    rel = rel[order]

    num_nodes = dataset_obj["node_feat"].shape[0]
    num_rels = int(rel.max()) + 1 if rel.size > 0 else 1

    # Simple numeric edge features used by all models.
    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)
    rel_norm = rel / max(1, num_rels - 1)
    edge_feat = np.stack([ts_norm, rel_norm], axis=1).astype(np.float32)

    tr, va, te = temporal_split(ts)

    mc_node_ids = dataset_obj.get("node_multiclass_node_ids", np.arange(num_nodes, dtype=np.int64))
    mc_labels = dataset_obj["node_multiclass"]
    ml_node_ids = dataset_obj.get("node_multilabel_node_ids", np.arange(num_nodes, dtype=np.int64))
    ml_labels = dataset_obj["node_multilabel"]

    obj = {
        "num_nodes": int(num_nodes),
        "num_rels": int(num_rels),
        "num_node_types": int(dataset_obj["node_type"].max()) + 1,
        "node_feat": torch.from_numpy(dataset_obj["node_feat"].astype(np.float32)),
        "node_type": torch.from_numpy(dataset_obj["node_type"].astype(np.int64)),
        "edges": {
            "src": torch.from_numpy(src),
            "dst": torch.from_numpy(dst),
            "ts": torch.from_numpy(ts),
            "rel": torch.from_numpy(rel),
            "edge_feat": torch.from_numpy(edge_feat),
            "edge_label": torch.ones(src.shape[0], dtype=torch.float32),
        },
        "split": {
            "train": torch.from_numpy(tr.astype(np.int64)),
            "val": torch.from_numpy(va.astype(np.int64)),
            "test": torch.from_numpy(te.astype(np.int64)),
        },
        "node_multiclass": {
            "node_ids": torch.from_numpy(np.asarray(mc_node_ids, dtype=np.int64)),
            "labels": torch.from_numpy(np.asarray(mc_labels, dtype=np.int64)),
        },
        "node_multilabel": {
            "node_ids": torch.from_numpy(np.asarray(ml_node_ids, dtype=np.int64)),
            "labels": torch.from_numpy(np.asarray(ml_labels, dtype=np.float32)),
        },
        "meta": {
            "dataset_name": dataset_obj["name"],
            "num_edges": int(src.shape[0]),
            "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
            "extra": dataset_obj.get("meta_extra", {}),
        },
    }

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out_pt)
    return obj


def export_tgn_format(obj: Dict, out_dir: Path, dataset_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    src = obj["edges"]["src"].numpy()
    dst = obj["edges"]["dst"].numpy()
    ts = obj["edges"]["ts"].numpy()
    feat = obj["edges"]["edge_feat"].numpy()

    # TGN preprocess expects rows: u,i,ts,label,feat...
    csv_path = out_dir / f"{dataset_name}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("u,i,ts,label,f0,f1\n")
        for i in range(len(src)):
            f.write(f"{int(src[i])},{int(dst[i])},{float(ts[i])},1,{float(feat[i,0])},{float(feat[i,1])}\n")

    np.save(out_dir / f"ml_{dataset_name}.npy", np.vstack([np.zeros((1, feat.shape[1]), dtype=np.float32), feat]))
    node_feat = obj["node_feat"].numpy().astype(np.float32)
    np.save(out_dir / f"ml_{dataset_name}_node.npy", np.vstack([np.zeros((1, node_feat.shape[1]), dtype=np.float32), node_feat]))


def export_tgb2_like_csv(obj: Dict, out_dir: Path, dataset_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    src = obj["edges"]["src"].numpy()
    dst = obj["edges"]["dst"].numpy()
    rel = obj["edges"]["rel"].numpy()
    ts = obj["edges"]["ts"].numpy()

    df = pd.DataFrame(
        {
            "u": src,
            "i": dst,
            "ts": ts,
            "edge_type": rel,
            "label": np.ones_like(src),
            "idx": np.arange(src.shape[0]),
        }
    )
    df.to_csv(out_dir / f"{dataset_name}_edgelist.csv", index=False)

    nt = pd.DataFrame({"node_id": np.arange(obj["num_nodes"]), "node_type": obj["node_type"].numpy()})
    nt.to_csv(out_dir / f"{dataset_name}_nodetype.csv", index=False)


def main():
    global TEXT_ENCODER, TEXT_DIM, TEXT_MODEL_NAME, TEXT_BATCH_SIZE
    p = argparse.ArgumentParser("Prepare dataset into unified .pt format")
    p.add_argument("--dataset", default='aminer_paper', required=True, choices=["amazon_movies", "raw_cn", "raw_us", "dblp", "aminer_paper"]) #"kg4ai"
    p.add_argument("--archive", required=True, type=str, default="/home/xuxj/2026/datasets/Aminer/AMiner-Paper.rar")
    p.add_argument("--max-edges", type=int, default=200000)
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
    p.add_argument("--aminer-min-year", type=int, default=2002)
    p.add_argument("--aminer-max-year", type=int, default=2012)
    p.add_argument("--export-baselines", action="store_true")
    p.add_argument("--baseline-out-dir", type=str, default="")
    args = p.parse_args()

    TEXT_ENCODER = str(args.text_encoder).strip().lower()
    TEXT_DIM = int(max(8, args.text_dim))
    TEXT_MODEL_NAME = str(args.text_model_name).strip()
    TEXT_BATCH_SIZE = int(max(1, args.text_batch_size))

    archive = Path(args.archive)
    out_pt = Path(args.out_pt)

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

    obj = to_unified(d, out_pt)
    print(f"saved unified dataset to {out_pt}")
    print(f"nodes={obj['num_nodes']} edges={obj['edges']['src'].shape[0]} rels={obj['num_rels']}")

    if args.export_baselines:
        if not args.baseline_out_dir:
            raise ValueError("--baseline-out-dir is required when --export-baselines is set")
        bdir = Path(args.baseline_out_dir)
        export_tgn_format(obj, bdir / "tgn", d["name"])
        export_tgb2_like_csv(obj, bdir / "tgb2", d["name"])
        print(f"baseline adapters exported to {bdir}")


if __name__ == "__main__":
    main()
