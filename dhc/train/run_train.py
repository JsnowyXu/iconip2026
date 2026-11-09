import argparse
import json
from pathlib import Path

import torch

from dhc.data.unified_dataset import load_unified_pt
from dhc.models.dhc_model import DHCModel
from dhc.train.trainer import Trainer
from dhc.utils.seed import set_seed


def infer_task_type(data) -> str:
    meta = data.meta if isinstance(data.meta, dict) else {}
    extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
    for key in ["task_view_name", "dataset_name"]:
        value = str(extra.get(key, "") if key == "task_view_name" else meta.get(key, "")).strip().lower()
        if value.endswith("edge_transductive"):
            return "edge_transductive"
        if value.endswith("edge_inductive"):
            return "edge_inductive"
        if value.endswith("edge_patent_ipc"):
            return "edge_patent_ipc"
        if value.endswith("node_multilabel"):
            return "node_multilabel"
        if value.endswith("node_multiclass"):
            return "node_multiclass"
    return "edge_transductive"


def parse_args():
    p = argparse.ArgumentParser("Train DHC model")
    p.add_argument("--data", type=str, required=True, help="Path to unified .pt")
    p.add_argument("--out-dir", type=str, default="results/dhc_default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gpu-id", type=int, default=0, help="CUDA device id")
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument(
        "--monitor-metric",
        type=str,
        default="auto",
        choices=["auto", "val_mrr", "val_ap", "val_auc", "val_node_ml_micro_f1", "val_node_ml_map", "val_node_mc_acc"],
    )
    p.add_argument(
        "--task-type",
        type=str,
        default="auto",
        choices=["auto", "edge_transductive", "edge_inductive", "edge_patent_ipc", "node_multilabel", "node_multiclass"],
    )

    p.add_argument("--mem-dim", type=int, default=128)
    p.add_argument("--rel-dim", type=int, default=64)
    p.add_argument("--time-dim", type=int, default=32)
    p.add_argument("--msg-dim", type=int, default=128)
    p.add_argument("--community-dim", type=int, default=128)
    p.add_argument("--num-communities", type=int, default=64)
    p.add_argument("--community-assign-temperature", type=float, default=1.0)
    p.add_argument("--community-assign-topk", type=int, default=0)
    p.add_argument("--no-recent-neighbors", action="store_true")
    p.add_argument("--recent-neighbor-size", type=int, default=10)
    p.add_argument("--neighbor-aggregation", type=str, default="attn", choices=["attn", "mean"])
    p.add_argument("--no-edge-type-features", action="store_true")

    p.add_argument("--no-relation-memory", action="store_true")
    p.add_argument("--no-community", action="store_true")

    p.add_argument("--lambda-cl", type=float, default=0.1)
    p.add_argument("--lambda-community-entropy", type=float, default=0.0)
    p.add_argument("--lambda-mc", type=float, default=1.0)
    p.add_argument("--lambda-ml", type=float, default=1.0)
    p.add_argument(
        "--relation-loss-weights",
        type=str,
        default="",
        help="Comma-separated relation loss weights, e.g., patent_ipc=2.0,assignee_ipc=0.4,0=1.5",
    )
    p.add_argument("--ml-pos-weight", action="store_true", default=False)
    p.add_argument("--ml-pos-weight-max", type=float, default=50.0)
    p.add_argument("--ml-pos-weight-power", type=float, default=0.5)
    p.add_argument("--ml-threshold-search", action="store_true", default=False)
    p.add_argument("--ml-threshold-num-candidates", type=int, default=99)
    p.add_argument("--ml-calib-max-nodes", type=int, default=0)
    p.add_argument("--node-eval-max-nodes", type=int, default=0)
    p.add_argument("--typed-negative-ratio", type=float, default=0.6)
    p.add_argument("--popular-negative-ratio", type=float, default=0.2)
    p.add_argument("--save-memory-artifacts", action="store_true", default=False)
    p.add_argument("--no-lightweight-checkpoint", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        gpu_id = int(max(0, min(args.gpu_id, torch.cuda.device_count() - 1)))
        device = torch.device(f"cuda:{gpu_id}")
    else:
        gpu_id = -1
        device = torch.device("cpu")
    print(f"[run_train] device={device}")
    data = load_unified_pt(args.data, device=device)
    task_type = infer_task_type(data) if args.task_type == "auto" else args.task_type
    print(f"[run_train] task_type={task_type}")

    num_multiclass = 0
    num_multilabel = 0
    if data.node_multiclass is not None:
        num_multiclass = int(data.node_multiclass["labels"].max().item()) + 1
    if data.node_multilabel is not None:
        num_multilabel = int(data.node_multilabel["labels"].shape[1])

    model = DHCModel(
        num_nodes=data.num_nodes,
        num_rels=data.num_rels,
        num_node_types=data.num_node_types,
        node_feat=data.node_feat,
        node_type=data.node_type,
        edge_feat_dim=data.edge_feat.shape[1],
        mem_dim=args.mem_dim,
        rel_dim=args.rel_dim,
        time_dim=args.time_dim,
        msg_dim=args.msg_dim,
        community_dim=args.community_dim,
        num_communities=args.num_communities,
        num_multiclass=num_multiclass,
        num_multilabel=num_multilabel,
        use_relation_memory=not args.no_relation_memory,
        use_community=not args.no_community,
        community_assign_temperature=args.community_assign_temperature,
        community_assign_topk=args.community_assign_topk,
        use_recent_neighbors=not args.no_recent_neighbors,
        recent_neighbor_size=args.recent_neighbor_size,
        use_edge_type_features=not args.no_edge_type_features,
        neighbor_aggregation=args.neighbor_aggregation,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[run_train] params total={total_params} trainable={trainable_params}")

    trainer = Trainer(
        model=model,
        data=data,
        device=device,
        task_type=task_type,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lambda_cl=args.lambda_cl,
        lambda_community_entropy=args.lambda_community_entropy,
        lambda_mc=args.lambda_mc,
        lambda_ml=args.lambda_ml,
        relation_loss_weights=args.relation_loss_weights,
        ml_use_pos_weight=args.ml_pos_weight,
        ml_pos_weight_max=args.ml_pos_weight_max,
        ml_pos_weight_power=args.ml_pos_weight_power,
        ml_threshold_search=args.ml_threshold_search,
        ml_threshold_num_candidates=args.ml_threshold_num_candidates,
        ml_calib_max_nodes=args.ml_calib_max_nodes,
        node_eval_max_nodes=args.node_eval_max_nodes,
        typed_negative_ratio=args.typed_negative_ratio,
        popular_negative_ratio=args.popular_negative_ratio,
        save_memory_artifacts=args.save_memory_artifacts,
        lightweight_checkpoint=not args.no_lightweight_checkpoint,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        monitor_metric=args.monitor_metric,
        verbose=True,
        out_dir=args.out_dir,
    )
    result = trainer.train()
    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        obj["model_stats"] = {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "device": str(device),
            "gpu_id": int(gpu_id),
        }
        obj["run_args"] = vars(args)
        metrics_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "model_stats.json").write_text(
        json.dumps(
            {
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "device": str(device),
                "gpu_id": int(gpu_id),
                "args": vars(args),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(result)


if __name__ == "__main__":
    main()
