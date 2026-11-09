#!/usr/bin/env python3
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import numpy as np

ROOT_TGB2 = Path('/home/xuxj/2026/baselines/src/TGB2-main/TGB2-main')
sys.path.insert(0, str(ROOT_TGB2))

from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.memory_module import TGNMemory
from modules.msg_agg import LastAggregator
from modules.msg_func import IdentityMessage
from modules.neighbor_loader import LastNeighborLoader


def auc_ap(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        order = np.argsort(y_score)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(y_score)) + 1
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    prec = tp / np.maximum(tp + fp, 1)
    ap = (prec * (y == 1)).sum() / max((y == 1).sum(), 1)
    return float(auc), float(ap)


def mrr_from_pos_neg(pos_scores, neg_scores):
    pos = np.asarray(pos_scores, dtype=np.float64).reshape(-1, 1)
    neg = np.asarray(neg_scores, dtype=np.float64)
    ranks = 1 + (neg >= pos).sum(axis=1)
    return float((1.0 / np.maximum(ranks, 1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default='tgn', choices=['tgn', 'tgn_edgetype'])
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=200)
    ap.add_argument('--gpu-id', type=int, default=0)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--num-neighbors', type=int, default=10)
    ap.add_argument('--eval-num-negs', type=int, default=20)
    ap.add_argument('--out', default='/home/xuxj/2026/dyn_hetero_community/results/baseline_status/tgb2_custom_metrics.csv')
    args = ap.parse_args()

    if args.gpu_id < 0:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        if 0 <= args.gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu_id}')
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    obj = torch.load(args.data, map_location='cpu')

    # Keep the full temporal graph on CPU and move only the active minibatch
    # to GPU. The larger paper-scale tasks can exceed VRAM if the whole graph
    # is copied up front.
    src = obj['edges']['src'].long()
    dst = obj['edges']['dst'].long()
    ts = obj['edges']['ts'].long()
    msg = obj['edges']['edge_feat'].float()
    rel = obj['edges']['rel'].long()
    edge_label = obj['edges'].get('edge_label', torch.ones_like(obj['edges']['src']).float()).float()
    num_rels = int(obj.get('num_rels', int(rel.max().item()) + 1 if rel.numel() > 0 else 1))
    if args.model == 'tgn_edgetype':
        rel_msg = torch.nn.functional.one_hot(rel.clamp_min(0), num_classes=max(1, num_rels)).float()
        msg = torch.cat([msg, rel_msg], dim=1)

    train_idx = obj['split']['train'].long()
    val_idx = obj['split']['val'].long()
    test_idx = obj['split']['test'].long()

    num_nodes = int(obj['num_nodes'])
    msg_dim = msg.shape[1]
    mem_dim = 100
    time_dim = 100

    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=msg_dim,
        memory_dim=mem_dim,
        time_dim=time_dim,
        message_module=IdentityMessage(msg_dim, mem_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
    gnn = GraphAttentionEmbedding(in_channels=mem_dim, out_channels=mem_dim, msg_dim=msg_dim, time_enc=memory.time_enc).to(device)
    link_pred = LinkPredictor(mem_dim).to(device)
    neighbor_loader = LastNeighborLoader(num_nodes, size=max(1, int(args.num_neighbors)), device=device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=float(args.learning_rate),
    )
    criterion = torch.nn.BCELoss()
    params = list(memory.parameters()) + list(gnn.parameters()) + list(link_pred.parameters())
    total_params = int(sum(p.numel() for p in params))
    trainable_params = int(sum(p.numel() for p in params if p.requires_grad))

    def train_epoch():
        memory.train(); gnn.train(); link_pred.train()
        memory.reset_state(); neighbor_loader.reset_state()
        total_loss = 0.0
        idx = train_idx
        n_batch = math.ceil(idx.numel() / args.batch_size)
        for bi in range(n_batch):
            b = idx[bi*args.batch_size:(bi+1)*args.batch_size]
            if b.numel() == 0:
                continue
            bsrc = src[b].to(device)
            bdst = dst[b].to(device)
            bts = ts[b].to(device)
            bmsg = msg[b].to(device)
            blabel = (edge_label[b] > 0.5).to(device)
            neg_dst = torch.randint(0, num_nodes, (bsrc.size(0),), device=device)

            n_id = torch.cat([bsrc, bdst, neg_dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id)
            e_id_cpu = e_id.cpu()
            z = gnn(z, last_update, edge_index, ts[e_id_cpu].to(device), msg[e_id_cpu].to(device))

            pos_out = link_pred(z[assoc[bsrc]], z[assoc[bdst]]).squeeze(-1)
            neg_out = link_pred(z[assoc[bsrc]], z[assoc[neg_dst]]).squeeze(-1)
            if blabel.any():
                loss = criterion(pos_out[blabel], torch.ones_like(pos_out[blabel])) + criterion(
                    neg_out[blabel], torch.zeros_like(neg_out[blabel])
                )
            else:
                loss = None

            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            memory.update_state(bsrc, bdst, bts, bmsg)
            neighbor_loader.insert(bsrc, bdst)
            memory.detach()
            if loss is not None:
                total_loss += float(loss.item())

        return total_loss / max(n_batch, 1)

    @torch.no_grad()
    def evaluate(eval_idx, history_idx):
        # Reset before switching to eval mode. The TGB2 memory module flushes
        # pending messages inside train(False), which can trigger a large
        # one-shot allocation on long streams if we keep the training-state
        # message store alive here.
        memory.reset_state(); neighbor_loader.reset_state()
        memory.eval(); gnn.eval(); link_pred.eval()

        def replay(idx):
            for bi in range(math.ceil(idx.numel() / args.batch_size)):
                b = idx[bi*args.batch_size:(bi+1)*args.batch_size]
                if b.numel() == 0:
                    continue
                bsrc = src[b].to(device)
                bdst = dst[b].to(device)
                bts = ts[b].to(device)
                bmsg = msg[b].to(device)
                memory.update_state(bsrc, bdst, bts, bmsg)
                neighbor_loader.insert(bsrc, bdst)

        replay(history_idx)
        y_true, y_score = [], []
        pos_all, neg_all = [], []
        for bi in range(math.ceil(eval_idx.numel() / args.batch_size)):
            b = eval_idx[bi*args.batch_size:(bi+1)*args.batch_size]
            if b.numel() == 0:
                continue
            bsrc = src[b].to(device)
            bdst = dst[b].to(device)
            bts = ts[b].to(device)
            blabel = (edge_label[b] > 0.5).to(device)

            neg_dst_list = [torch.randint(0, num_nodes, (bsrc.size(0),), device=device) for _ in range(args.eval_num_negs)]
            n_id = torch.cat([bsrc, bdst] + neg_dst_list).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id)
            e_id_cpu = e_id.cpu()
            z = gnn(z, last_update, edge_index, ts[e_id_cpu].to(device), msg[e_id_cpu].to(device))
            if blabel.any():
                src_sup = bsrc[blabel]
                dst_sup = bdst[blabel]
                pos = link_pred(z[assoc[src_sup]], z[assoc[dst_sup]]).squeeze(-1)
                neg_scores = []
                for neg_dst in neg_dst_list:
                    neg = link_pred(z[assoc[src_sup]], z[assoc[neg_dst[blabel]]]).squeeze(-1)
                    neg_scores.append(neg.detach().cpu().numpy())
                neg_scores = np.stack(neg_scores, axis=1)

                y_true.extend([1]*pos.numel())
                y_score.extend(pos.detach().cpu().tolist())
                y_true.extend([0]*(neg_scores.shape[0] * neg_scores.shape[1]))
                y_score.extend(neg_scores.reshape(-1).tolist())
                pos_all.append(pos.detach().cpu().numpy())
                neg_all.append(neg_scores)

            memory.update_state(bsrc, bdst, bts, msg[b].to(device))
            neighbor_loader.insert(bsrc, bdst)

        if len(pos_all) == 0:
            return 0.0, 0.5, 0.0
        auc, ap = auc_ap(np.array(y_true), np.array(y_score))
        mrr = mrr_from_pos_neg(np.concatenate(pos_all, axis=0), np.concatenate(neg_all, axis=0))
        return mrr, auc, ap

    epoch_secs = []
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        loss = train_epoch()
        val_mrr, val_auc, val_ap = evaluate(val_idx, train_idx)
        ep_sec = time.time() - t0
        epoch_secs.append(ep_sec)
        print(f'epoch={ep} loss={loss:.6f} val_mrr={val_mrr:.6f} val_auc={val_auc:.6f} val_ap={val_ap:.6f} sec={ep_sec:.2f}')

    test_mrr, test_auc, test_ap = evaluate(test_idx, torch.cat([train_idx, val_idx]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    avg_ep_sec = float(np.mean(epoch_secs)) if len(epoch_secs) > 0 else 0.0
    out.write_text(
        'split,mrr,auc,ap,avg_epoch_sec,total_params,trainable_params,device,eval_num_negs\n'
        .replace('split,', 'split,model,')
        + f'val,{args.model},{val_mrr},{val_auc},{val_ap},{avg_ep_sec},{total_params},{trainable_params},{device},{args.eval_num_negs}\n'
        + f'test,{args.model},{test_mrr},{test_auc},{test_ap},{avg_ep_sec},{total_params},{trainable_params},{device},{args.eval_num_negs}\n',
        encoding='utf-8',
    )
    print(
        f'model={args.model} test_mrr={test_mrr:.6f} test_auc={test_auc:.6f} test_ap={test_ap:.6f} '
        f'params={total_params} trainable={trainable_params} avg_epoch_sec={avg_ep_sec:.2f}'
    )


if __name__ == '__main__':
    main()
