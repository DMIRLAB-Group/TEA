import argparse
import copy
import logging
import os
import re
import sys
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from datetime import datetime
from multiprocessing import Queue, Process
from tqdm import tqdm
import gc
import networkx as nx
import numpy as np
import pandas as pd
import torch
from typing import Optional
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from time import time
import torch.multiprocessing
import joblib
torch.multiprocessing.set_sharing_strategy('file_system')
np.set_printoptions(threshold=1000000)


# ---------------------------------------------------------------------------------------------------------------------
# Helpers


def get_logger(filename=None):
    """
    logging configuration: set the basic configuration of the logging system
    :param filename:
    :return:
    """

    log_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)-5.5s] %(message)s', datefmt='%Y-%b-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.INFO)
    logger.addHandler(std_handler)

    return logger


class StepwiseLR:

    def __init__(self, optimizer: Optimizer, init_lr: Optional[float],
                 gamma: Optional[float], decay_rate: Optional[float]):
        """
            A lr_scheduler that update learning rate using the following schedule:

            .. math::
                \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

            where `i` is the iteration steps.

            Parameters:
                - **optimizer**: Optimizer
                - **init_lr** (float, optional): initial learning rate. Default: 0.01
                - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
                - **decay_rate** (float, optional): :math:`p` . Default: 0.75
        """
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        lr = self.get_lr()
        self.iter_num += 1
        for param_group in self.optimizer.param_groups:
            if "lr_mult" not in param_group:
                param_group["lr_mult"] = 1
            param_group['lr'] = lr * param_group["lr_mult"]


# ---------------------------------------------------------------------------------------------------------------------
# Model


class FFN(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(FFN, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SAS(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(SAS, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        edim = args.edim
        droprate = args.droprate
        self.edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args

        # Self-Attention Block, encode user behavior sequences
        num_heads = 1
        self.item_attn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_attn_layer = nn.MultiheadAttention(edim, num_heads, droprate)
        self.item_ffn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_ffn = FFN(edim, args.droprate)
        self.item_last_layernorm = nn.LayerNorm(edim, eps=1e-8)

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.normal_(self.user_embs.weight, std=0.01)
        nn.init.normal_(self.item_embs.weight, std=0.01)

        if args.use_pos_emb:
            self.posn_embs = nn.Embedding(args.seq_maxlen, edim, padding_idx=0)
            nn.init.normal_(self.posn_embs.weight, std=0.01)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)
        self.trained_user_feat = None

    def seq2feat(self, seq_iid):
        timeline_mask = torch.BoolTensor(seq_iid == 0).to(self.dev)  # mask the padding item
        seqs = self.item_embs(seq_iid.to(self.dev)) * (self.item_embs.embedding_dim ** 0.5)  # Rescale emb
        if self.args.use_pos_emb:
            positions = np.tile(np.array(range(seq_iid.shape[1]), dtype=np.int64), [seq_iid.shape[0], 1])
            seqs += self.posn_embs(torch.LongTensor(positions).to(self.dev))

        seqs = self.dropout(seqs)

        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        seqs = torch.transpose(seqs, 0, 1)  # seqlen x B x d
        query = self.item_attn_layernorm(seqs)
        mha_outputs, _ = self.item_attn_layer(query, seqs, seqs, attn_mask=attention_mask)

        seqs = query + mha_outputs
        seqs = torch.transpose(seqs, 0, 1)  # B x seqlen x d
        seqs = self.item_ffn_layernorm(seqs)
        seqs = self.item_ffn(seqs)
        seqs *= ~timeline_mask.unsqueeze(-1)  # B x seqlen x d
        seqs = self.item_last_layernorm(seqs)

        return seqs

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch
        hu = self.seq2feat(seq)  # B x sl x d
        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))
        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)
        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                seq = seq.long()
                eval_iid = eval_iid.long()
                hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
                hu = self.seq2feat(seq)[:, -1, :]
                hu = hu.unsqueeze(1).expand_as(hi)

                batch_score = self.pred(hu, hi)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores


# ---------------------------------------------------------------------------------------------------------------------
# Train and Evaluate

def parse_sampled_batch(batch):
    uid, seq, pos, neg, nbr, nbr_iid = batch
    uid = uid.long()
    seq = seq.long()
    pos = pos.long()
    neg = neg.long()
    nbr = nbr.long()
    nbr_iid = nbr_iid.long()
    batch = [uid, seq, pos, neg, nbr, nbr_iid]
    indices = torch.where(pos != 0)
    return batch, indices


def evaluate(model, eval_loader, eval_users):
    model.eval()
    hr5 = hr10 = hr20 = ndcg5 = ndcg10 = ndcg20 = 0.0
    all_scores = model.eval_all_users(eval_loader)
    all_scores = all_scores[eval_users - 1]  # user 0 not in all scores
    #print('all_score =', all_scores[:5])
    ranks = (-1.0 * all_scores).argsort(1).argsort(1).cpu().numpy()
    #print('all_ranks =', ranks[:5])
    ranks = ranks[:, 0]
    #print('all_ranks =', ranks[:5])

    for rank in ranks:
        if rank < 5:
            hr5 += 1
            ndcg5 += 1 / np.log2(rank + 2)
        if rank < 10:
            hr10 += 1
            ndcg10 += 1 / np.log2(rank + 2)
        if rank < 20:
            hr20 += 1
            ndcg20 += 1 / np.log2(rank + 2)

    num_eval_user = len(eval_users)

    hr5 /= num_eval_user
    hr10 /= num_eval_user
    hr20 /= num_eval_user

    ndcg5 /= num_eval_user
    ndcg10 /= num_eval_user
    ndcg20 /= num_eval_user

    return hr5, hr10, hr20, ndcg5, ndcg10, ndcg20


def train(model, opt, shdlr, train_loader, args):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        parsed_batch, indices = parse_sampled_batch(batch)
        opt.zero_grad()
        shdlr.step()
        pos_logits, neg_logits = model(parsed_batch)
        loss = F.binary_cross_entropy_with_logits(pos_logits[indices], torch.ones_like(pos_logits)[indices]) + \
               F.binary_cross_entropy_with_logits(neg_logits[indices], torch.zeros_like(neg_logits)[indices])
        loss.backward()
        opt.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


class WechatDataset(Dataset):
    def __init__(self, data, item_num, seq_maxlen, is_train, is_test):
        self.uid = data['train_uid']
        self.seq = data['train_seq']
        self.pos = data['train_pos']
        self.nbr = data['train_nbr']
        self.nbr_iid = data['train_nbr_iid']

        self.user_train = data['user_train'][()]
        self.user_valid = data['user_valid'][()]
        self.user_test = data['user_test'][()]

        self.eval_users = set(data['eval_users'].tolist())

        self.item_num = item_num
        self.seq_maxlen = seq_maxlen
        self.is_train = is_train
        self.is_test = is_test

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        user = self.uid[idx]
        seq = self.seq[idx]
        nbr = self.nbr[idx]
        nbr_iid = self.nbr_iid[idx].toarray()
        if self.is_train:
            pos = self.pos[idx]
            neg = self.get_neg_samples(user, seq)
            return user, seq, pos, neg, nbr, nbr_iid
        else:
            eval_iids = self.get_neg_samples(user, seq)
            return user, seq, nbr, nbr_iid, eval_iids

    def get_neg_samples(self, user, seq):
        if self.is_train:
            neg_sample_size = len(np.nonzero(seq)[0])
            if neg_sample_size == 0:  # user_train[user]只有一个item，并在pos中用作next item
                return np.zeros((self.seq_maxlen), dtype=np.int64)
            else:
                neg_list = []
        else:
            neg_sample_size = 100
            if user not in self.eval_users:
                return np.zeros((neg_sample_size), dtype=np.int64)
            else:
                if self.is_test: eval_iid = self.user_test[user][0]
                else: eval_iid = self.user_valid[user][0]
                neg_list = [eval_iid]

        rated_set = set(self.user_train[user][:, 0].tolist())
        if len(self.user_valid[user]): rated_set.add(self.user_valid[user][0])
        if len(self.user_test [user]): rated_set.add(self.user_test [user][0])
        rated_set.add(0)

        while len(neg_list) < neg_sample_size:
            neg = np.random.randint(low=1, high=self.item_num)
            while neg in rated_set:
                neg = np.random.randint(low=1, high=self.item_num)
            neg_list.append(neg)

        if self.is_train:
            samples = np.zeros_like(seq, dtype=np.int64)
            samples[-neg_sample_size:] = neg_list
        else:
            samples = np.array(neg_list, dtype=np.int64)

        return samples


def load_ds(args, item_num):
    data = np.load(f'datasets/{args.dataset}/processed_data.npz', allow_pickle=True)

    train_loader = DataLoader(
        dataset=WechatDataset(data, item_num, args.seq_maxlen, is_train=True, is_test=False),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
        pin_memory=True)

    val_loader = DataLoader(
        dataset=WechatDataset(data, item_num, args.seq_maxlen, is_train=False, is_test=False),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False)

    test_loader = DataLoader(
        dataset=WechatDataset(data, item_num, args.seq_maxlen, is_train=False, is_test=True),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False)

    user_train = data['user_train'][()]
    eval_users = data['eval_users']

    return train_loader, val_loader, test_loader, user_train, eval_users



def main():
    parser = argparse.ArgumentParser(description='SASRec')
    parser.add_argument('--model', default='SASRec')
    parser.add_argument('--dataset', default='Yelp')

    # Model Config
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--rnn_type', default='gru', help='lstm/gru/lin/mlp')
    parser.add_argument('--pred_type', default='dot', help='dot/cos/mlp')
    parser.add_argument('--cos_tau', type=float, default=0.1, help='cosine rescale factor')
    parser.add_argument('--use_pos_emb', type=int, default=True)
    parser.add_argument('--seq_maxlen', type=int, default=50, help='fixed, or change with sampled train_batches')
    parser.add_argument('--nbr_maxlen', type=int, default=20, help='fixed, or change with sampled train_batches')

    # Train Config
    parser.add_argument('--batch_size', type=int, default=1024, help='fixed, or change with sampled train_batches')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--l2rg', type=float, default=1e-4)
    parser.add_argument('--emb_reg', type=float, default=0.0)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--check_epoch', type=int, default=1)
    parser.add_argument('--loss_type', default='bce', help='bce/bpr')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)

    # Something else
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--test_time', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=False)
    args = parser.parse_args()

    user_num = 270770 + 1
    item_num = 184134 + 1

    print('Loading...')
    st = time()
    train_loader, val_loader, test_loader, user_train, eval_users = load_ds(args, item_num)
    print('Loaded Yelp dataset with {} users {} items in {:.2f}s'.format(user_num, item_num, time()-st))

    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    model_path = f'saved_models/{args.model}_{args.dataset}_{timestr}.pth'
    logger = get_logger(os.path.join('logs', f'{args.model}_{args.dataset}_{timestr}.log'))
    logger.info(args)
    device = torch.device(args.device)

    metrics_list = []
    for r in range(args.repeat):
        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = SAS(user_num, item_num, args)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2rg)
        lr_scheduler = StepwiseLR(opt, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decay_rate)
        best_score = patience_cnt = 0

        for epoch in range(1, args.max_epochs):
            st = time()
            train_loss = train(model, opt, lr_scheduler, train_loader, args)
            print('Epoch:{} Train Loss={:.4f} Time={:.2f}s LR={:.8f}'.format(
                epoch, train_loss, time()-st, lr_scheduler.get_lr()))

            if epoch % args.check_epoch == 0 and epoch >= 0:
                val_metrics = evaluate(model, val_loader, eval_users)
                hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = val_metrics
                logger.info(
                    'Iter={} Epoch={:04d} Val HR(5/10/20)={:.4f}/{:.4f}/{:.4f} NDCG(5/10/20)={:.4f}/{:.4f}/{:.4f}'
                    .format(r, epoch, hr5, hr10, hr20, ndcg5, ndcg10, ndcg20))

                if best_score < hr10:
                    torch.save(model.state_dict(), model_path)
                    print('Validation score increased: {:.4f} --> {:.4f}'.format(best_score, hr10))
                    best_score = hr10
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt == args.patience:
                    print('Early Stop!!!')
                    break

        print('Testing', end='')
        model.load_state_dict(torch.load(model_path))
        test_metrics = evaluate(model, test_loader, eval_users)
        hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = test_metrics
        logger.info('Iter={} Tst HR(5/10/20)={:.4f}/{:.4f}/{:.4f} NDCG(5/10/20)={:.4f}/{:.4f}/{:.4f}'
                    .format(r, hr5, hr10, hr20, ndcg5, ndcg10, ndcg20))
        metrics_list.append(test_metrics)

    metrics = np.array(metrics_list)
    means = metrics.mean(axis=0)
    stds = metrics.std(axis=0)
    print('Test Summary:')
    logger.info('Mean hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        means[0], means[1], means[2], means[3], means[4], means[5]))
    logger.info('Std  hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        stds[0], stds[1], stds[2], stds[3], stds[4], stds[5]))
    logger.info("Done")


if __name__ == '__main__':
    main()
