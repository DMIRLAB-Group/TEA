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


class DGRec(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(DGRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        edim = args.edim
        self.edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args

        self.rnn = nn.LSTM(input_size=edim, hidden_size=edim, num_layers=1, batch_first=True)
        self.nbr_lin = nn.Linear(edim + edim, edim, bias=False)
        self.gat_lin1 = nn.Linear(edim, edim, bias=False)
        self.gat_lin2 = nn.Linear(edim, edim, bias=False)
        self.gat_lin3 = nn.Linear(edim, edim, bias=False)
        self.fuse_lin = nn.Linear(edim + edim, edim, bias=False)

        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5/user_num, b=0.5/user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5/item_num, b=0.5/item_num)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def dgrec_forward(self, uid, seq, nbr, nbr_iid):
        # get mask
        batch_size, nbr_maxlen, seq_maxlen = nbr_iid.shape
        nbr_mask = torch.BoolTensor(nbr == 0).to(self.dev)  # B x nl
        nbr_seq_mask = torch.BoolTensor(nbr_iid == 0).to(self.dev)  # B x nl x sl

        # user short term feat
        user_seq_emb = self.item_embs(seq.to(self.dev))  # B x sl x d
        user_seq_feat, _ = self.rnn(user_seq_emb)  # B x sl x d
        user_seq_feat = user_seq_feat[:, -1, :]  # B x d

        # nbr short term feat
        nbr_seq_emb = self.item_embs(nbr_iid.to(self.dev))  # B x nl x sl x d
        nbr_seq_emb *= ~nbr_seq_mask.unsqueeze(-1)
        nbr_seq_emb = nbr_seq_emb.view(batch_size * nbr_maxlen, seq_maxlen, self.edim)
        nbr_seq_feat, _ = self.rnn(nbr_seq_emb)  # (B*nl) x sl x d
        nbr_seq_feat = nbr_seq_feat[:, -1, :]  # (B*nl) x d
        nbr_seq_feat = nbr_seq_feat.view(batch_size, nbr_maxlen, self.edim)  # B x nl x d

        # nbr long term feat
        nbr_emb = self.user_embs(nbr.to(self.dev))  # B x nl x d

        # nbr final feat
        nbr_feat = self.dropout(self.act(self.nbr_lin(
            torch.cat([nbr_seq_feat, nbr_emb], dim=-1))))  # B x nl x d

        # 3 layer dot GAT
        hu = user_seq_feat.unsqueeze(1)  # B x 1 x d
        attn = (hu * nbr_feat).sum(dim=-1, keepdims=True)  # B x nl x 1
        attn = attn + (-1e9) * nbr_mask.unsqueeze(-1)
        attn = F.softmax(attn, dim=1)
        hu = (attn * nbr_feat).sum(dim=1)  # B x d
        hu = self.dropout(self.act(self.gat_lin1(hu)))

        hu = hu.unsqueeze(1)  # B x 1 x d
        attn = (hu * nbr_feat).sum(dim=-1, keepdims=True)  # B x nl x 1
        attn = attn + (-1e9) * nbr_mask.unsqueeze(-1)
        attn = F.softmax(attn, dim=1)
        hu = (attn * nbr_feat).sum(dim=1)  # B x d
        hu = self.dropout(self.act(self.gat_lin2(hu)))

        hu = hu.unsqueeze(1)  # B x 1 x d
        attn = (hu * nbr_feat).sum(dim=-1, keepdims=True)  # B x nl x 1
        attn = attn + (-1e9) * nbr_mask.unsqueeze(-1)
        attn = F.softmax(attn, dim=1)
        hu = (attn * nbr_feat).sum(dim=1)  # B x d
        hu = self.dropout(self.act(self.gat_lin3(hu)))

        # user final feat
        hu = self.fuse_lin(torch.cat([hu, user_seq_feat], dim=-1))  # B x d

        return hu

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch

        hu = self.dgrec_forward(uid, seq, nbr, nbr_iid)

        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))
        hu = hu.unsqueeze(1).expand_as(pos_hi)

        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)

        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                uid = uid.long()
                seq = seq.long()
                nbr = nbr.long()
                nbr_iid = nbr_iid.long()
                eval_iid = eval_iid.long()

                hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
                hu = self.dgrec_forward(uid, seq, nbr, nbr_iid)
                hu = hu.unsqueeze(1).expand_as(hi)

                batch_score = self.pred(hu, hi)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores

    def get_parameters(self):
        param_list = [
            {'params': self.rnn.parameters()},
            {'params': self.nbr_lin.parameters()},
            {'params': self.gat_lin1.parameters()},
            {'params': self.gat_lin2.parameters()},
            {'params': self.gat_lin3.parameters()},
            {'params': self.fuse_lin.parameters()},
            {'params': self.user_embs.parameters(), 'weight_decay': 0},
            {'params': self.item_embs.parameters(), 'weight_decay': 0},
        ]

        return param_list


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
    ranks = (-1.0 * all_scores).argsort(1).argsort(1).cpu().numpy()
    ranks = ranks[:, 0]

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
        pos_logits, neg_logits = model(parsed_batch)
        loss = F.binary_cross_entropy_with_logits(pos_logits[indices], torch.ones_like(pos_logits[indices])) + \
               F.binary_cross_entropy_with_logits(neg_logits[indices], torch.zeros_like(neg_logits[indices]))
        loss.backward()
        opt.step()
        shdlr.step()

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
        drop_last=False)

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
    parser = argparse.ArgumentParser(description='DGRec')
    parser.add_argument('--dataset', default='Yelp')
    parser.add_argument('--model', default='DGRec')

    # Model Config
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=50, help='fixed, or change with sampled train_batches')
    parser.add_argument('--nbr_maxlen', type=int, default=20, help='fixed, or change with sampled train_batches')

    # Train Config
    parser.add_argument('--batch_size', type=int, default=1024, help='fixed, or change with sampled train_batches')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--l2rg', type=float, default=5e-4)
    parser.add_argument('--emb_reg', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--check_epoch', type=int, default=5)
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
    print('Loaded {} dataset with {} users {} items in {:.2f}s'.format(args.dataset, user_num, item_num, time()-st))
    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")  # best_model_id_timestr = '20210123_19h39m22s'
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

        model = DGRec(user_num, item_num, args)
        model = model.to(device)
        opt = torch.optim.Adam(model.get_parameters(), lr=args.lr, weight_decay=args.l2rg)
        lr_scheduler = StepwiseLR(opt, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decay_rate)
        best_score = patience_cnt = 0
        for epoch in range(1, args.max_epochs):
            st = time()
            train_loss = train(model, opt, lr_scheduler, train_loader, args)
            print('Epoch:{} Train Loss={:.4f} Time={:.2f}s LR={:.8f}'.format(
                epoch, train_loss, time()-st, lr_scheduler.get_lr()))

            if epoch % args.check_epoch == 0 and epoch >= 20:
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

        print('Testing')
        model.load_state_dict(torch.load(model_path))
        test_metrics = evaluate(model, test_loader, eval_users)
        hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = test_metrics
        logger.info('Iter={} Tst HR(5/10/20)={:.4f}/{:.4f}/{:.4f} NDCG(5/10/20)={:.4f}/{:.4f}/{:.4f}'
                    .format(r, hr5, hr10, hr20, ndcg5, ndcg10, ndcg20))
        metrics_list.append(test_metrics)

    metrics = np.array(metrics_list)
    means = metrics.mean(axis=0)
    stds = metrics.std(axis=0)
    print(f'{args.model} {args.dataset} Test Summary:')
    logger.info('Mean hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        means[0], means[1], means[2], means[3], means[4], means[5]))
    logger.info('Std  hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        stds[0], stds[1], stds[2], stds[3], stds[4], stds[5]))
    logger.info("Done")


if __name__ == '__main__':
    main()