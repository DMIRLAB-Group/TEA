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


class TransRec(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(TransRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)
        self.args = args

        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        self.item_beta = nn.Embedding(item_num, 1, padding_idx=0)
        self.trans = nn.Parameter(torch.zeros((args.edim, )))

        nn.init.uniform_(self.user_embs.weight, a=-6 / args.edim, b=6 / args.edim)
        nn.init.uniform_(self.item_embs.weight, a=-6 / args.edim, b=6 / args.edim)
        nn.init.uniform_(self.item_beta.weight, a=-6 / args.edim, b=6 / args.edim)
        nn.init.uniform_(self.trans.data, a=-6 / args.edim, b=6 / args.edim)

    def l2_distance(self, x, y):
        return (x - y).square().sum(dim=-1, keepdims=True)

    def clip_by_norm(self, tensor, clip_norm, dim=None):
        clip_norm = torch.FloatTensor([clip_norm]).to(tensor.device)
        l2sum = tensor.pow(2).sum(dim=dim, keepdims=True)
        pred = l2sum > 0
        l2sum_safe = torch.where(pred, l2sum, torch.ones_like(l2sum))
        l2norm = torch.where(pred, l2sum_safe.sqrt(), l2sum)
        values_clip = tensor * clip_norm / torch.max(l2norm, clip_norm)
        return values_clip

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch

        user_emb = self.clip_by_norm(self.user_embs(uid.to(self.dev)), clip_norm=1, dim=-1)  # B x d
        seq_emb = self.clip_by_norm(self.item_embs(seq.to(self.dev)), clip_norm=1, dim=-1)  # B x sl x d
        pos_emb = self.clip_by_norm(self.item_embs(pos.to(self.dev)), clip_norm=1, dim=-1)
        neg_emb = self.clip_by_norm(self.item_embs(neg.to(self.dev)), clip_norm=1, dim=-1)

        user_emb = user_emb.unsqueeze(1).expand_as(seq_emb)  # B x sl x d
        h_out = user_emb + self.trans + seq_emb

        pos_bias = self.item_beta(pos.to(self.dev))  # B x sl x 1
        neg_bias = self.item_beta(neg.to(self.dev))  # B x sl x 1

        pos_logits = pos_bias - self.l2_distance(h_out, pos_emb)  # B x sl x 1
        neg_logits = neg_bias - self.l2_distance(h_out, neg_emb)  # B x sl x 1

        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                user_emb = self.user_embs(uid.long().to(self.dev))  # B x d
                last_emb = self.item_embs(seq[:, -1].long().to(self.dev))  # B x d
                item_embs = self.item_embs(eval_iid.long().to(self.dev))  # B x item_len x d
                item_bias = self.item_beta(eval_iid.long().to(self.dev))  # B x item_len x 1
                h_out = user_emb + self.trans + last_emb
                h_out = h_out.unsqueeze(1).expand_as(item_embs)
                batch_score = item_bias - self.l2_distance(h_out, item_embs)
                all_scores.append(batch_score.squeeze())

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
        shdlr.step()
        pos_logits, neg_logits = model(parsed_batch)
        loss = -1.0 * (pos_logits[indices] - neg_logits[indices]).sigmoid().log().mean()
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
    parser = argparse.ArgumentParser(description='TransRec')
    parser.add_argument('--model', default='TransRec')
    parser.add_argument('--dataset', default='Yelp')

    # Model Config
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=50, help='fixed, or change with sampled train_batches')

    # Train Config
    parser.add_argument('--batch_size', type=int, default=1024, help='fixed, or change with sampled train_batches')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--l2rg', type=float, default=0.0)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--max_epochs', type=int, default=200)
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

        model = TransRec(user_num, item_num, args)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
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
