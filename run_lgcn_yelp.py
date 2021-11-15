import argparse
import copy
import logging
import os
import re
import sys
import scipy.sparse as sp
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
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt



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



def get_ui_graph(dataset, user_train, user_num, item_num):
    saved_path = f'datasets/{dataset}/norm_adj.npz'
    if os.path.exists(saved_path):
        norm_adj = sp.load_npz(saved_path)
        print('Loaded normalized joint rating matrix.')
    else:
        print('Generating sparse rating matrix...')
        train_users = []
        train_items = []

        for user in range(1, user_num):
            items = user_train[user]
            if len(items):
                train_users.extend(len(items) * [user])
                train_items.extend(items[:, 0].tolist())

        print('Step1: list to csr_matrix...')

        rating_maxtrix = sp.csr_matrix(
            (np.ones(len(train_users), dtype=np.int8), (train_users, train_items)),
            shape=(user_num, item_num)
        ).tolil()

        print('Step2: csr to dok...')

        adj_mat = sp.dok_matrix(
            (user_num + item_num, user_num + item_num),
            dtype=np.int8).tolil()

        print('adj_mat =', adj_mat.dtype, adj_mat.shape)
        print('rating_maxtrix =', rating_maxtrix.dtype, rating_maxtrix.shape)

        print('Step3: slicing...')

        adj_mat[:user_num, user_num:] = rating_maxtrix
        adj_mat[user_num:, :user_num] = rating_maxtrix.T
        adj_mat = adj_mat.todok().astype(np.float16)

        print('Step4: Normalizing...')

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo()
        sp.save_npz(saved_path, norm_adj)
        print('norm_adj saved at', saved_path)

    print('Npz to SparseTensor...')
    row = torch.Tensor(norm_adj.row).long()
    col = torch.Tensor(norm_adj.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(norm_adj.data)
    ui_graph = torch.sparse.FloatTensor(index, data, torch.Size(norm_adj.shape)).coalesce()
    return ui_graph



class LightGCN(nn.Module):
    def __init__(self,
                 edim,
                 n_layers,
                 user_num,
                 item_num,
                 ui_graph,
                 args):
        super(LightGCN, self).__init__()
        self.n_layers = n_layers
        self.num_users = user_num
        self.num_items = item_num
        self.args = args
        self.dev = torch.device(args.device)
        self.Graph = ui_graph.to(self.dev)

        self.embedding_user = torch.nn.Embedding(self.num_users, edim)
        self.embedding_item = torch.nn.Embedding(self.num_items, edim)
        nn.init.uniform_(self.embedding_user.weight, a=-0.5/user_num, b=0.5/user_num)
        nn.init.uniform_(self.embedding_item.weight, a=-0.5/item_num, b=0.5/item_num)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [torch.cat([users_emb, items_emb])]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, batch):
        uid, pos, neg = batch
        indices = torch.where(pos != 0)
        uid = uid.unsqueeze(1).expand_as(pos)

        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = \
            self.getEmbedding(
                uid.long().to(self.dev),
                pos.long().to(self.dev),
                neg.long().to(self.dev))

        pos_scores = (users_emb * pos_emb).sum(dim=-1)
        neg_scores = (users_emb * neg_emb).sum(dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores[indices] - pos_scores[indices]))
        loss += self.args.emb_reg * 0.5 * (
                userEmb0.norm(2, dim=-1).pow(2).mean() +
                posEmb0.norm(2, dim=-1).pow(2).mean() +
                negEmb0.norm(2, dim=-1).pow(2).mean())

        return loss

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            all_users, all_items = self.computer()
            for eval_batch in tqdm(eval_loader):
                uid, eval_iid = eval_batch
                users_emb = all_users[uid.long().to(self.dev)]  # B x d
                items_emb = all_items[eval_iid.long().to(self.dev)]  # B x item_len x d
                users_emb = users_emb.unsqueeze(1).expand_as(items_emb)
                batch_score = (users_emb * items_emb).sum(dim=-1)
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores



def train(model, opt, lr_scheduler, train_loader, args):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        opt.zero_grad()
        loss = model.bpr_loss(batch)
        # print('loss={:.4f}'.format(loss.item()))
        loss.backward()
        opt.step()
        lr_scheduler.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

class WechatDataset(Dataset):
    def __init__(self, data, item_num, seq_maxlen, is_train, is_test):
        self.uid = data['train_uid']
        self.seq = data['train_seq']
        self.pos = data['train_pos']
        # self.nbr = data['train_nbr']
        # self.nbr_iid = data['train_nbr_iid']

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
        if self.is_train:
            pos = self.pos[idx]
            neg = self.get_neg_samples(user, seq)
            return user, pos, neg
        else:
            eval_iids = self.get_neg_samples(user, seq)
            return user, eval_iids

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


def main():
    parser = argparse.ArgumentParser(description='LightGCN')
    parser.add_argument('--dataset', default='Yelp')
    parser.add_argument('--model', default='LightGCN')

    # Model Config
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=50, help='fixed, or change with sampled train_batches')
    parser.add_argument('--nbr_maxlen', type=int, default=20, help='fixed, or change with sampled train_batches')

    # Train Config
    parser.add_argument('--batch_size', type=int, default=1024, help='fixed, or change with sampled train_batches')
    parser.add_argument('--droprate', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--lr_gamma', type=float, default=0.0)
    parser.add_argument('--lr_decay_rate', type=float, default=0.0)
    parser.add_argument('--l2rg', type=float, default=0.0)
    parser.add_argument('--emb_reg', type=float, default=5e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--check_epoch', type=int, default=1)
    parser.add_argument('--loss_type', default='bpr', help='bce/bpr')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)

    # Something else
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--test_time', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=False)
    args = parser.parse_args()

    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    model_path = f'saved_models/{args.model}_{args.dataset}_{timestr}.pth'
    logger = get_logger(os.path.join('logs', f'{args.model}_{args.dataset}_{timestr}.log'))
    logger.info(args)
    device = torch.device(args.device)

    user_num = 270770 + 1
    item_num = 184134 + 1

    print('Loading...')
    st = time()
    train_loader, val_loader, test_loader, user_train, eval_users = load_ds(args, item_num)
    print('Loaded Yelp dataset with {} users {} items in {:.2f}s'.format(user_num, item_num, time()-st))

    #df, user_num, item_num = load_ds(args.dataset)
    #user_train, user_valid, user_test = data_partition(df)
    #dataset = [user_train, user_valid, user_test, user_num, item_num]
    ui_graph = get_ui_graph(args.dataset, user_train, user_num, item_num)

    metrics_list = []
    for r in range(args.repeat):
        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = LightGCN(args.edim, 3, user_num, item_num, ui_graph, args)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2rg)
        lr_scheduler = StepwiseLR(opt, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decay_rate)
        best_score = patience_cnt = 0

        for epoch in range(1, args.max_epochs):
            st = time()
            train_loss = train(model, opt, lr_scheduler, train_loader, args)
            print('Epoch:{} Train Loss={:.4f} Time={:.2f}s LR={:.8f}'.format(
                epoch, train_loss, time() - st, lr_scheduler.get_lr()))

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

        print('Testing...')
        model.load_state_dict(torch.load(model_path))
        test_metrics = evaluate(model, test_loader, eval_users)
        hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = test_metrics
        logger.info('Iter={} Tst HR(5/10/20)={:.4f}/{:.4f}/{:.4f} NDCG(5/10/20)={:.4f}/{:.4f}/{:.4f}'
                    .format(r, hr5, hr10, hr20, ndcg5, ndcg10, ndcg20))
        metrics_list.append(test_metrics)

    metrics = np.array(metrics_list)
    means = metrics.mean(axis=0)
    stds = metrics.std(axis=0)
    print(f'{args.model}_{args.dataset} Test Summary:')
    logger.info('Mean hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        means[0], means[1], means[2], means[3], means[4], means[5]))
    logger.info('Std  hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        stds[0], stds[1], stds[2], stds[3], stds[4], stds[5]))
    logger.info("Done")


if __name__ == '__main__':
    main()
