import argparse
import copy
import logging
import os
import re
import sys
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
import pickle as pkl
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=1000000)
import scipy.sparse as sp


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


def save_pkl(file, obj):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)


def load_pkl(file):
    with open(file, 'rb') as f:
        data = pkl.load(f)
    return data


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


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def get_nbr(u2u, user, nbr_maxlen):
    nbr = np.zeros([nbr_maxlen, ], dtype=np.int64)
    nbr_len = len(u2u[user])
    if nbr_len == 0:
        pass
    elif nbr_len > nbr_maxlen:
        np.random.shuffle(u2u[user])
        nbr[:] = u2u[user][:nbr_maxlen]
    else:
        nbr[:nbr_len] = u2u[user]

    return nbr


def get_nbr_iids(user_train, user, nbrs, time_splits):
    start_idx = np.nonzero(time_splits)[0][0]
    user_first_ts = time_splits[start_idx]
    user_last_ts = time_splits[-1]
    nbr_maxlen = len(nbrs)
    seq_maxlen = len(time_splits)
    nbrs_iids = np.zeros((nbr_maxlen, seq_maxlen))

    for i, nbr in enumerate(nbrs):
        if nbr == 0 or nbr == user:
            continue

        nbr_hist = user_train[nbr]

        if len(nbr_hist) == 0:
            continue

        nbr_first_ts = nbr_hist[0][1]
        nbr_last_ts = nbr_hist[-1][1]

        if nbr_first_ts > user_last_ts or nbr_last_ts <= user_first_ts:
            continue

        sample_list = list()
        for j in range(start_idx + 1, seq_maxlen):
            start_time = time_splits[j - 1]
            end_time = time_splits[j]

            if start_time != end_time:
                sample_list = list(filter(None, map(
                    lambda x: x[0] if x[1] > start_time and x[1] <= end_time else None, nbr_hist
                )))

            if len(sample_list):
                # print('st={} et={} sl={}'.format(start_time, end_time, sample_list))
                nbrs_iids[i, j] = np.random.choice(sample_list)

    return nbrs_iids


def sample_function(user_train, u2u, user_num, item_num, batch_size, seq_maxlen, nbr_maxlen, result_queue, seed):
    # print('seed=', seed)

    def sample():
        user = np.random.randint(1, user_num)
        while len(user_train[user]) <= 1: user = np.random.randint(1, user_num)

        seq = np.zeros(seq_maxlen, dtype=np.int64)
        pos = np.zeros(seq_maxlen, dtype=np.int64)
        ts = np.zeros(seq_maxlen, dtype=np.int64)
        neg = np.zeros(seq_maxlen, dtype=np.int64)
        nxt = user_train[user][-1, 0]
        idx = seq_maxlen - 1

        exclude_items = set(user_train[user][:, 0].tolist())
        for (item, time_stamp) in reversed(user_train[user][:-1]):
            seq[idx] = item
            ts[idx] = time_stamp
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, item_num, exclude_items)
            nxt = item
            idx -= 1
            if idx == -1: break

        nbr = get_nbr(u2u, user, nbr_maxlen)
        nbr_iids = get_nbr_iids(user_train, user, nbr, ts)

        return user, seq, pos, neg, nbr, nbr_iids

    np.random.seed(seed)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, user_train, u2u, user_num, item_num, batch_size, seq_maxlen, nbr_maxlen, n_workers=1, seed=0):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        np.random.seed(seed)
        for i in range(n_workers):
            self.processors.append(Process(
                target=sample_function,
                args=(user_train, u2u, user_num, item_num,
                      batch_size, seq_maxlen, nbr_maxlen,
                      self.result_queue, np.random.randint(low=1, high=1e8))))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def preprocess_uir(df, prepro='origin', binary=False, pos_threshold=None, level='ui'):
    # set rating >= threshold as positive samples
    if pos_threshold is not None:
        df = df.query(f'rate >= {pos_threshold}').reset_index(drop=True)

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rate'] = 1.0

    # which type of pre-dataset will use
    if prepro == 'origin':
        pass

    elif prepro.endswith('filter'):
        pattern = re.compile(r'\d+')
        filter_num = int(pattern.findall(prepro)[0])

        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        if level == 'ui':
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()
        else:
            raise ValueError(f'Invalid level value: {level}')

        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid level value: {level}')

        gc.collect()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')

    return df


def load_ds(dataset='Ciao'):
    # Ciao Raw #u2i=284086, #u2u=57544
    # Epin Raw #u2i=922267, #u2u=355813

    rating = pd.DataFrame()

    rating_mat = loadmat(f'datasets/{dataset}/rating_with_timestamp.mat')
    if dataset == 'Ciao':
        rating = rating_mat['rating']
    elif dataset == 'Epinions':
        rating = rating_mat['rating_with_timestamp']

    df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
    df.drop(columns=['cate', 'help'], inplace=True)
    df.drop_duplicates(subset=['user', 'item', 'ts'], keep='first', inplace=True)
    df = preprocess_uir(df, prepro='origin', binary=True, pos_threshold=3)
    df.drop(columns=['rate'], inplace=True)
    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)

    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1

    train_batches = load_pkl(f'datasets/{dataset}/processed_train_set_n100_bs1024_sl50_nl20.pkl')

    uu_elist = loadmat(f'datasets/{dataset}/trust.mat')['trust']
    g = nx.Graph()
    g.add_nodes_from(list(range(user_num)))
    g.add_edges_from(uu_elist)
    g.add_edges_from([[u, u] for u in g.nodes])  # add self-loop to avoid NaN attention scores
    u2u = nx.to_dict_of_lists(g)

    print(f'Loaded {dataset} dataset with {user_num} users, {item_num} items, '
          f'{len(df.values)} u2i, {len(uu_elist)} u2u. ')

    print('Average neighbors: {:.4f}'.format(np.mean([len(v) for k, v in u2u.items()])))

    return df, u2u, train_batches, user_num, item_num


def data_partition(df):
    print('Splitting train/val/test set...')
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)

    user_items_dict = defaultdict(list)

    def apply_fn1(grp):
        key_id = grp['user'].values[0]
        user_items_dict[key_id] = grp[['item', 'ts']].values

    df.groupby('user').apply(apply_fn1)

    for user in user_items_dict:
        nfeedback = len(user_items_dict[user])
        if nfeedback < 5:
            user_train[user] = user_items_dict[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = user_items_dict[user][:-2]
            user_valid[user] = []
            user_valid[user].append(user_items_dict[user][-2])
            user_test[user] = []
            user_test[user].append(user_items_dict[user][-1])

    return user_train, user_valid, user_test


def evaluate(model, dataset, args, sample_size=100, is_test=False):
    model.eval()
    [user_train, user_valid, user_test, u2u, user_num, item_num] = copy.deepcopy(dataset)
    device = torch.device(args.device)
    test_user = 0.0
    hr5 = hr10 = hr20 = ndcg5 = ndcg10 = ndcg20 = 0.0
    all_users, all_items = model.computer()

    for user in tqdm(range(1, user_num)):
        if len(user_train[user]) < 1 or len(user_valid[user]) < 1: continue

        rated_iids = set(user_train[user][:, 0].tolist())
        rated_iids.add(0)

        if is_test: eval_iid = [user_test[user][0][0]]
        else: eval_iid = [user_valid[user][0][0]]

        for _ in range(sample_size - 1):
            t = np.random.randint(1, item_num)
            while t in rated_iids: t = np.random.randint(1, item_num)
            eval_iid.append(t)

        eval_iid = torch.from_numpy(np.array(eval_iid)).long()
        user = torch.LongTensor([user])

        item_emb = all_items[eval_iid].to(device)
        user_emb = all_users[user].to(device)
        user_emb = user_emb.expand_as(item_emb)
        logits = (user_emb * item_emb).sum(dim=-1)

        rank = (-1.0 * logits).argsort().argsort()[0].item()
        test_user += 1

        if rank < 5:
            hr5 += 1
            ndcg5 += 1 / np.log2(rank + 2)
        if rank < 10:
            hr10 += 1
            ndcg10 += 1 / np.log2(rank + 2)
        if rank < 20:
            hr20 += 1
            ndcg20 += 1 / np.log2(rank + 2)

    hr5 /= test_user
    hr10 /= test_user
    hr20 /= test_user

    ndcg5 /= test_user
    ndcg10 /= test_user
    ndcg20 /= test_user

    return hr5, hr10, hr20, ndcg5, ndcg10, ndcg20


def get_ui_graph(dataset, user_train, user_num, item_num):
    saved_path = f'datasets/{dataset}/norm_adj.npz'
    if os.path.exists(saved_path):
        norm_adj = sp.load_npz(saved_path)
        print('Loaded normalized joint rating matrix.')
    else:
        print('Generating sparse rating matrix...', end='')
        train_users = []
        train_items = []

        for user in range(1, user_num):
            items = user_train[user]
            if len(items):
                train_users.extend(len(items) * [user])
                train_items.extend(items[:, 0].tolist())

        rating_maxtrix = sp.csr_matrix(
            (np.ones(len(train_users)), (train_users, train_items)),
            shape=(user_num, item_num), dtype=np.int8).tolil()

        adj_mat = sp.dok_matrix(
            (user_num + item_num, user_num + item_num),
            dtype=np.int8).tolil()

        adj_mat[:user_num, user_num:] = rating_maxtrix
        adj_mat[user_num:, :user_num] = rating_maxtrix.T
        adj_mat = adj_mat.todok().astype(np.float16)

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo()
        sp.save_npz(saved_path, norm_adj)
        print('norm_adj saved at', saved_path)

    row = torch.Tensor(norm_adj.row).long()
    col = torch.Tensor(norm_adj.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(norm_adj.data)
    ui_graph = torch.sparse.FloatTensor(index, data, torch.Size(norm_adj.shape)).coalesce()
    return ui_graph



# ---------------------------------------------------------------------------------------------------------------------
# Model

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

# ---------------------------------------------------------------------------------------------------------------------
# Train and Evaluate

def parse_sampled_batch(batch):
    uid, seq, pos, neg, nbr, nbr_iid = batch
    uid = torch.from_numpy(uid).long()
    pos = torch.from_numpy(pos).long()
    neg = torch.from_numpy(neg).long()
    return uid, pos, neg


def train(model, opt, shdlr, train_batches, cur_idx, num_batch, args):
    model.train()
    total_loss = 0.0
    for batch in train_batches[cur_idx:cur_idx+num_batch]:
        parsed_batch = parse_sampled_batch(batch[0])
        opt.zero_grad()
        loss = model.bpr_loss(parsed_batch)
        loss.backward()
        opt.step()
        shdlr.step()
        total_loss += loss.item()

    return total_loss / num_batch

def main():
    parser = argparse.ArgumentParser(description='LightGCN')
    parser.add_argument('--model', default='LightGCN')
    parser.add_argument('--dataset', default='Epinions')

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
    parser.add_argument('--l2rg', type=float, default=2e-6)
    parser.add_argument('--emb_reg', type=float, default=0.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--check_epoch', type=int, default=20)
    parser.add_argument('--loss_type', default='bpr', help='bce/bpr')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)

    # Something else
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--test_time', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=False)
    args = parser.parse_args()

    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    model_path = f'saved_models/{args.model}_{args.dataset}_{timestr}.pth'
    logger = get_logger(os.path.join('logs', f'{args.model}_{timestr}.log'))
    logger.info(args)
    device = torch.device(args.device)

    df, u2u, train_batches, user_num, item_num = load_ds(args.dataset)
    user_train, user_valid, user_test = data_partition(df)
    dataset = [user_train, user_valid, user_test, u2u, user_num, item_num]
    ui_graph = get_ui_graph(args.dataset, user_train, user_num, item_num)
    num_batch = len(user_train) // args.batch_size

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
        cur_idx = best_score = patience_cnt = 0

        for epoch in range(1, args.max_epochs):
            st = time()
            train_loss = train(model, opt, lr_scheduler, train_batches, cur_idx, num_batch, args)
            print('Epoch:{} Train Loss={:.4f} Time={:.2f}s LR={:.8f}'.format(
                epoch, train_loss, time() - st, lr_scheduler.get_lr()))

            if cur_idx < (2100-num_batch):
                cur_idx += num_batch
            else:
                cur_idx = 0
                np.random.shuffle(train_batches)

            if epoch % args.check_epoch == 0:
                val_metrics = evaluate(model, dataset, args, is_test=False)
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
        test_metrics = evaluate(model, dataset, args, is_test=True)
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
