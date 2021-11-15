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
import joblib
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
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


def load_ds(dataset, args):
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
    print("HH")
    train_batches = load_pkl(f'datasets/{dataset}/processed_train_set_n100_bs1024_sl50_nl20meta10timelimit2500000.pkl')
    print("GG")
    uu_elist = loadmat(f'datasets/{dataset}/trust.mat')['trust']
    g = nx.Graph()
    g.add_nodes_from(list(range(user_num)))
    g.add_edges_from(uu_elist)
    g.add_edges_from([[u, u] for u in g.nodes])  # add self-loop to avoid NaN attention scores
    u2u = nx.to_dict_of_lists(g)

    print(f'Loaded {dataset} dataset with {user_num} users, {item_num} items, '
          f'{len(df.values)} u2i, {len(uu_elist)} u2u. ')

    print('Average neighbors: {:.4f}'.format(np.mean([len(v) for k, v in u2u.items()])))

    all_d = np.load(f'./datasets/{args.dataset}/{args.time_limit}process_data.npz', allow_pickle=True)
    item_train = all_d['item_train'][()]

    return df, u2u, train_batches, user_num, item_num, item_train


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


# def evaluate(model, dataset, args, item_train, sample_size=100, is_test=False):
#     model.eval()
#     [user_train, user_valid, user_test, u2u, user_num, item_num] = copy.deepcopy(dataset)
#
#     test_user = 0.0
#     hr5 = hr10 = hr20 = ndcg5 = ndcg10 = ndcg20 = 0.0
#
#     for user in tqdm(range(1, user_num)):
#         if len(user_train[user]) < 1 or len(user_valid[user]) < 1: continue
#
#         seq = torch.zeros((args.seq_maxlen,))
#         time_splits = torch.zeros((args.seq_maxlen,))
#         meta = torch.zeros((args.seq_maxlen, args.meta_maxlen))
#         idx = args.seq_maxlen - 1
#
#         if is_test:  # append the valid item
#             seq[idx] = user_valid[user][0][0]
#             time_splits[idx] = user_valid[user][0][1]
#             tmp_list = []
#             for i in item_train[seq[idx]]:
#                 if abs(i[1] - time_splits[idx]) < args.time_limit:
#                     tmp_list.append(i[0])
#             if len(tmp_list) >= args.meta_maxlen:
#                 meta[idx] = torch.from_numpy(np.array(np.random.choice(tmp_list, size=args.meta_maxlen), dtype=np.int64))
#             else:
#                 for i, tmp_user in enumerate(tmp_list):
#                     meta[idx, i] = tmp_user
#
#             idx -= 1
#
#         for item, time_stamp in reversed(user_train[user]):
#             seq[idx] = item
#             time_splits[idx] = time_stamp
#             tmp_list = []
#             for i in item_train[seq[idx]]:
#                 if abs(i[1] - time_splits[idx]) < args.time_limit:
#                     tmp_list.append(i[0])
#             if len(tmp_list) >= args.meta_maxlen:
#                 meta[idx] = torch.from_numpy(
#                     np.array(np.random.choice(tmp_list, size=args.meta_maxlen), dtype=np.int64))
#             else:
#                 for i, tmp_user in enumerate(tmp_list):
#                     meta[idx, i] = tmp_user
#             idx -= 1
#             if idx == -1: break
#
#         rated_iids = set(user_train[user][:, 0].tolist())
#         rated_iids.add(0)
#
#         if is_test:
#             eval_iid = [user_test[user][0][0]]
#         else:
#             eval_iid = [user_valid[user][0][0]]
#
#         for _ in range(sample_size - 1):
#             t = np.random.randint(1, item_num)
#             while t in rated_iids: t = np.random.randint(1, item_num)
#             eval_iid.append(t)
#         eval_iid = torch.from_numpy(np.array(eval_iid))
#
#         nbr = get_nbr(u2u, user, args.nbr_maxlen)
#         nbr = torch.from_numpy(nbr)
#         nbr_iid = get_nbr_iids(user_train, user, nbr, time_splits)
#         nbr_iid = torch.from_numpy(nbr_iid)
#
#         user = torch.LongTensor([user])
#         seq = seq.unsqueeze(0).long()
#         nbr = nbr.unsqueeze(0).long()
#         nbr_iid = nbr_iid.unsqueeze(0).long()
#         eval_iid = eval_iid.unsqueeze(0).long()
#         meta = meta.unsqueeze(0).long()
#
#         eval_input = [user, seq, nbr, nbr_iid, eval_iid, meta]
#         logits = model.dual_predict(eval_input)  # - for 1st argsort DESC
#         rank = (-1.0 * logits).argsort().argsort()[0].item()
#         test_user += 1
#
#         if rank < 5:
#             hr5 += 1
#             ndcg5 += 1 / np.log2(rank + 2)
#         if rank < 10:
#             hr10 += 1
#             ndcg10 += 1 / np.log2(rank + 2)
#         if rank < 20:
#             hr20 += 1
#             ndcg20 += 1 / np.log2(rank + 2)
#
#     hr5 /= test_user
#     hr10 /= test_user
#     hr20 /= test_user
#
#     ndcg5 /= test_user
#     ndcg10 /= test_user
#     ndcg20 /= test_user
#
#     return hr5, hr10, hr20, ndcg5, ndcg10, ndcg20

def evaluate(model, dataset, args, item_train, sample_size=100, is_test=False):
    model.eval()
    [user_train, user_valid, user_test, u2u, user_num, item_num] = copy.deepcopy(dataset)
    all_data = np.load(f'datasets/{args.dataset}/user_seq_pos_nbr_nbriid_meta_avameta.npz', allow_pickle=True)
    seq_list = all_data['seq_list']
    nbr_list = all_data['nbr_list']
    nbr_iid_list = all_data['nbr_iid_list']
    meta_list = all_data['meta_list']
    test_user = 0.0
    hr5 = hr10 = hr20 = ndcg5 = ndcg10 = ndcg20 = 0.0

    for user in tqdm(range(1, user_num)):
        if len(user_train[user]) < 1 or len(user_valid[user]) < 1: continue

        rated_iids = set(user_train[user][:, 0].tolist())
        rated_iids.add(0)

        if is_test:
            eval_iid = [user_test[user][0][0]]
        else:
            eval_iid = [user_valid[user][0][0]]

        for _ in range(sample_size - 1):
            t = np.random.randint(1, item_num)
            while t in rated_iids: t = np.random.randint(1, item_num)
            eval_iid.append(t)
        eval_iid = torch.from_numpy(np.array(eval_iid))

        # nbr = get_nbr(u2u, user, args.nbr_maxlen)
        # nbr = torch.from_numpy(nbr)
        # nbr_iid = get_nbr_iids(user_train, user, nbr, time_splits)
        # nbr_iid = torch.from_numpy(nbr_iid)

        user = torch.LongTensor([user])
        seq = torch.tensor(seq_list[user]).unsqueeze(0).long()
        nbr = torch.tensor(nbr_list[user]).unsqueeze(0).long()
        nbr_iid = torch.tensor(nbr_iid_list[user].todense()).unsqueeze(0).long()
        eval_iid = eval_iid.unsqueeze(0).long()
        meta = torch.tensor(meta_list[user]).unsqueeze(0).long()

        eval_input = [user, seq, nbr, nbr_iid, eval_iid, meta]
        logits = model.dual_predict(eval_input)  # - for 1st argsort DESC
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


class CRFGNN(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(CRFGNN, self).__init__()
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

        # Fuse Layer
        self.seq_lin = nn.Linear(edim + edim//8 + edim + edim//8, edim)

        # LSTM Block, encode user neighbors hist item
        self.rnn = nn.GRU(input_size=edim, hidden_size=edim, num_layers=1, batch_first=True, dropout=droprate)
        self.meta_rnn = nn.GRU(input_size=edim, hidden_size=edim//8, num_layers=1, batch_first=True, dropout=droprate)

        # Social Aggregation Block
        self.nbr_item_fsue_lin = nn.Linear(edim + edim, edim)
        self.nbr_ffn_layernom = nn.LayerNorm(edim, eps=1e-8)
        self.nbr_ffn = FFN(edim, args.droprate)
        self.nbr_last_layernorm = nn.LayerNorm(edim, eps=1e-8)

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5/user_num, b=0.5/user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5/item_num, b=0.5/item_num)

        if args.use_pos_emb:
            self.posn_embs = nn.Embedding(args.seq_maxlen, edim, padding_idx=0)
            nn.init.uniform_(self.posn_embs.weight, a=-0.5/args.seq_maxlen, b=0.5/args.seq_maxlen)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)

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

    def nbr2feat(self, uid, nbr, nbr_iid):

        # Get mask and neighbors length
        batch_size, nbr_maxlen, seq_maxlen = nbr_iid.shape
        nbr_mask = torch.BoolTensor(nbr == 0).to(self.dev)  # B x nl
        nbr_seq_mask = torch.BoolTensor(nbr_iid == 0).to(self.dev)  # B x nl x sl
        nbr_len = (nbr_maxlen - nbr_mask.sum(1))  # B

        # Get embs
        uid = uid.to(self.dev).unsqueeze(-1)  # B x 1
        nbr = nbr.to(self.dev)  # B x nl
        nbr_iid = nbr_iid.to(self.dev)  # B x nl x sl
        user_emb = self.dropout(self.user_embs(uid))  # B x  1 x d
        nbr_emb = self.dropout(self.user_embs(nbr))  # B x nl x d
        nbr_item_emb = self.dropout(self.item_embs(nbr_iid))  # B x nl x sl x d

        # Static Social Network Features
        nbr_emb *= ~nbr_mask.unsqueeze(-1)  # B x nl x d
        nbr_len = nbr_len.view(batch_size, 1, 1)  # B x 1  x 1
        nbr_feat = nbr_emb.sum(dim=1, keepdim=True) / nbr_len  # B x 1  x d

        # Temporal Neighbor-Items Features
        nbr_seq_mask = nbr_seq_mask.unsqueeze(-1)  # B x nl x sl x 1
        nbr_seq_mask = nbr_seq_mask.permute(0, 2, 1, 3)  # B x sl x nl x 1
        nbr_item_emb = nbr_item_emb.permute(0, 2, 1, 3)  # B x sl x nl x d
        nbr_item_emb *= ~nbr_seq_mask  # B x sl x nl x d
        nbr_seq_len = (seq_maxlen - nbr_seq_mask.sum(dim=2))  # B x sl x 1
        nbr_seq_feat = nbr_item_emb.sum(dim=2) / nbr_seq_len  # B x sl x d
        nbr_seq_feat, _ = self.rnn(nbr_seq_feat)  # B x sl x d

        nbr_feat = nbr_feat.expand_as(nbr_seq_feat)
        nbr_feat = self.nbr_item_fsue_lin(torch.cat([nbr_feat, nbr_seq_feat], dim=-1))
        nbr_feat = self.nbr_ffn_layernom(nbr_feat)
        nbr_feat = self.nbr_ffn(nbr_feat)
        nbr_feat = self.nbr_last_layernorm(nbr_feat)

        return nbr_feat

    def meta2feat(self, uid, seq, meta):
        seq_len = seq.shape[-1]
        user_num = uid.shape[0]
        meta_maxlen = meta.shape[-1]
        # print("st uid",uid.shape,'seq',seq.shape,'meta',meta.shape)
        uid = uid.unsqueeze(-1).unsqueeze(-1)#B * 1 * 1
        seq = seq.unsqueeze(-1)#B * sl *1
        meta_mask = torch.BoolTensor(meta != 0).sum(-1).to(self.dev)#B*sl
        user_emb = self.dropout(self.user_embs(uid.to(self.dev)))#B*1*1*dim
        seq_emb = self.dropout(self.item_embs(seq.to(self.dev)))#B*sl*1*dim
        meta_emb = self.dropout(self.user_embs(meta.to(self.dev))).unsqueeze(-2)#B * sl * meta_maxlen * 1 * dim
        # print("emb user",user_emb.shape,'seq',seq_emb.shape,'meta',meta_emb.shape)
        meta_path_emb = torch.cat((seq_emb, user_emb.expand_as(seq_emb)), dim=-2).unsqueeze(-3).expand(-1,-1,meta_maxlen,-1,-1)#B*sl*1*2*dim
        # print('fi_meta_path_emb',meta_path_emb.shape)
        meta_path_emb = torch.cat((meta_emb,meta_path_emb), dim=-2)#B*sl*meta_maxlen*3*dim   3为metapath长度 u-i-u
        # print('second_meta_path_emb',meta_path_emb.shape)
        meta_path_emb = meta_path_emb.view(-1, 3, self.edim)

        meta_path_emb, _ = self.meta_rnn(meta_path_emb)
        meta_path_emb = meta_path_emb.view(user_num,seq_len,meta_maxlen,3,-1)

        tmp_user_emb = meta_path_emb[:,:,:,2,:].sum(1).sum(1)# B*dim
        # print('tmp_user_emb',tmp_user_emb.shape)
        tmp_item_emb = meta_path_emb[:,:,:,1,:].sum(2)#B*sl*dim
        # print('tmp_item_emb',tmp_item_emb.shape)
        user_mask = meta_mask.sum(-1,keepdim=True)#B*1
        item_mask = meta_mask.unsqueeze(-1)#B*sl*1
        # print('user mask', user_mask.shape,'item_mask',item_mask.shape)
        user_mask[torch.where(user_mask==0)] = 1.0
        item_mask[torch.where(item_mask==0)] = 1.0
        fin_user_emb = tmp_user_emb/user_mask # B
        fin_item_emb = tmp_item_emb/item_mask #B * sl * dim
        # print('fin_user_emb',fin_user_emb.shape,'fin_item_emb',fin_item_emb.shape)
        return fin_user_emb, fin_item_emb

    def pred(self, hu, hi):
        if len(hu.shape) != len(hi.shape):
            hu = hu.unsqueeze(1).expand_as(hi)
        logits = (hu * hi).sum(dim=-1)
        return logits

    def dual_pred(self, seq_hu, nbr_hu, hi):
        seq_logits = (seq_hu * hi).sum(dim=-1)
        nbr_logits = (nbr_hu * hi).sum(dim=-1)
        return seq_logits + nbr_logits

    def dual_forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid, meta = batch

        meta_user_emb, meta_item_emb = self.meta2feat(uid, seq, meta)
        uid = uid.unsqueeze(1).expand_as(seq)
        user_emb = self.dropout(self.user_embs(uid.to(self.dev)))

        # Encode user behavior sequence
        seq_feat = self.seq2feat(seq)  # B x sl x d
        meta_user_emb = meta_user_emb.unsqueeze(-2).expand(-1, seq.shape[-1], -1)
        seq_feat = self.seq_lin(torch.cat([seq_feat, meta_item_emb, user_emb, meta_user_emb], dim=-1))

        # Propagate user intent to his neighbors through time
        nbr_feat = self.nbr2feat(uid, nbr, nbr_iid)  # B x sl x d

        # CRF Layer (Dual score predictor)
        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))  # B x sl x d
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))  # B x sl x ns x d

        pos_logits = self.dual_pred(seq_feat, nbr_feat, pos_hi)  # B x sl
        pos_logits = pos_logits.unsqueeze(-1)                    # B x sl x 1

        seq_feat = seq_feat.unsqueeze(-2).expand_as(neg_hi)      # B x sl x ns x d
        nbr_feat = nbr_feat.unsqueeze(-2).expand_as(neg_hi)      # B x sl x ns x d
        neg_logits = self.dual_pred(seq_feat, nbr_feat, neg_hi)  # B x sl x ns

        return pos_logits, neg_logits, user_emb, pos_hi, neg_hi

    def dual_predict(self, eval_batch):
        self.eval()
        with torch.no_grad():
            uid, seq, nbr, nbr_iid, eval_iid, meta = eval_batch
            user_emb = self.user_embs(uid.to(self.dev))
            meta_user_emb, meta_item_emb = self.meta2feat(uid, seq, meta)
            meta_item_emb = meta_item_emb[:, -1, :]
            seq_feat = self.seq2feat(seq)[:, -1, :]  # B x d
            seq_feat = self.seq_lin(torch.cat([seq_feat, meta_item_emb, user_emb, meta_user_emb], dim=-1))
            nbr_feat = self.nbr2feat(uid, nbr, nbr_iid)[:, -1, :]  # B x d
            hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
            seq_feat = seq_feat.unsqueeze(1).expand_as(hi)
            nbr_feat = nbr_feat.unsqueeze(1).expand_as(hi)

            logits = self.dual_pred(seq_feat, nbr_feat, hi)

            return logits.squeeze()

    def get_parameters(self):
        param_list = [
            {'params': self.item_attn_layernorm.parameters()},
            {'params': self.item_attn_layer.parameters()},
            {'params': self.item_ffn_layernorm.parameters()},
            {'params': self.item_ffn.parameters()},
            {'params': self.item_last_layernorm.parameters()},
            {'params': self.seq_lin.parameters()},

            {'params':  self.nbr_item_fsue_lin.parameters()},
            {'params': self.nbr_ffn_layernom.parameters()},
            {'params': self.nbr_ffn.parameters()},
            {'params': self.nbr_last_layernorm.parameters()},
            {'params': self.rnn.parameters()},

            {'params': self.user_embs.parameters(), 'weight_decay': 0},
            {'params': self.item_embs.parameters(), 'weight_decay': 0},
        ]

        if self.args.use_pos_emb:
            param_list.append({'params': self.posn_embs.parameters(), 'weight_decay': 0})

        return param_list


class CRFGNN_gat(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(CRFGNN_gat, self).__init__()
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

        # Fuse Layer
        self.seq_lin = nn.Linear(edim + edim//8 + edim + edim//8, edim)

        # LSTM Block, encode user neighbors hist item
        self.rnn = nn.GRU(input_size=edim, hidden_size=edim, num_layers=1, batch_first=True, dropout=droprate)
        self.meta_rnn = nn.GRU(input_size=edim, hidden_size=edim//8, num_layers=1, batch_first=True, dropout=droprate)

        # Social Aggregation Block
        self.user_attn0 = nn.Linear(edim + edim, edim, bias=False)
        self.user_attn1 = nn.Linear(edim, 1, bias=False)
        self.item_attn0 = nn.Linear(edim + edim, edim, bias=False)
        self.item_attn1 = nn.Linear(edim, 1, bias=False)

        self.nbr_item_fsue_lin = nn.Linear(edim + edim, edim)
        self.nbr_ffn_layernom = nn.LayerNorm(edim, eps=1e-8)
        self.nbr_ffn = FFN(edim, args.droprate)
        self.nbr_last_layernorm = nn.LayerNorm(edim, eps=1e-8)

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5/user_num, b=0.5/user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5/item_num, b=0.5/item_num)

        if args.use_pos_emb:
            self.posn_embs = nn.Embedding(args.seq_maxlen, edim, padding_idx=0)
            nn.init.uniform_(self.posn_embs.weight, a=-0.5/args.seq_maxlen, b=0.5/args.seq_maxlen)

        self.act = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(args.droprate)

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

    def nbr2feat(self, uid, nbr, nbr_iid):

        # Get mask and neighbors length
        batch_size, nbr_maxlen, seq_maxlen = nbr_iid.shape
        nbr_mask = torch.BoolTensor(nbr == 0).to(self.dev)  # B x nl
        nbr_seq_mask = torch.BoolTensor(nbr_iid == 0).to(self.dev)  # B x nl x sl

        # Get embs
        uid = uid.to(self.dev).unsqueeze(-1)  # B x 1
        nbr = nbr.to(self.dev)  # B x nl
        nbr_iid = nbr_iid.to(self.dev)  # B x nl x sl
        user_emb = self.dropout(self.user_embs(uid))  # B x  1 x d
        nbr_emb = self.dropout(self.user_embs(nbr))  # B x nl x d
        nbr_item_emb = self.dropout(self.item_embs(nbr_iid))  # B x nl x sl x d

        # Static Social Network Features
        user_attn_self_emb = user_emb.expand_as(nbr_emb)  # B x nl x d
        user_attn_h = self.user_attn1(self.leaky_relu(self.dropout(self.item_attn0(
            torch.cat([user_attn_self_emb, nbr_emb], dim=-1)))))  # B x nl x 1
        user_attn_score = user_attn_h + -1e9 * nbr_mask.unsqueeze(-1)
        user_attn_a = F.softmax(user_attn_score, dim=1)
        nbr_feat = (user_attn_a * nbr_emb).sum(dim=1, keepdims=True)  # B x 1 x d

        # Temporal Neighbor-Items Features
        item_attn_self_emb = user_emb.unsqueeze(1).expand_as(nbr_item_emb)  # B x nl x sl x d
        item_attn_h = self.item_attn1(self.leaky_relu(self.dropout(self.item_attn0(
            torch.cat([item_attn_self_emb, nbr_item_emb], dim=-1)))))  # B x nl x sl x 1
        item_attn_score = item_attn_h + -1e9 * nbr_seq_mask.unsqueeze(-1)
        item_attn_a = F.softmax(item_attn_score, dim=1)
        nbr_seq_feat = (item_attn_a * nbr_item_emb).sum(dim=1)  # B x sl x d

        # GRU
        nbr_seq_feat, _ = self.rnn(nbr_seq_feat)  # B x sl x d

        nbr_feat = nbr_feat.expand_as(nbr_seq_feat)
        nbr_feat = self.nbr_item_fsue_lin(torch.cat([nbr_feat, nbr_seq_feat], dim=-1))
        nbr_feat = self.nbr_ffn_layernom(nbr_feat)
        nbr_feat = self.nbr_ffn(nbr_feat)
        nbr_feat = self.nbr_last_layernorm(nbr_feat)

        return nbr_feat

    def meta2feat(self, uid, seq, meta):
        seq_len = seq.shape[-1]
        user_num = uid.shape[0]
        meta_maxlen = meta.shape[-1]
        # print("st uid",uid.shape,'seq',seq.shape,'meta',meta.shape)
        uid = uid.unsqueeze(-1).unsqueeze(-1)#B * 1 * 1
        seq = seq.unsqueeze(-1)#B * sl *1
        meta_mask = torch.BoolTensor(meta != 0).sum(-1).to(self.dev)#B*sl
        user_emb = self.dropout(self.user_embs(uid.to(self.dev)))#B*1*1*dim
        seq_emb = self.dropout(self.item_embs(seq.to(self.dev)))#B*sl*1*dim
        meta_emb = self.dropout(self.user_embs(meta.to(self.dev))).unsqueeze(-2)#B * sl * meta_maxlen * 1 * dim
        # print("emb user",user_emb.shape,'seq',seq_emb.shape,'meta',meta_emb.shape)
        meta_path_emb = torch.cat((seq_emb, user_emb.expand_as(seq_emb)), dim=-2).unsqueeze(-3).expand(-1,-1,meta_maxlen,-1,-1)#B*sl*1*2*dim
        # print('fi_meta_path_emb',meta_path_emb.shape)
        meta_path_emb = torch.cat((meta_emb,meta_path_emb), dim=-2)#B*sl*meta_maxlen*3*dim   3为metapath长度 u-i-u
        # print('second_meta_path_emb',meta_path_emb.shape)
        meta_path_emb = meta_path_emb.view(-1, 3, self.edim)

        meta_path_emb, _ = self.meta_rnn(meta_path_emb)
        meta_path_emb = meta_path_emb.view(user_num,seq_len,meta_maxlen,3,-1)

        tmp_user_emb = meta_path_emb[:,:,:,2,:].sum(1).sum(1)# B*dim
        # print('tmp_user_emb',tmp_user_emb.shape)
        tmp_item_emb = meta_path_emb[:,:,:,1,:].sum(2)#B*sl*dim
        # print('tmp_item_emb',tmp_item_emb.shape)
        user_mask = meta_mask.sum(-1,keepdim=True)#B*1
        item_mask = meta_mask.unsqueeze(-1)#B*sl*1
        # print('user mask', user_mask.shape,'item_mask',item_mask.shape)
        user_mask[torch.where(user_mask==0)] = 1.0
        item_mask[torch.where(item_mask==0)] = 1.0
        fin_user_emb = tmp_user_emb/user_mask # B
        fin_item_emb = tmp_item_emb/item_mask #B * sl * dim
        # print('fin_user_emb',fin_user_emb.shape,'fin_item_emb',fin_item_emb.shape)
        return fin_user_emb, fin_item_emb

    def pred(self, hu, hi):
        if len(hu.shape) != len(hi.shape):
            hu = hu.unsqueeze(1).expand_as(hi)
        logits = (hu * hi).sum(dim=-1)
        return logits

    def dual_pred(self, seq_hu, nbr_hu, hi):
        seq_logits = (seq_hu * hi).sum(dim=-1)
        nbr_logits = (nbr_hu * hi).sum(dim=-1)
        return seq_logits + nbr_logits

    def dual_forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid, meta = batch
        meta_user_emb, meta_item_emb = self.meta2feat(uid, seq, meta)
        # Propagate user intent to his neighbors through time
        nbr_feat = self.nbr2feat(uid, nbr, nbr_iid)  # B x sl x d

        # Encode user behavior sequence
        seq_feat = self.seq2feat(seq)  # B x sl x d
        uid = uid.unsqueeze(1).expand_as(seq)
        user_emb = self.dropout(self.user_embs(uid.to(self.dev)))
        meta_user_emb = meta_user_emb.unsqueeze(-2).expand(-1, seq.shape[-1], -1)
        seq_feat = self.seq_lin(torch.cat([seq_feat, meta_item_emb, user_emb, meta_user_emb], dim=-1))

        # CRF Layer (Dual score predictor)
        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))  # B x sl x d
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))  # B x sl x ns x d

        pos_logits = self.dual_pred(seq_feat, nbr_feat, pos_hi)  # B x sl
        pos_logits = pos_logits.unsqueeze(-1)                    # B x sl x 1

        seq_feat = seq_feat.unsqueeze(-2).expand_as(neg_hi)      # B x sl x ns x d
        nbr_feat = nbr_feat.unsqueeze(-2).expand_as(neg_hi)      # B x sl x ns x d
        neg_logits = self.dual_pred(seq_feat, nbr_feat, neg_hi)  # B x sl x ns

        return pos_logits, neg_logits, user_emb, pos_hi, neg_hi

    def dual_predict(self, eval_batch):
        self.eval()
        with torch.no_grad():
            uid, seq, nbr, nbr_iid, eval_iid, meta = eval_batch
            user_emb = self.user_embs(uid.to(self.dev))
            meta_user_emb, meta_item_emb = self.meta2feat(uid, seq, meta)
            meta_item_emb = meta_item_emb[:, -1, :]
            seq_feat = self.seq2feat(seq)[:, -1, :]  # B x d
            seq_feat = self.seq_lin(torch.cat([seq_feat, meta_item_emb, user_emb, meta_user_emb], dim=-1))
            nbr_feat = self.nbr2feat(uid, nbr, nbr_iid)[:, -1, :]  # B x d
            hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
            seq_feat = seq_feat.unsqueeze(1).expand_as(hi)
            nbr_feat = nbr_feat.unsqueeze(1).expand_as(hi)

            logits = self.dual_pred(seq_feat, nbr_feat, hi)

            return logits.squeeze()

    def get_parameters(self):
        param_list = [
            {'params': self.item_attn_layernorm.parameters()},
            {'params': self.item_attn_layer.parameters()},
            {'params': self.item_ffn_layernorm.parameters()},
            {'params': self.item_ffn.parameters()},
            {'params': self.item_last_layernorm.parameters()},
            {'params': self.seq_lin.parameters()},

            {'params': self.user_attn0.parameters()},
            {'params': self.user_attn1.parameters()},
            {'params': self.item_attn0.parameters()},
            {'params': self.item_attn1.parameters()},

            {'params':  self.nbr_item_fsue_lin.parameters()},
            {'params': self.nbr_ffn_layernom.parameters()},
            {'params': self.nbr_ffn.parameters()},
            {'params': self.nbr_last_layernorm.parameters()},
            {'params': self.rnn.parameters()},

            {'params': self.user_embs.parameters(), 'weight_decay': 0},
            {'params': self.item_embs.parameters(), 'weight_decay': 0},
        ]

        if self.args.use_pos_emb:
            param_list.append({'params': self.posn_embs.parameters(), 'weight_decay': 0})

        return param_list


# ---------------------------------------------------------------------------------------------------------------------
# Train and Evaluate

def parse_sampled_batch(batch, num_item, args):
    uid, seq, pos, neg, nbr, nbr_iid, meta = batch
    uid = torch.from_numpy(uid).long()
    seq = torch.from_numpy(seq).long()
    pos = torch.from_numpy(pos).long()
    meta = torch.from_numpy(meta).long()
    # neg = torch.from_numpy(neg).long()

    neg = torch.from_numpy(
        np.random.randint(low=1, high=num_item, size=(args.batch_size, args.seq_maxlen, args.neg_size))
    ).long()

    pos_mask = pos.unsqueeze(-1).clone()
    pos_mask[pos != 0] = 1
    neg *= pos_mask

    nbr = torch.from_numpy(nbr).long()
    tmp_list = list()
    for tmp in nbr_iid:
        tmp_list.append(tmp.todense())
    nbr_iid = torch.from_numpy(np.array(tmp_list)).long()
    batch = [uid, seq, pos, neg, nbr, nbr_iid, meta]
    indices = torch.where(pos != 0)
    return batch, indices


def train(model, opt, shdlr, train_batches, cur_idx, num_batch, num_item, args):
    model.train()
    total_loss = 0.0

    for batch in train_batches[cur_idx:cur_idx+num_batch]:
        parsed_batch, indices = parse_sampled_batch(batch[0], num_item, args)
        opt.zero_grad()
        shdlr.step()
        pos_logits, neg_logits, user_emb, pos_hi, neg_hi = model.dual_forward(parsed_batch)

        # Label Loss
        loss = 0.0
        if args.loss_type == 'bce':
            loss += F.binary_cross_entropy_with_logits(pos_logits[indices], torch.ones_like(pos_logits)[indices]) + \
                    F.binary_cross_entropy_with_logits(neg_logits[indices], torch.zeros_like(neg_logits)[indices])
        elif args.loss_type == 'bpr':  # single negative item
            loss += F.softplus(neg_logits[indices] - pos_logits[indices]).mean()
        elif args.loss_type == 'sfm':  # multiple negative items
            uid, seq, pos, neg, nbr, nbr_iid, meta = parsed_batch
            all_items = torch.cat([pos.unsqueeze(-1), neg], dim=-1)  # B x sl x (1 + ns)
            all_indices = torch.where(all_items != 0)
            logits = torch.cat([pos_logits, neg_logits], dim=-1)  # B x sl x (1 + ns)
            logits = logits[all_indices].view(-1, 1 + args.neg_size)
            device = torch.device(f'{args.device}')
            labels = torch.zeros((logits.shape[0])).long().to(device)
            loss += F.cross_entropy(logits, labels)

        # Embedding Reg term
        user_norm = user_emb.norm(dim=-1).pow(2).mean()
        item_norm = pos_hi.norm(dim=-1).pow(2).mean() + neg_hi.norm(dim=-1).pow(2).mean()
        emb_reg_loss = args.emb_reg * 0.5 * (user_norm + item_norm)
        loss += emb_reg_loss
        loss.backward()
        opt.step()
        total_loss += loss.item()

    return total_loss / num_batch

def main():
    parser = argparse.ArgumentParser(description='TEA')
    parser.add_argument('--dataset', default='Epinions')
    parser.add_argument('--model', default='TEA')

    # Model Config
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--use_pos_emb', type=int, default=True)
    parser.add_argument('--seq_maxlen', type=int, default=50, help='fixed, or change with sampled train_batches')
    parser.add_argument('--nbr_maxlen', type=int, default=20, help='fixed, or change with sampled train_batches')
    parser.add_argument('--neg_size', type=int, default=50, help='Negative samples number')

    # Train Config
    parser.add_argument('--batch_size', type=int, default=1024, help='fixed, or change with sampled train_batches')
    parser.add_argument('--droprate', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.85)
    parser.add_argument('--l2rg', type=float, default=5e-4)
    parser.add_argument('--emb_reg', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--check_epoch', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=100)
    parser.add_argument('--loss_type', default='sfm', help='bce/bpr/sfm')
    parser.add_argument('--num_workers', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--time_limit', type=int, default=2500000)
    parser.add_argument('--meta_maxlen', type=int, default=10)

    # Something else
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--test_time', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=False)
    parser.add_argument('--model_type', type=str, default='gcn')
    args = parser.parse_args()

    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    model_path = f'saved_models/{args.model}_{args.model_type}_{args.dataset}_{timestr}best.pth'
    logger = get_logger(os.path.join('logs', f'{args.model}_best.log'))
    logger.info(args)
    device = torch.device(args.device)

    df, u2u, train_batches, user_num, item_num, item_train = load_ds(args.dataset, args)
    user_train, user_valid, user_test = data_partition(df)
    dataset = [user_train, user_valid, user_test, u2u, user_num, item_num]
    num_batch = len(user_train) // args.batch_size
    print('load fin')

    metrics_list = []
    for r in range(args.repeat):
        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        if args.model_type == 'gcn':
            model = CRFGNN(user_num, item_num, args)
        else:
            model = CRFGNN_gat(user_num, item_num, args)
        model = model.to(device)
        opt = torch.optim.Adam(model.get_parameters(), lr=args.lr, weight_decay=args.l2rg)

        lr_scheduler = StepwiseLR(opt, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decay_rate)
        cur_idx = best_score = patience_cnt = 0

        for epoch in range(1, args.max_epochs):
            st = time()
            train_loss = train(model, opt, lr_scheduler, train_batches, cur_idx, num_batch, item_num, args)
            print('Epoch:{} Train Loss={:.4f} Time={:.2f}s LR={:.8f}'.format(
                epoch, train_loss, time() - st, lr_scheduler.get_lr()))

            if cur_idx < (2100-num_batch):
                cur_idx += num_batch
            else:
                cur_idx = 0
                np.random.shuffle(train_batches)

            if epoch % args.check_epoch == 0 and epoch >= args.start_epoch:
                val_metrics = evaluate(model, dataset, args, item_train, is_test=False)
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
        test_metrics = evaluate(model, dataset, args, item_train, is_test=True)
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
    res_file = 'Epinions'+args.model_type+'res.txt'
    with open(res_file, 'a') as f:
        f.write(f'Epinions {args.model_type} repeat{args.repeat} epoch{args.max_epochs} batch_size{args.batch_size} lr{args.lr} loss_type{args.loss_type} drop{args.droprate}\n')
        f.write('Mean hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}\n'.format(
            means[0], means[1], means[2], means[3], means[4], means[5]))
        f.write('Std  hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}\n'.format(
            stds[0], stds[1], stds[2], stds[3], stds[4], stds[5]))

if __name__ == '__main__':
    main()
