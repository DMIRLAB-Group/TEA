





# MAGNN metapaths
# user: u-u, u-i-u, u-u-i










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


def get_uui(user_train, user, u2u, mtp_maxlen):
    uui_candidates = list()

    nbrs = u2u[user]

    if len(nbrs) == 0:
        return np.zeros((mtp_maxlen, 2))

    for nbr in nbrs:
        if nbr == 0 or nbr == user:
            continue

        nbr_hist = user_train[nbr]

        if len(nbr_hist) == 0:
            continue

        nbr_iids = nbr_hist[:, 0]
        nbr_uids = np.zeros_like(nbr_iids) + nbr
        tmp = np.hstack(nbr_uids, nbr_iids)
        uui_candidates.append(tmp)
        print('appending...', uui_candidates.shape)

    uui_candidates = np.vstack(uui_candidates)
    np.random.shuffle(uui_candidates)
    print('uui cand =', uui_candidates.shape)

    indices = np.arange(len(uui_candidates))
    if len(uui_candidates) >= mtp_maxlen:
        sampled_indices = np.random.choice(indices)
        uui_arr = uui_candidates[sampled_indices]
    else:
        padding_size = len(uui_candidates)
        padding = np.zeros((padding_size, 2))
        uui_arr = np.vstack([uui_candidates, padding])

    return uui_arr


def get_uiu(user_train, user, u2u, mtp_maxlen):
    uiu_candidates = list()

    # TODO: find item-users
    for item in user_train[user]:
        pass




    pass


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

    test_user = 0.0
    hr5 = hr10 = hr20 = ndcg5 = ndcg10 = ndcg20 = 0.0

    for user in tqdm(range(1, user_num)):
        if len(user_train[user]) < 1 or len(user_valid[user]) < 1: continue

        seq = torch.zeros((args.seq_maxlen,))
        time_splits = torch.zeros((args.seq_maxlen,))
        idx = args.seq_maxlen - 1

        if is_test:  # append the valid item
            seq[idx] = user_valid[user][0][0]
            time_splits[idx] = user_valid[user][0][1]
            idx -= 1

        for item, time_stamp in reversed(user_train[user]):
            seq[idx] = item
            time_splits[idx] = time_stamp
            idx -= 1
            if idx == -1: break

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

        nbr = get_nbr(u2u, user, args.nbr_maxlen)
        nbr = torch.from_numpy(nbr)
        nbr_iid = get_nbr_iids(user_train, user, nbr, time_splits)
        nbr_iid = torch.from_numpy(nbr_iid)

        user = torch.LongTensor([user])
        seq = seq.unsqueeze(0).long()
        nbr = nbr.unsqueeze(0).long()
        nbr_iid = nbr_iid.unsqueeze(0).long()
        eval_iid = eval_iid.unsqueeze(0).long()

        eval_input = [user, seq, nbr, nbr_iid, eval_iid]
        logits = model.predict(eval_input)  # - for 1st argsort DESC
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


class UserEncoder(nn.Module):
    def __init__(self, edim, args):
        super(UserEncoder, self).__init__()
        self.intra_attn_uu  = nn.Linear(edim + edim, 1, bias=False)
        self.intra_attn_uiu = nn.Linear(edim + edim, 1, bias=False)
        self.intra_attn_uui = nn.Linear(edim + edim, 1, bias=False)
        self.inter_attn_0 = nn.Linear(edim, edim//2, bias=True)
        self.inter_attn_1 = nn.Linear(edim//2, 1, bias=False)
        self.dropout = nn.Dropout(args.droprate)
        self.dev = torch.device(args.device)

    def foward(self, user_embs, item_embs, uid, nbr, uiu, uui):

        # self emb
        uid = uid.to(self.dev).unsqueeze(-1)  # B x 1
        self_emb = self.dropout(user_embs(uid))  # B x 1 x d

        # get uu mask
        uu_mask = torch.BoolTensor(nbr == 0).to(self.dev)  # B x nl

        # encode uu
        nbr = nbr.to(self.dev)  # B x nl
        nbr_emb = self.dropout(user_embs(nbr))  # B x nl x d
        uu_feat = (self_emb + nbr_emb) / 2.0  # broadcast at dim=1, mean aggr, B x nl x d

        # calc uu attn & aggr
        uu_h_attn = torch.cat([self_emb.expand_as(uu_feat), uu_feat], dim=-1)  # B x nl x 2d
        uu_attn = self.intra_attn_uu(uu_h_attn)  # B x nl x 1
        uu_attn = F.leaky_relu(uu_attn)
        uu_attn += -1e8 * uu_mask.unsqueeze(dim=-1)  # B x nl x 1
        uu_attn = F.softmax(uu_attn, dim=1)
        uu_feat = (uu_attn * uu_feat).sum(dim=1)  # B x d
        uu_feat = F.relu(uu_feat)

        # get uiu mask
        # uiu = B x ml(metapath_maxlen) x 2 --> [(uid), mp_item_id, mp_user_id]
        uiu_mask = torch.BoolTensor(uiu[:, 0] == 0).to(self.dev)  # B x ml

        # encode uiu
        uiu = uiu.to(self.dev)  # B x ml x 2
        mp_iid, mp_uid = uiu[:, 0], uiu[:, 1]  # B x ml x 1
        mp_item_emb = self.dropout(item_embs(mp_iid))  # B x ml x d
        mp_user_emb = self.dropout(user_embs(mp_uid))  # B x ml x d
        uiu_feat = (self_emb + mp_item_emb + mp_user_emb) / 3.0  # B x ml x d

        # calc uiu attn & aggr
        uiu_h_attn = torch.cat([self_emb.expand_as(uiu_feat), uiu_feat], dim=-1)  # B x ml x 2d
        uiu_attn = self.intra_attn_uiu(uiu_h_attn)  # B x ml x 1
        uiu_attn = F.leaky_relu(uiu_attn)
        uiu_attn += -1e8 * uiu_mask.unsqueeze(dim=-1)  # B x ml x 1
        uiu_attn = F.softmax(uiu_attn, dim=1)
        uiu_feat = (uiu_attn * uiu_feat).sum(dim=1)  # B x d
        uiu_feat = F.relu(uiu_feat)

        # get uui mask
        # uui = B x ml(metapath_maxlen) x 2 --> [(uid), mp_user_id, mp_item_id]
        uui_mask = torch.BoolTensor(uui[:, 0] == 0).to(self.dev)  # B x ml

        # encoder uui
        uui = uui.to(self.dev)  # B x ml x 2
        mp_uid, mp_iid = uui[:, 0], uui[:, 1]  # B x ml x 1
        mp_user_emb = self.dropout(user_embs(mp_uid))  # B x ml x d
        mp_item_emb = self.dropout(item_embs(mp_iid))  # B x ml x d
        uui_feat = (self_emb + mp_user_emb + mp_item_emb) / 3.0  # B x ml x d

        # calc uui attn & aggr
        uui_h_attn = torch.cat([self_emb.expand_as(uui_feat), uui_feat], dim=-1)  # B x ml x 2d
        uui_attn = self.intra_attn_uui(uui_h_attn)  # B x ml x 1
        uui_attn = F.leaky_relu(uui_attn)
        uui_attn += -1e8 * uui_mask.unsqueeze(dim=-1)  # B x ml x 1
        uui_attn = F.softmax(uui_attn, dim=1)
        uui_feat = (uui_attn * uui_feat).sum(dim=1)  # B x d
        uui_feat = F.relu(uui_feat)

        # calc inter mp attn & aggr
        uu_mp_h = F.tanh(self.dropout(self.inter_attn_0(uu_feat)))  # B x (d//2)
        uu_mp_h = uu_mp_h.mean(dim=0)  # 1 x (d//2)
        uu_mp_attn_score = self.inter_attn_1(uu_mp_h)  # 1 x 1

        uiu_mp_h = F.tanh(self.dropout(self.inter_attn_0(uiu_feat)))
        uiu_mp_h = uiu_mp_h.mean(dim=0)
        uiu_mp_attn_score = self.inter_attn_1(uiu_mp_h)

        uui_mp_h = F.tanh(self.dropout(self.inter_attn_0(uui_feat)))
        uui_mp_h = uui_mp_h.mean(dim=0)
        uui_mp_attn_score = self.inter_attn_1(uui_mp_h)

        mp_attn_score = torch.cat([uu_mp_attn_score, uiu_mp_attn_score, uui_mp_attn_score], dim=0)
        mp_attn = F.softmax(mp_attn_score, dim=0)  # 2 x 1

        user_feat = mp_attn[0] * uu_feat + mp_attn[1] * uiu_feat + mp_attn[2] * uui_feat
        user_feat = F.relu(user_feat)

        return user_feat


class MAGNN(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(MAGNN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5/user_num, b=0.5/user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5/item_num, b=0.5/item_num)

        self.user_encoder = UserEncoder(args.edim, args)
        self.dropout = nn.Dropout(args.droprate)

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def forward(self, batch):
        uid, pos, neg, nbr, uiu, uui = batch
        hu = self.user_encoder(self.user_embs, self.item_embs, uid, nbr, uiu, uui)
        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))
        hu = hu.unsqueeze(1).expand_as(pos_hi)
        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)
        return pos_logits, neg_logits

    def predict(self, eval_batch):
        self.eval()
        with torch.no_grad():
            uid, nbr, nbr_iid, uiu, uui, eval_iid = eval_batch
            hi = self.item_embs(eval_iid.to(self.dev))  # 1 x item_len x d
            hu = self.user_encoder(self.user_embs, self.item_embs, uid, nbr, uiu, uui)
            hu = hu.unsqueeze(1).expand_as(hi)
            logits = self.pred(hu, hi)
            return logits.view(-1)

# ---------------------------------------------------------------------------------------------------------------------
# Train and Evaluate

def parse_sampled_batch(batch):
    uid, pos, neg, nbr = batch
    uid = torch.from_numpy(uid).long()
    pos = torch.from_numpy(pos).long()
    neg = torch.from_numpy(neg).long()
    nbr = torch.from_numpy(nbr).long()

    # metapaths
    uiu = get_uiu(user_train, user, u2u, mtp_maxlen)
    uui = get_uui(user_train, user, u2u, mtp_maxlen)

    batch = [uid, pos, neg, nbr, uiu, uui]
    indices = torch.where(pos != 0)
    return batch, indices


def train(model, opt, shdlr, train_batches, cur_idx, num_batch, args):
    model.train()
    total_loss = 0.0
    for batch in train_batches[cur_idx:cur_idx+num_batch]:
        parsed_batch, indices = parse_sampled_batch(batch[0])
        opt.zero_grad()
        shdlr.step()
        pos_logits, neg_logits = model(parsed_batch)

        loss = F.binary_cross_entropy_with_logits(pos_logits[indices], torch.ones_like(pos_logits)[indices]) + \
               F.binary_cross_entropy_with_logits(neg_logits[indices], torch.zeros_like(neg_logits)[indices])

        loss.backward()
        opt.step()

        total_loss += loss.item()

    return total_loss / num_batch


def main():
    parser = argparse.ArgumentParser(description='MAGNN')
    parser.add_argument('--dataset', default='Epinions')
    parser.add_argument('--model', default='MAGNN')

    # Model Config
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--use_pos_emb', type=int, default=True)
    parser.add_argument('--seq_maxlen', type=int, default=50, help='fixed, or change with sampled train_batches')
    parser.add_argument('--nbr_maxlen', type=int, default=20, help='fixed, or change with sampled train_batches')
    parser.add_argument('--mtp_maxlen', type=int, default=100, help='number of sampled metapaths')

    # Train Config
    parser.add_argument('--batch_size', type=int, default=1024, help='fixed, or change with sampled train_batches')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--l2rg', type=float, default=5e-4)
    parser.add_argument('--emb_reg', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--check_epoch', type=int, default=40)
    parser.add_argument('--loss_type', default='bpr', help='bce/bpr')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--patience', type=int, default=5)

    # Something else
    parser.add_argument('--repeat', type=int, default=1)
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
    num_batch = len(user_train) // args.batch_size

    for r in range(args.repeat):
        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = MAGNN(user_num, item_num, u2u, args)
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


        print('Testing', end='')
        metrics_list = []
        model.load_state_dict(torch.load(model_path))
        for _ in range(args.test_time):
            test_metrics = evaluate(model, dataset, args, is_test=True)
            hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = test_metrics
            logger.info('Iter={} Tst HR(5/10/20)={:.4f}/{:.4f}/{:.4f} NDCG(5/10/20)={:.4f}/{:.4f}/{:.4f}'
                        .format(r, hr5, hr10, hr20, ndcg5, ndcg10, ndcg20))
            metrics_list.append(test_metrics)

        metrics = np.array(metrics_list)
        print('Iter={} Test Summary:'.format(r))
        print('Mean hr5, hr10, hr20, ndcg5, ndcg10, ndcg20')
        print(metrics.mean(axis=1))
        print('Std  hr5, hr10, hr20, ndcg5, ndcg10, ndcg20')
        print(metrics.std(axis=1))

    print("Done")


if __name__ == '__main__':
    main()
