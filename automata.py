from read_model import get_params, get_qkv, last_nonzero_idx, get_model_out, test_model

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from train.model import PositionalEncoding, DigitEmbedding
import numpy as np
from train.config import read_config

# read config
try:
    config = read_config('config.json')
except FileNotFoundError:
    config = read_config('train/config.json')


def get_keys(model, x=None, embedding=DigitEmbedding()):
    if x:
        embed_x = model.embedding(x)
        K = model['k_matrix']
        return torch.matmul(embed_x, K.transpose(0, 1)).detach().numpy()
    return get_qkv(model)['k'].detach().numpy()


def encode_lst_as_states(model, data, compress_strategy="keep_all"):
    # data: [seq_len]
    # compress_strategy: "keep_all" or "keep_last_n", n is an integer
    # return: [seq_len * embed_dim] as a "state" for clustering
    last_nonzero = last_nonzero_idx(data)
    key_lst = np.stack([get_keys(model)[k] for k in data])
    if compress_strategy == "keep_all":
        return key_lst.reshape(-1)
    elif "keep_last" in compress_strategy:
        num_last = int(compress_strategy.split('_')[-1])
        if num_last == 1:
            return key_lst[last_nonzero]
        else:
            key_lst_out = np.empty(0)
            # if last_nonzero + 1 <= num_last just keep all
            # else 1 global + num_last - 1 last keys
            start_idx = 0 if last_nonzero - num_last + 1 < 0 else last_nonzero - num_last + 2
            end_idx = num_last if last_nonzero + 1 < num_last else last_nonzero + 1
            if end_idx - start_idx == num_last - 1:
                # this means we need to add a global key
                key_lst_out = np.concatenate((key_lst_out, sum(key_lst[:start_idx]) / start_idx))
            for k in range(start_idx, end_idx):
                key_lst_out = np.concatenate((key_lst_out, key_lst[k]))
            return key_lst_out


def km_model(model, dataset):
    # dataset: [num_samples, seq_len]
    # return: a kmeans model
    key_lst = []
    for k in range(len(dataset)):
        data = dataset[k]
        key_lst.append(encode_lst_as_states(model, data))
    key_lst = np.stack(key_lst)
    kmeans = KMeans(n_clusters=7, n_init=10, random_state=0).fit(key_lst)
    return kmeans


def get_transition_matrix(model, dataset, kmeans_mdl, vocab_size, randomize=False):
    centers = kmeans_mdl.cluster_centers_
    n_clusters = len(centers)
    trans_matrix = None
    if randomize > 0:
        # randomize the transition matrix by assigning random small values to each entry
        trans_matrix = np.random.rand(vocab_size, n_clusters, n_clusters) * randomize
    else:
        trans_matrix = np.zeros((vocab_size, n_clusters, n_clusters))
    # 0 is the padding token, so should have no transition
    trans_matrix[0] = np.identity(n_clusters)
    len_dataset = len(dataset)
    for data_idx in range(len_dataset):
        data = dataset[data_idx]
        last_nonzero = last_nonzero_idx(data)
        data_key = encode_lst_as_states(model, data)
        if last_nonzero == 0:
            # if the data is all padding, then we just skip it
            continue
        next_tok = get_model_out(data, model).argmax()
        # predict method use 2D array, but only 1D is needed, thus the [0]
        state_start = kmeans_mdl.predict(np.array([data_key]))[0]
        data_next = data.clone()
        data_next[last_nonzero + 1] = next_tok
        data_next_key = encode_lst_as_states(model, data_next)
        state_next = kmeans_mdl.predict(np.array([data_next_key]))[0]
        trans_matrix[next_tok, state_start, state_next] += 1
    # normalize
    for word in trans_matrix:
        for row in word:
            if row.sum() > 0:
                row /= row.sum()
    return trans_matrix


def get_state_value_set(model, dataset, kmeans_mdl):
    # dataset: [num_samples, seq_len]
    # return: an average value for each state
    v = get_qkv(model)['v']
    state_num = len(kmeans_mdl.cluster_centers_)
    v_state = [torch.zeros_like(torch.stack([v[k] for k in dataset[0]]))] * state_num
    v_num = [0] * state_num

    for i in range(len(dataset)):
        data = dataset[i]
        data_key = encode_lst_as_states(model, data)
        v_data = torch.stack([v[k] for k in data])
        state = kmeans_mdl.predict(np.array([data_key]))[0]
        v_state[state] += v_data
        v_num[state] += 1

    for i in range(state_num):
        if v_num[i] > 0:
            v_state[i] /= v_num[i]
    return v_state


def calc_state_output(model, dataset, state_id, lst_tok, kmeans_model):
    seq_len = len(dataset[0])
    params = get_params(model)
    values = get_state_value_set(model, dataset, kmeans_model)[state_id]
    q = params['q_matrix'][lst_tok]
    # keys should be converted to [seq_len, embed_dim]
    state = kmeans_model.cluster_centers_[state_id]
    # state = [n * embed_dim], convert to [n, embed_dim]
    keys = np.reshape(state, (-1, q.shape[0]))
    # pad with zeros
    if len(keys) < seq_len:
        keys = np.concatenate((keys, np.zeros((seq_len - len(keys), q.shape[1]))))
    keys = torch.FloatTensor(keys)
    attn_weights = torch.matmul(q, keys.transpose(0, 1))
    v = torch.matmul(attn_weights, torch.FloatTensor(values))
    out_0 = torch.matmul(v, params['mlp'].transpose(0, 1)) + params['mlp_bias']
    out_relu = F.relu(out_0)
    out = torch.matmul(out_relu, params['mlp2'].transpose(0, 1)) + params['mlp2_bias']
    softmax_out = F.softmax(out, dim=-1)
    return softmax_out.argmax()


def predict_by_automata(model, dataset, idx, kmeans_model, trans_matrix, mode="one_trans"):
    data = dataset[idx]
    if mode == "all_trans":
        predict_len = last_nonzero_idx(data)
        data = [data[0]] + [0] * (len(data) - 1)
        lst_nonzero = last_nonzero_idx(data)
        state_distribution = np.zeros(len(kmeans_model.cluster_centers_))
        state_distribution[kmeans_model.predict(np.array([encode_lst_as_states(model, data)]))[0]] = 1
        while lst_nonzero < predict_len:
            # predict the next token
            trans_matrix_now = trans_matrix[data[lst_nonzero]]
            state_distribution = np.matmul(trans_matrix_now, state_distribution)
            data[lst_nonzero + 1] = calc_state_output(model, dataset, state_distribution.argmax(), data[lst_nonzero],
                                                      kmeans_model)
            lst_nonzero = last_nonzero_idx(data)
        return calc_state_output(model, dataset, state_distribution.argmax(), data[lst_nonzero], kmeans_model)

    elif mode == "one_trans":
        data_key = encode_lst_as_states(model, data)
        state_start = kmeans_model.predict(np.array([data_key]))[0]
        state_distribution = np.zeros(len(kmeans_model.cluster_centers_))
        state_distribution[state_start] = 1
        last_nonzero = last_nonzero_idx(data)
        # predict the next token
        new_state_distribution = np.matmul(trans_matrix[data[last_nonzero]], state_distribution)
        # TODO: here we only use the argmax, but we can use average value weighted by the distribution.
        return calc_state_output(model, dataset, new_state_distribution.argmax(), data[last_nonzero], kmeans_model)
