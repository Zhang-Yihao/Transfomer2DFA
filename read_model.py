import torch
import torch.nn.functional as F
import copy
from train.dset import target_func


def get_params(model):
    param_dict = {'embedding': copy.deepcopy(model['embedding.weight']),
                  'q_matrix': copy.deepcopy(model['q_matrix.weight']),
                  'k_matrix': copy.deepcopy(model['k_matrix.weight']),
                  'v_matrix': copy.deepcopy(model['v_matrix.weight']),
                  'mlp': copy.deepcopy(model['mlp.0.weight']),
                  'mlp_bias': copy.deepcopy(model['mlp.0.bias']),
                  'mlp2': copy.deepcopy(model['mlp.2.weight']),
                  'mlp2_bias': copy.deepcopy(model['mlp.2.bias'])}
    return param_dict


def get_qkv(model):
    param_dict = get_params(model)
    q = param_dict['q_matrix']
    k = param_dict['k_matrix']
    v = param_dict['v_matrix']
    embedding = param_dict['embedding']
    return {'q': torch.matmul(embedding, q.transpose(0, 1)),
            'k': torch.matmul(embedding, k.transpose(0, 1)),
            'v': torch.matmul(embedding, v.transpose(0, 1))}


def last_nonzero_idx(x):
    for k in range(len(x) - 1, -1, -1):
        if x[k] != 0:
            return k
    return 0


def get_model_out(data, model):
    model_params = get_params(model)
    last_nonzero = last_nonzero_idx(data)
    k_data = [model_params['k_matrix'][data[k]] for k in range(len(data))]
    q_data = model_params['q_matrix'][data[last_nonzero]]
    v_data = [model_params['v_matrix'][data[k]] for k in range(len(data))]
    attn_weights = torch.matmul(q_data, torch.stack(k_data).transpose(0, 1))
    v = torch.matmul(attn_weights, torch.stack(v_data))
    out_0 = torch.matmul(v, model_params['mlp'].transpose(0, 1)) + model_params['mlp_bias']
    out_relu = F.relu(out_0)
    out = torch.matmul(out_relu, model_params['mlp2'].transpose(0, 1)) + model_params['mlp2_bias']
    softmax_out = F.softmax(out, dim=-1)
    return softmax_out


def test_model(model, dataset, target_func_name):
    len_dataset = len(dataset)
    num_correct = 0
    for k in range(len_dataset):
        data = dataset[k]
        model_out = get_model_out(data, model)
        if model_out.argmax() == target_func(data, target_func_name):
            num_correct += 1
    return num_correct / len_dataset
