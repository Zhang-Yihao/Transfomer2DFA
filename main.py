from automata import predict_by_automata,get_transition_matrix, km_model
from train.dset import NumberSequenceDataset
from read_model import get_model_out
import torch
from tqdm import tqdm
# TODO: add argparse, no need to hardcode
model = torch.load('model/model_self_regex0_42.0_92992.pt')
dataset = NumberSequenceDataset(1000, 10, "regex0")

kmeans_mdl = km_model(model, dataset)
trans_mat = get_transition_matrix(model, dataset, kmeans_mdl, 5)

num_corr = 0
# use tqdm to show progress bar
for i in tqdm(range(len(dataset))):
    data = dataset[i]
    if get_model_out(data, model).argmax() == predict_by_automata(model, dataset, i, kmeans_mdl, trans_mat):
        num_corr += 1

print("Accuracy: ", num_corr / len(dataset))
