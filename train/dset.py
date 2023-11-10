import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from train.config import read_config

# read config
config = read_config('config.json')


def target_func(x, target_func_name=config.func_name):
    def find_last_nonzero(lst):
        for k in range(len(lst) - 1, -1, -1):
            if lst[k] != 0:
                return k
        return 0

    if target_func_name == "simple":
        return (find_last_nonzero(x) + 4) % 3 + 1
    elif target_func_name == "long_term_dependency":
        if int(x[0]) < 3:
            return (find_last_nonzero(x) + 3) % 4 + 1
        elif int(x[0]) < 6:
            return (find_last_nonzero(x) + 4) % 3 + 1
        else:
            return (find_last_nonzero(x) + 6) % 2 + 1
    elif target_func_name == "regex0":
        # (1, 1, 2, 2)*
        len_x = find_last_nonzero(x) + 1
        num_lst = [1, 1, 2, 2]
        return num_lst[len_x % 4]
    elif target_func_name == "regex1":
        # (1, 3, 2, 4)*
        len_x = find_last_nonzero(x) + 1
        num_lst = [1, 3, 2, 4]
        return num_lst[len_x % 4]
    elif target_func_name == "regex2":
        # (1, 1, 3)*
        len_x = find_last_nonzero(x) + 1
        num_lst = [1, 1, 3]
        return num_lst[len_x % 3]


# generate a simple sequence of digits that only depends on the previous digit
def generate_sequence(max_seq_length=20, target_func_name="simple"):
    start_num = 1 if "regex" in target_func_name else np.random.randint(1, config.vocab_size-1)
    sequence = [start_num]
    seq_length = np.random.randint(1, max_seq_length-2)
    for _ in range(seq_length - 1):
        sequence.append(target_func(sequence, target_func_name))
    # padding
    sequence = sequence + [0] * (max_seq_length - len(sequence))
    return sequence


# define a PyTorch Dataset
class NumberSequenceDataset(Dataset):
    def __init__(self, num_samples, seq_length, func_name="simple"):
        self.samples = []
        self.func_name = func_name
        for _ in range(num_samples):
            sequence = generate_sequence(seq_length, func_name)
            self.samples.append(sequence)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.LongTensor(self.samples[idx])


# divide the dataset into train and test sets
def get_dataloader(num_samples, seq_length, test_split, func_name="simple", batch_size=32):
    full_dataset = NumberSequenceDataset(num_samples, seq_length, func_name)

    # dividing process
    train_size = int((1 - test_split) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # create dataloaders instances
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
