import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


# generate a simple sequence of digits that only depends on the previous digit
def generate_sequence(max_seq_length=20):
    start_num = str(np.random.randint(1, 10))
    sequence = [start_num]
    seq_length = np.random.randint(2, max_seq_length - 1)
    for _ in range(seq_length - 1):
        # a_n = (a_{n-1} + 4) % 9 + 1
        next_num = (int(sequence[-1]) + 4) % 9 + 1
        sequence.append(str(next_num))
    # the last digit is the digit to predict
    target = str((int(sequence[-1]) + 4) % 9 + 1)
    # padding
    sequence = sequence + ['0'] * (max_seq_length - len(sequence))
    return sequence, target


# generate_sequence_glob generates a sequence of digits, but the relationship is more complex
def generate_sequence_glob(max_seq_length=20):
    start_num = str(np.random.randint(1, 10))
    sequence = [start_num]
    seq_length = np.random.randint(5, max_seq_length - 1)
    for _ in range(seq_length - 1):
        # a_n depends on a_{n-1} and a_{0}
        if int(sequence[0]) < 3:
            next_num = (int(sequence[-1]) + 3) % 9 + 1
        elif int(sequence[0]) < 6:
            next_num = (int(sequence[-1]) + 4) % 10 + 1
        else:
            next_num = (int(sequence[-1]) + 7) % 7 + 1
        sequence.append(str(next_num))
    # the last digit is the digit to predict
    target = ""
    if int(sequence[0]) < 3:
        target = str((int(sequence[-1]) + 3) % 9 + 1)
    elif int(sequence[0]) < 6:
        target = str((int(sequence[-1]) + 4) % 10 + 1)
    else:
        target = str((int(sequence[-1]) + 7) % 7 + 1)
    # padding
    sequence = sequence + ['0'] * (max_seq_length - len(sequence))
    return sequence, target


def generate_sequence_no_compute(max_seq_length=20):
    rand_dict = {'0': '0', '1': '7', '2': '6', '3': '2', '4': '1', '5': '8', '6': '3', '7': '4', '8': '9', '9': '5',
                 '10': '10'}
    start_num = str(np.random.randint(1, 10))
    sequence = [start_num]
    seq_length = np.random.randint(5, max_seq_length - 1)
    next_num = ""
    for _ in range(seq_length - 1):
        next_num = rand_dict[sequence[-1]]
        sequence.append(next_num)
    # the last digit is the digit to predict
    target = rand_dict[sequence[-1]]
    # padding
    sequence = sequence + ['0'] * (max_seq_length - len(sequence))
    return sequence, target


# define a PyTorch Dataset
class NumberSequenceDataset(Dataset):
    def __init__(self, num_samples, seq_length, func=generate_sequence):
        self.samples = []
        self.targets = []
        self.func = func
        for _ in range(num_samples):
            sequence, target = func(seq_length)
            self.samples.append(sequence)
            # convert target to probability distribution for cross-entropy loss
            target = int(target)
            target_dist = [0] * 11
            target_dist[target] = 1
            self.targets.append(target_dist)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # target is float tensor as prob, input is long tensor
        return torch.LongTensor([int(digit) for digit in self.samples[idx]]), \
            torch.FloatTensor(self.targets[idx])


# divide the dataset into train and test sets
def get_dataloader(num_samples, seq_length, test_split, func=generate_sequence, batch_size=32):
    full_dataset = NumberSequenceDataset(num_samples, seq_length, func)

    # dividing process
    train_size = int((1 - test_split) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # create dataloaders instances
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
