import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


# generate a random sequence of digits
def generate_sequence(max_seq_length=50):
    start_num = str(np.random.randint(1, 10))
    sequence = [start_num]
    seq_length = np.random.randint(2, max_seq_length - 1)
    for _ in range(seq_length - 1):
        # a_n = (a_{n-1} + 3) % 9 + 1
        next_num = (int(sequence[-1]) + 4) % 9 + 1
        sequence.append(str(next_num))
    # the last digit is the digit to predict
    target = str((int(sequence[-1]) + 4) % 9 + 1)
    # padding
    sequence = sequence + ['0'] * (max_seq_length - len(sequence))
    return sequence, target


# define a PyTorch Dataset
class NumberSequenceDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.samples = []
        self.targets = []
        for _ in range(num_samples):
            sequence, target = generate_sequence(seq_length)
            self.samples.append(sequence)
            # convert target to probability distribution for cross-entropy loss
            target = int(target)
            target_dist = [0] * 10
            target_dist[target - 1] = 1
            self.targets.append(target_dist)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.LongTensor([int(digit) for digit in self.samples[idx]]), \
            torch.LongTensor(self.targets[idx])


# divide the dataset into train and test sets
def get_dataloader(num_samples, seq_length, test_split):
    full_dataset = NumberSequenceDataset(num_samples, seq_length)

    # dividing process
    train_size = int((1 - test_split) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # create dataloaders instances
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader
