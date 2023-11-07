import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


# 生成符合语言规则的数据
def generate_sequence(start_num, max_seq_length=50):
    sequence = [start_num]
    seq_length = 10 #np.random.randint(2, max_seq_length)
    for _ in range(seq_length - 1):
        next_num = (sequence[-1] + 7) % 9
        sequence.append(next_num)
    return sequence


# 定义数据集类
class NumberSequenceDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.samples = []
        for _ in range(num_samples):
            start_num = np.random.randint(0, 9)  # 随机选择起始数字
            sequence = generate_sequence(start_num, seq_length)
            self.samples.append(sequence)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 转换为PyTorch张量
        return torch.tensor(sample[:-1], dtype=torch.long), torch.tensor(sample[-1], dtype=torch.long)


# 分割数据集
def get_dataloader(num_samples, seq_length, test_split):
    full_dataset = NumberSequenceDataset(num_samples, seq_length)

    # 分割数据集为训练集和测试集
    train_size = int((1 - test_split) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 创建DataLoader实例
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader
