from dataclasses import dataclass
import json

# 定义 Config 数据类
@dataclass
class Config:
    num_samples: int
    seq_length: int
    test_split: float
    vocab_size: int
    embed_dim: int
    mlp_hidden_dim: int
    batch_size: int
    num_epochs: int
    learning_rate: float

# 从 JSON 文件中读取配置
def read_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Config(**config_dict)