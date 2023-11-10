from dataclasses import dataclass
import json


# define config class with dataclass decorator
@dataclass
class Config:
    num_samples: int
    seq_length: int
    test_split: float
    padding_idx: int
    vocab_size: int
    embed_dim: int
    mlp_hidden_dim: int
    interact: bool
    batch_size: int
    num_epochs: int
    learning_rate: float
    model_name: str
    func_name: str
    pred_num: int


# read config from json file
def read_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Config(**config_dict)
