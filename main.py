import torch

from dset import get_dataloader
from model import SimpleAttentionModel
from config.config import read_config

# 读取配置
config = read_config('config/config.json')

# 创建数据集
train_loader, test_loader = get_dataloader(
    config.num_samples, config.seq_length, config.test_split
)

# 创建模型
model = SimpleAttentionModel(
    config.vocab_size, config.embed_dim, config.mlp_hidden_dim
)

# 定义损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# 训练模型
model.train()
for epoch in range(config.num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

# 测试模型
model.eval()
num_correct = 0
num_samples = 0
for x, y in test_loader:
    y_pred = model(x)
    _, y_pred = y_pred.max(dim=1)
    num_correct += (y_pred == y).sum()
    num_samples += y.size(0)
print(f'Accuracy: {num_correct / num_samples * 100:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'checkpoint/model.pt')