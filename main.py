import torch

from dset import get_dataloader
from model import SimpleAttentionModel
from config.config import read_config

# read config
config = read_config('config/config.json')

# get dataloader
train_loader, test_loader = get_dataloader(
    config.num_samples, config.seq_length, config.test_split
)

# define model
model = SimpleAttentionModel(
    config.vocab_size, config.embed_dim, config.mlp_hidden_dim
)

# define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# train
model.train()
for epoch in range(config.num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x)
        # batch loss
        loss = loss_fn(y_pred, y.max(dim=1)[1])
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

# test
model.eval()
num_correct = 0
num_samples = 0
for x, y in test_loader:
    y_pred = model(x)
    # get the index of the max log-probability
    _, y_pred = y_pred.max(dim=1)
    y_correct = y.max(dim=1)[1]
    num_correct += (y_pred == y_correct).sum()
    num_samples += y.size(0)
print(f'Accuracy: {num_correct / num_samples * 100:.2f}%')

# give some examples
model.eval()
for x, y in test_loader:
    y_pred = model(x)
    _, y_pred = y_pred.max(dim=1)
    y_correct = y.max(dim=1)[1]
    for i in range(5):
        print(f'Input: {x[i].tolist()}')
        print(f'Ground truth: {y_correct[i].item() + 1}')
        print(f'Prediction: {y_pred[i].item() + 1}')
        print('---')

# interactive mode
dig_str = ""
while (dig_str != "q"):
    dig_str = input("Enter a digit sequence (or q to quit): ")
    if dig_str == "q":
        break
    dig_list = [int(dig) for dig in dig_str.split()]
    dig_list = dig_list + [0] * (config.seq_length - len(dig_list))
    x = torch.LongTensor(dig_list).unsqueeze(0)
    y_pred = model(x)
    _, y_pred = y_pred.max(dim=1)
    print(f'Predicted next digit: {y_pred.item() + 1}')

# save model
torch.save(model.state_dict(), 'checkpoint/model.pt')