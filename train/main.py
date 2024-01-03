import torch
import numpy as np
from train.dset import get_dataloader, target_func
from train.model import SimpleAttentionModel, SelfAttentionModel, EnhancedAttentionModel
from train.config import read_config
from train.tomita import *


def pred_next_n_digits(xin, n, mdl):
    # Parallel is here poorly implemented
    # xinput: [batch_size, seq_len]
    # n: number of digits to predict
    # return: y_pred_out: [batch_size, n, vocab_size]
    y_pred_out = []
    for xinput in xin:
        x_input_copy = [xinput]
        y_pred_n = []
        for k in range(n):
            next_prd = mdl(torch.LongTensor(x_input_copy[k]).unsqueeze(0))
            y_pred_n.append(next_prd[0])
            next_prd_idx = next_prd.argmax(dim=1)
            # get last non-zero digit, and modify its next digit to the predicted one
            last_nonzero_idx = (x_input_copy[k] != 0).nonzero()[-1]
            if k != n - 1:
                # make a copy that has no relation to the original one
                x_input_copy.append(x_input_copy[k].clone())
                x_input_copy[k + 1][last_nonzero_idx + 1] = next_prd_idx
        y_pred_n = torch.stack(y_pred_n, dim=1)
        y_pred_out.append(y_pred_n)
    y_pred_out = torch.stack(y_pred_out, dim=0)
    return y_pred_out


def get_correct_n_digits(xin, n, _config):
    # xinput: [batch_size, seq_len]
    # n: number of digits to predict
    # return: y_correct_out: [batch_size, n]
    y_correct_out = []
    for xinput in xin:
        xinput = xinput.numpy()
        y_correct_n = []
        for _ in range(n):
            next_correct = target_func(xinput, _config.func_name)
            y_correct_n.append(next_correct)
            # get last non-zero digit, and modify its next digit to the predicted one
            last_nonzero_idx = (xinput != 0).nonzero()[-1]
            xinput[last_nonzero_idx + 1] = next_correct
        y_correct_n = torch.LongTensor(y_correct_n)
        y_correct_out.append(y_correct_n)
    y_correct_out = torch.stack(y_correct_out, dim=0)
    return y_correct_out


if __name__ == "__main__":
    # read config
    try:
        config = read_config('config.json')
    except FileNotFoundError:
        config = read_config('train/config.json')

    train_loader, test_loader = get_dataloader(
        config.num_samples, config.seq_length, config.test_split, config.func_name, config.batch_size
    )

    # define model
    model = {
        "simple": SimpleAttentionModel,
        "self": SelfAttentionModel,
        "enhanced": EnhancedAttentionModel
    }[config.model_name](config.vocab_size, config.embed_dim, config.mlp_hidden_dim, config.output_size)

    # define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # use GPU if available
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # train
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(config.num_epochs):
        for batch_idx, x in enumerate(train_loader):
            data = x[0].to(device)
            label = x[1].flatten().to(device)
            optimizer.zero_grad()
            if "tomita" in config.func_name:
                # x = (data, label), stored in a tensor of length batch_size
                y_pred = model(data)
                tomita_func = eval(config.func_name)
                loss = loss_fn(y_pred, label)
            elif config.pred_num == 1:
                y_pred = model(x)
                loss = loss_fn(y_pred, torch.LongTensor([target_func(seq, config.func_name) for seq in x]))
            else:
                y_pred = pred_next_n_digits(x, config.pred_num, model)
                y_correct = get_correct_n_digits(x, config.pred_num)
                loss = loss_fn(y_pred, y_correct)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 and batch_idx == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

    # test
    model.eval()
    num_correct = 0
    num_samples = 0
    for x in test_loader:
        y_pred = model(x[0])
        # get the index of the max log-probability
        _, y_pred = y_pred.max(dim=1)
        if "tomita" in config.func_name:
            data = x[0].to(device)
            label = x[1].flatten().to(device)
            y_pred = model(data)
            _, y_pred = y_pred.max(dim=1)
            y = label
        else:
            y = torch.LongTensor([target_func(seq, config.func_name) for seq in x])
        num_correct += (y_pred == y).sum()
        num_samples += y.size(0)
    acc = float(num_correct) / num_samples * 100
    print(f'Accuracy: {acc:.4f}%')

    # give some examples
    for x in test_loader:
        y_pred = model(x[0])
        _, y_pred = y_pred.max(dim=1)
        print(f'Input: {x[0][0].tolist()}')
        if "tomita" in config.func_name:
            print(f'Ground truth: {x[1][0].item()}')
            print(f'Prediction: {y_pred[0].item()}')
        else:
            print(f'Ground truth: {target_func(x[0].tolist(), config.func_name)}')
            print(f'Prediction: {y_pred[0].item()}')
        print('---')
"""
    # interactive mode
    interact = config.interact
    if interact:
        dig_str = ""
        while dig_str != "q":
            dig_str = input("Enter a digit sequence (or q to quit): ")
            if dig_str == "q":
                break
            dig_list = [int(dig) for dig in dig_str.split()]
            dig_list = dig_list + [0] * (config.seq_length - len(dig_list))
            x = torch.LongTensor(dig_list).unsqueeze(0)
            y_pred = model(x)
            _, y_pred = y_pred.max(dim=1)
            print(f'Predicted next digit: {y_pred.item()}')

    # save model
    model_rand_id = np.random.randint(100000)
    model_id = f'{config.model_name}_{config.func_name}_{acc}_{model_rand_id}'
    torch.save(model.state_dict(), '../model/model_{}.pt'.format(model_id))
    print(f'Saved model to ../model/model_{model_id}.pt')
"""