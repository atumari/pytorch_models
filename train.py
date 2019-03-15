import numpy as np
from sklearn.utils import shuffle
import numba
from tqdm import tqdm
import torch
from torch.nn import functional as F


def train(x, y, model, optimizer, clip, batch_size, device, train_mode=True):
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    pred = model(x)
    
    loss = F.binary_cross_entropy_with_logits(pred, y, reduction="sum")

    if train_mode:
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    
    return loss.item(), pred


def trainIters(model, data, target, val_data, val_target, optimizer, device, num_epochs,
                                batch_size, clip, fold, oof_pred, val_mask, rep, early_stopping_num, save_file):

    data_len = len(data)
    val_len = len(val_data)
    best_auc = 0
    count = 0
    for i in range(num_epochs):
        train_loss = 0
        val_loss = 0
        train_preds = np.zeros_like(target, dtype=np.float32)
        val_preds = np.zeros_like(val_target, dtype=np.float32)

        data, target = shuffle(data, target)
        data_ = torch.from_numpy(data)
        target_ = torch.from_numpy(target)
        val_data_ = torch.from_numpy(val_data)
        val_target_ = torch.from_numpy(val_target)
        model.train() 
        for j in tqdm(range((data_len -1) // batch_size + 1)):
            x = data_[j*batch_size: min((j+1)*batch_size, data_len)]
            y = target_[j*batch_size: min((j+1)*batch_size, data_len)]

            loss, preds = train(x, y, model, optimizer, clip, batch_size, device, train_mode=True)
            train_loss += loss
            train_preds[j*batch_size: min((j+1)*batch_size, data_len)] = preds.detach().cpu().numpy()
            
        model.eval()
        for j in tqdm(range((val_len - 1) // batch_size + 1)):
            x = val_data_[j*batch_size: min((j+1)*batch_size, val_len)]
            y = val_target_[j*batch_size: min((j+1)*batch_size, val_len)]
            loss, preds = train(x, y, model, optimizer, clip, batch_size, device, train_mode=False)
            val_loss += loss
            val_preds[j*batch_size: min((j+1)*batch_size, val_len)] = preds.detach().cpu().numpy()
         

        train_loss /= data_len
        val_loss /= val_len

        train_auc = fast_auc(target.flatten(), train_preds.flatten())
        val_auc = fast_auc(val_target.flatten(), val_preds.flatten())

        print("{}epoch train_loss: {}, val_loss: {}\n".format(i, train_loss, val_loss))
        print("train auc: {}".format(train_auc))
        print("val   auc: {}".format(val_auc))

        if val_auc > best_auc:
            count = 0
            best_auc = val_auc
            torch.save({
                "model": model.state_dict(),
                # "optim": optimizer.state_dict()
                }, save_file.format(fold, rep))
            oof_pred[val_mask] += val_preds.flatten()
            print("Save Model at {}".format(save_file.format(fold, rep)))
        else:
            count += 1
            if count > early_stopping_num:
                print("BREAK at {}epochs".format(i))
                break
        print("Best Score: {}\n".format(best_auc))


@numba.jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc



def train_dae(x, y, model, optimizer, clip, batch_size, device, train_mode=True):

    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    emb, decs = model(x)
    
    loss = 0
    for i in range(len(decs)):
        loss += F.nll_loss(decs[i], y[:, i], reduction='sum')


    if train_mode:
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    
    return loss.item(), emb


def trainIters_dae(model, data, optimizer, device, num_epochs,
                                        batch_size, clip, noise_perc, save_file):

    data_len_ = len(data)
    block_size = data_len_ // 5
    best_loss = 1000
    data_2 = np.zeros((block_size, data.shape[1]), dtype=np.int64)
    for i in range(num_epochs):
        train_loss = 0

        if i % 5 == 0:
            data = shuffle(data)

        data_block = data[(i % 5) * block_size : (i % 5 + 1) * block_size ]
        data_len = len(data_block)
        data_2[:] = data_block

        noise_num = int(data_len * noise_perc * len(data_block[0]) // 2)
        c = np.random.randint(len(data_block[0]), size=noise_num)
        r = np.random.randint(data_len, size=2*noise_num)
        for j in range(noise_num):
            data_2[r[2*j], c[j]], data_2[r[2*j+1], c[j]] = data_2[r[2*j+1], c[j]], data_2[r[2*j], c[j]] 

        data_ = torch.from_numpy(data_2)
        target_ = torch.from_numpy(data_block)
        model.train() 
        for j in tqdm(range((data_len -1) // batch_size + 1)):
            x = data_[j*batch_size: min((j+1)*batch_size, data_len)]
            y = target_[j*batch_size: min((j+1)*batch_size, data_len)]

            loss, emb = train_dae(x, y, model, optimizer, clip, batch_size, device, train_mode=True)
            train_loss += loss
       
        train_loss /= data_len

        print("{}epoch train_loss: {}\n".format(i, train_loss))

        if best_loss > train_loss :
            best_loss = train_loss
            torch.save({
                "model": model.state_dict(),
                # "optim": optimizer.state_dict()
                }, save_file)

            print("Save Model at {}".format(save_file))
        print("Best Score: {}\n".format(best_loss))