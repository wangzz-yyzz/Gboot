import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch
from info_nce import info_nce

from prepare.pems_dataset import PEMSDataset


def convert_adj_to_coo(adj):
    """
    Convert adj.npy to coo format
    :param adj: [N, N]
    """
    edge_index_temp = sp.coo_matrix(adj)

    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index = torch.LongTensor(indices)

    return edge_index


def split_train_test(data, train_rate=0.6, val_rate=0.2, test_rate=0.2, x_len=12, y_len=12, overlap=False):
    """
    :param train_rate: int default 0.6
    :param val_rate: int default 0.2
    :param test_rate: int default 0.2
    :param data: [T, N]
    :param x_len: int default 12
    :param y_len: int default 12
    :return:
    """
    sample_len = x_len + y_len
    if not overlap:
        sample_num = data.shape[0] // sample_len
    else:
        sample_num = data.shape[0] - sample_len + 1
    train_num = int(sample_num * train_rate)
    val_num = int(sample_num * val_rate)
    test_num = sample_num - train_num - val_num
    week_sample_num = 24 * 24 / sample_len * 7

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []
    train_ext, val_ext, test_ext = [], [], []
    for i in range(sample_num):
        if overlap:
            sample = data[i:i + sample_len]
        else:
            sample = data[i * sample_len:(i + 1) * sample_len]
        if i < train_num:
            train_x.append(sample[:x_len])
            train_y.append(sample[x_len:])
            train_ext.append(i % week_sample_num)
        elif i < train_num + val_num:
            val_x.append(sample[:x_len])
            val_y.append(sample[x_len:])
            val_ext.append(i % week_sample_num)
        else:
            test_x.append(sample[:x_len])
            test_y.append(sample[x_len:])
            test_ext.append(i % week_sample_num)

    train_x = torch.Tensor(np.array(train_x))
    train_y = torch.Tensor(np.array(train_y))
    train_ext = torch.Tensor(np.array(train_ext))
    val_x = torch.Tensor(np.array(val_x))
    val_y = torch.Tensor(np.array(val_y))
    val_ext = torch.Tensor(np.array(val_ext))
    test_x = torch.Tensor(np.array(test_x))
    test_y = torch.Tensor(np.array(test_y))
    test_ext = torch.Tensor(np.array(test_ext))
    return train_x, train_y, val_x, val_y, test_x, test_y, train_ext, val_ext, test_ext


def byol_loss_fn(x, y):
    return 2 - 2 * (x * y).sum(dim=-1)


def info_nce_loss_fn(x, aug_x):
    return info_nce(x, aug_x)


def max_min_normalize(data):
    """
    :param data: [T, F, N]
    :return:
    """
    return (data - data.min()) / (data.max() - data.min()), data.min(), data.max()


def mean_std_normalize(data, mean=None, std=None):
    """
    :param std:
    :param mean:
    :param data: [T, F, N]
    :return:
    """
    if mean is None or std is None:
        print("cal mean and std")
        mean = data.mean()
        std = data.std()
    return (data - mean) / (std + 1e-7), mean, std


def handle_bad_state(data):
    data = np.where(data == 0, np.nan, data)
    nan_pos = np.argwhere(np.isnan(data))

    nan_state = np.unique(nan_pos[:, 1])
    nan_state_num = []
    for state in nan_state:
        nan_state_num.append(np.sum(nan_pos[:, 1] == state))
    nan_state_num = np.array(nan_state_num)
    nan_state = nan_state[np.argsort(nan_state_num)[::-1]]
    nan_state_num = nan_state_num[np.argsort(nan_state_num)[::-1]]
    nan_ratio = nan_state_num / data.shape[0]
    nan_ratio = np.around(nan_ratio * 100, decimals=2)
    nan_state = nan_state[nan_ratio > 90]
    print("bad state: ", nan_state)
    return nan_state


def handle_bad_flow(data):
    print("zero ratio: ", (data == 0).sum() / data.size)
    data = np.nan_to_num(data)
    data = np.where(data == 0, np.nan, data)
    print("nan ratio: ", np.isnan(data).sum() / data.size)
    data = pd.DataFrame(data)
    data = data.interpolate(axis=0)
    data = data.to_numpy()
    print("after interpolate, nan ratio: ", np.isnan(data).sum() / data.size)
    data = np.nan_to_num(data)
    return data


def normalize(data, norm_type='mean_std'):
    """
    :param data: [T, F, N]
    :param norm_type: str default 'mean_std'
    :return:
    """
    if norm_type == 'mean_std':
        return mean_std_normalize(data)
    elif norm_type == 'max_min':
        return max_min_normalize(data)
    else:
        raise ValueError("norm_type must be in ['mean_std', 'max_min']")


def add_self_loop_for_adj(adj):
    """
    :param adj: [N, N]
    :return:
    """
    adj = adj - adj.diagonal() * np.eye(adj.shape[0])
    adj = adj + np.eye(adj.shape[0])
    return adj


def get_dataloader(batch_size=128, train_rate=0.6, val_rate=0.2, test_rate=0.2, x_len=12, y_len=12,
                   mode='train', shuffle=True, task='PEMS03', if_norm=True, verbose=True, data_type='flow',
                   trend=True, drop_last=False, overlap=True, if_handle_nan=False,
                   use_extra_adj=False, extra_adj=None, add_self_loop=False):
    """
    :param add_self_loop:
    :param extra_adj:
    :param use_extra_adj:
    :param if_handle_nan:
    :param overlap:
    :param drop_last:
    :param trend:
    :param data_type:
    :param verbose:
    :param if_norm: if normalize the data
    :param batch_size: int batch size
    :param train_rate: int default 0.6
    :param val_rate: int default 0.2
    :param test_rate: int default 0.2
    :param x_len: int default 12
    :param y_len: int default 12
    :param mode: str default 'train'
    :param shuffle: bool default True
    :param task: str default 'PEMS03'(PEMS03, PEMS04, PEMS07, PEMS08)
    :return:
    """
    data_path = 'src/data'
    if use_extra_adj:
        adj = extra_adj
    else:
        adj = np.load(data_path + os.sep + task + os.sep + task + '_adj.npz')['adj']
    if add_self_loop:
        print("check self loop", adj.diagonal().sum())
        adj = add_self_loop_for_adj(adj)
        print("check self loop", adj.diagonal().sum())
    data_special = None
    data = np.load(data_path + os.sep + task + os.sep + task + '_flow.npz')['data']
    if data_type == 'flow':
        pass
    elif data_type == 'trend':
        data_special = np.load(data_path + os.sep + task + os.sep + task + '_trend.npz')['data']
    elif data_type == 'seasonal':
        data_special = np.load(data_path + os.sep + task + os.sep + task + '_seasonal.npz')['data']
    elif data_type == 'resid':
        data_special = np.load(data_path + os.sep + task + os.sep + task + '_resid.npz')['data']
    else:
        raise ValueError("data_type must be in ['flow', 'trend', 'seasonal', 'resid']")

    if if_handle_nan:
        data = handle_bad_flow(data)

    (train_x, train_y, val_x, val_y, test_x, test_y,
     train_ext, val_ext, test_ext) = split_train_test(data=data,
                                                      train_rate=train_rate,
                                                      val_rate=val_rate,
                                                      test_rate=test_rate,
                                                      x_len=x_len,
                                                      y_len=y_len,
                                                      overlap=overlap)
    if data_special is not None:
        print("data_type: ", data_type)
        (_, train_y, _, val_y, _, test_y,
         train_ext, val_ext, test_ext) = split_train_test(data=data_special,
                                                          train_rate=train_rate,
                                                          val_rate=val_rate,
                                                          test_rate=test_rate,
                                                          x_len=x_len,
                                                          y_len=y_len,
                                                          overlap=overlap)

    train_x_trend, train_y_trend = None, None
    val_x_trend, val_y_trend = None, None
    test_x_trend, test_y_trend = None, None
    if trend:
        print("add trend data")
        data_special = np.load(data_path + os.sep + task + os.sep + task + '_history_flow.npz')['data']
        (train_x_trend, train_y_trend, val_x_trend, val_y_trend, test_x_trend, test_y_trend,
         train_ext_trend, val_ext_trend, test_ext_trend) = split_train_test(data=data_special,
                                                                            train_rate=train_rate,
                                                                            val_rate=val_rate,
                                                                            test_rate=test_rate,
                                                                            x_len=x_len,
                                                                            y_len=y_len,
                                                                            overlap=overlap)

    edge_index = convert_adj_to_coo(adj=adj)

    if if_norm:
        train_x, train_x_mean, train_x_std = mean_std_normalize(train_x)
        val_x, val_x_mean, val_x_std = mean_std_normalize(val_x, mean=train_x_mean, std=train_x_std)
        test_x, test_x_mean, test_x_std = mean_std_normalize(test_x, mean=train_x_mean, std=train_x_std)

        train_norm_y, train_y_mean, train_y_std = mean_std_normalize(train_y, mean=train_x_mean, std=train_x_std)
        val_norm_y, val_y_mean, val_y_std = mean_std_normalize(val_y, mean=train_x_mean, std=train_x_std)
        test_norm_y, test_y_mean, test_y_std = mean_std_normalize(test_y, mean=train_x_mean, std=train_x_std)
        if trend:
            train_x_trend, train_x_mean_trend, train_x_std_trend = mean_std_normalize(train_x_trend)
            val_x_trend, val_x_mean_trend, val_x_std_trend = mean_std_normalize(val_x_trend, mean=train_x_mean_trend,
                                                                                std=train_x_std_trend)
            test_x_trend, test_x_mean_trend, test_x_std_trend = mean_std_normalize(test_x_trend,
                                                                                   mean=train_x_mean_trend,
                                                                                   std=train_x_std_trend)

    else:
        train_y_mean, train_y_std = 0, 0
        val_y_mean, val_y_std = 0, 0
        test_y_mean, test_y_std = 0, 0

        train_norm_y, _, _ = train_y
        val_norm_y, _, _ = val_y
        test_norm_y, _, _ = test_y

    if verbose:
        # [T, F, N]
        print('train_x shape: ', train_x.shape)
        # [T, F, N]
        print('train_y shape: ', train_y.shape)
        print('val_x shape: ', val_x.shape)
        print('val_y shape: ', val_y.shape)
        print('test_x shape: ', test_x.shape)
        print('test_y shape: ', test_y.shape)
        # [N, N]
        print('edge_index shape: ', edge_index.shape)
        # [N]
        print('train_ext shape: ', train_ext.shape)
        print('val_ext shape: ', val_ext.shape)
        print('test_ext shape: ', test_ext.shape)

    if mode == 'train':
        data = PEMSDataset(x=train_x, y=train_y, edge_index=edge_index, norm_y=train_norm_y,
                           mean=train_y_mean, std=train_y_std, ext=train_ext,
                           trend_x=train_x_trend, trend_y=train_y_trend)
    elif mode == 'val':
        data = PEMSDataset(x=val_x, y=val_y, edge_index=edge_index, norm_y=val_norm_y,
                           mean=val_y_mean, std=val_y_std, ext=val_ext,
                           trend_x=val_x_trend, trend_y=val_y_trend)
    elif mode == 'test':
        data = PEMSDataset(x=test_x, y=test_y, edge_index=edge_index, norm_y=test_norm_y,
                           mean=test_y_mean, std=test_y_std, ext=test_ext,
                           trend_x=test_x_trend, trend_y=test_y_trend)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last), adj


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan, mask=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        if mask is not None:
            mask = mask
        else:
            mask = mask_np(y_true, null_val)
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                                y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mse = (y_true - y_pred) ** 2
        return np.sqrt(np.mean(np.nan_to_num(mask * mse)))


def masked_mae_np(y_true, y_pred, null_val=np.nan, mask=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        if mask is not None:
            mask = mask
        else:
            mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        print("mask: ", mask)
        mae = np.abs(y_true - y_pred)
        return np.mean(np.nan_to_num(mask * mae))


def masked_mae_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def random_mask(input_data, mask_prob, mask=None, batch_size=8):
    """
    :param batch_size:
    :param input_data: [B * N, T]
    :param mask_prob:
    :param mask:
    :return:
    """
    if mask is not None:
        mask = mask
    else:
        mask_list = []
        mask_shape = (input_data.shape[0] // batch_size, input_data.shape[1])
        for i in range(batch_size):
            mask = (torch.rand(mask_shape) > mask_prob).float()
            mask_list.append(mask)
        mask = torch.cat(mask_list, dim=0).to(input_data.device)

    masked_data = input_data * mask + (1 - mask) * -1

    return masked_data, mask


def add_small_random_gaussian_noise(input_data, mean=0, std=0.2, batch_size=8):
    noise_list = []
    noise_shape = (input_data.shape[0] // batch_size, input_data.shape[1])
    for i in range(batch_size):
        noise = torch.normal(mean=mean, std=std, size=noise_shape)
        noise_list.append(noise)
    noise = torch.cat(noise_list, dim=0).to(input_data.device)

    noisy_data = input_data + noise
    return noisy_data


def random_mask_edges(edges, mask_prob, batch_size=8):
    edge_num = edges.shape[1] // batch_size
    exist_edges = []
    for i in range(batch_size):
        mask = torch.rand(edge_num, device=edges.device) > mask_prob
        exist_edge = edges[:, i * edge_num:(i + 1) * edge_num][:, mask]
        exist_edges.append(exist_edge)
    edges = torch.cat(exist_edges, dim=1)
    return edges


def transpose_node_to_graph(feature, batch_size, hidden_dim):
    """transpose [B * N, F] to [B * F, N]"""
    feature = feature.view(batch_size, -1, hidden_dim).transpose(1, 2)
    feature = feature.reshape(batch_size * hidden_dim, -1)
    return feature


if __name__ == '__main__':
    x = torch.rand(2 * 3, 12) * 2 - 1
    y = torch.rand(2 * 3, 12) * 2 - 1
    import matplotlib.pyplot as plt

    node = 1
    plt.plot(x[node].numpy(), label='x')

    x = add_small_random_gaussian_noise(x, mean=0, std=0.8, batch_size=2)
    # x, _ = random_mask(x, mask_prob=0.1)
    plt.plot(x[node].numpy(), label='x shifted')
    plt.legend()
    plt.show()
