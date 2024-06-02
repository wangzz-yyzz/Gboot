import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from model.model_util import get_model
from prepare.data_util import get_dataloader, masked_mape_np


def test_all(device, task, model_name, save_model_path, adj, hidden_feature=64, add_history=True):
    test_loader, _ = get_dataloader(task=task, batch_size=64, shuffle=False, mode='test',
                                    x_len=12, y_len=12, verbose=False, trend=add_history,
                                    if_handle_nan=False)

    model = get_model(device=device, model_name=model_name, in_feature=12, out_feature=12,
                      node_num=adj.shape[0], adj=adj, hidden_feature=hidden_feature,
                      edge_mask=0, add_history=add_history)
    _ = model.load_state_dict(torch.load(save_model_path, map_location=device))
    model.eval()

    out_list = []
    y_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)

            out = model(data)

            out = out.cpu().detach().numpy().reshape(-1)
            y = data.y.cpu().detach().numpy().reshape(-1)

            out_list.extend(out)
            y_list.extend(y)

    out_list = np.array(out_list).reshape(-1)
    y_list = np.array(y_list).reshape(-1)
    mae = mean_absolute_error(y_list, out_list)
    rmse = mean_squared_error(y_list, out_list, squared=False)
    mape = masked_mape_np(y_list, out_list, 0)

    print("mae test loss: {:.4f}".format(mae))
    print("rmse test loss: {:.4f}".format(rmse))
    print("mape test loss: {:.4f}%".format(mape))

    return mae, rmse, mape
