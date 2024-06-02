import time

import nni
import numpy as np
import torch
import torch.cuda.amp as amp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from model.model_util import get_model, set_seed, cal_loss, init_tensorboard_writer
from prepare.data_util import get_dataloader, add_small_random_gaussian_noise, random_mask, random_mask_edges
from prepare.data_util import masked_mape_np
from prepare.metrics import test_all

params = {
    "batch_size": 32,
    "a": 0.3,
    "lr": 0.003,
    "std": 0.03,
    "drop": 0.04,
    "edge_mask": 0.2,
    "add_history": True,
    "hidden_feature": 128,
    "seed": 3,
    "task": "PEMS08",
}
use_nni = False
if use_nni:
    params = nni.get_next_parameter()

if __name__ == '__main__':
    batch_size = params['batch_size']
    a = params['a']
    lr = params['lr']
    std = params['std']
    drop = params['drop']
    edge_mask = params['edge_mask']
    add_history = params['add_history']
    hidden_feature = params['hidden_feature']
    seed = params['seed']
    task = params['task']

    epochs = 20
    x_len = 12
    y_len = 12
    model_name = 'Gboot'
    patience = epochs
    current_patience = 0
    w = [1.0]
    half = False
    val_step = 50 * 32 // batch_size
    step_size = 3  # 3
    gamma = 0.5  # 0.5

    save_model_path = ("log/best_" + task + "_" + model_name + "_a-" + str(a) +
                       "_std-" + str(std) + "_drop-" + str(drop) + "_edge-" + str(edge_mask) +
                       "_his-" + str(add_history) + "_seed-" + str(seed) + ".pth")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tensorboard_dir = ("log/tensor/tensor" + task + "_" + model_name + "_a-" + str(a) +
                       "_std-" + str(std) + "_drop-" + str(drop) + "_edge-" + str(edge_mask) +
                       "_his-" + str(add_history) + "_seed-" + str(seed))
    loss_file = ("log/loss/loss_" + task + "_" + model_name + "_a-" + str(a) +
                 "_std-" + str(std) + "_drop-" + str(drop) + "_edge-" + str(edge_mask) +
                 "_his-" + str(add_history) + "_seed-" + str(seed) + ".csv")
    writer = init_tensorboard_writer(tensorboard_dir)
    set_seed(seed=seed)
    print("device: {}".format(device))
    print("params: {}".format(params))

    train_loader, adj = get_dataloader(task=task, batch_size=batch_size, shuffle=True, mode='train',
                                       x_len=x_len, y_len=y_len, trend=add_history,
                                       if_handle_nan=False)
    test_loader, _ = get_dataloader(task=task, batch_size=batch_size, shuffle=False, mode='val',
                                    x_len=x_len, y_len=y_len, verbose=False, trend=add_history,
                                    if_handle_nan=False)

    model = get_model(device=device, model_name=model_name, in_feature=x_len, out_feature=y_len,
                      node_num=adj.shape[0], adj=adj, hidden_feature=hidden_feature,
                      edge_mask=edge_mask, add_history=add_history)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    scaler = amp.GradScaler(enabled=half)

    val_loss_list = []
    step = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        current_patience += 1
        total_loss = 0
        total_loss_feature = 0
        start_time = time.time()

        bar = tqdm(train_loader, disable=True)
        for data in bar:
            step += 1
            model.train()
            optimizer.zero_grad()
            with amp.autocast(enabled=half):
                data = data.to(device)

                data.aug = data.x.detach().clone()
                data.aug, mask = random_mask(data.aug, drop, batch_size=data.num_graphs)
                data.edge_index = random_mask_edges(data.edge_index, edge_mask, batch_size=data.num_graphs)
                data.aug = add_small_random_gaussian_noise(data.aug, std=std, batch_size=data.num_graphs)

                output, loss_trend = model(data)

                loss, loss_mae = cal_loss(output, data)
                loss = (1 - a) * loss + a * loss_trend

                writer.add_scalar('loss', loss_mae.item(), step)
                writer.add_scalar('loss_trend', loss_trend.item(), step)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if model_name == 'Gboot':
                model.update_moving_average()

            total_loss += loss_mae.item()
            total_loss_feature += loss_trend.item()
            bar.set_description("loss: {:.3f}, loss_feature: {:.3f}".format(loss_mae.item(), loss_trend.item()))

            if step % val_step == 0:
                model.eval()
                out_list_val = []
                y_list_val = []
                with torch.no_grad():
                    for data in tqdm(test_loader, desc="val", disable=True):
                        data = data.to(device)

                        output = model(data)

                        output = output.cpu().detach().numpy().reshape(-1)
                        y = data.y.cpu().detach().numpy().reshape(-1)

                        out_list_val.extend(output)
                        y_list_val.extend(y)

                out_list_val = np.array(out_list_val).reshape(-1)
                y_list_val = np.array(y_list_val).reshape(-1)
                mae, rmse, mape = 0, 0, 0
                data_len = len(y_list_val) // len(w)
                for i in range(len(w)):
                    mae += w[i] * mean_absolute_error(y_list_val[i * data_len: (i + 1) * data_len],
                                                      out_list_val[i * data_len: (i + 1) * data_len])
                    rmse += w[i] * mean_squared_error(y_list_val[i * data_len: (i + 1) * data_len],
                                                      out_list_val[i * data_len: (i + 1) * data_len], squared=False)
                    mape += w[i] * masked_mape_np(y_list_val[i * data_len: (i + 1) * data_len],
                                                  out_list_val[i * data_len: (i + 1) * data_len], 0)

                test_loss = mae + rmse + mape

                if test_loss < best_val_loss:
                    current_patience = 0
                    best_val_loss = test_loss
                    torch.save(model.state_dict(),
                               save_model_path)
                    print("save best model")

                end_time = time.time()
                print("step: {}, "
                      "val_mae: {:.3f}, "
                      "val_rmse: {:.3f}, "
                      "val_mape: {:.3f}, "
                      "time: {:.1f}"
                      .format(step,
                              mae,
                              rmse,
                              mape,
                              end_time - start_time))
                writer.add_scalar('val_mae', mae, step)
                writer.add_scalar('val_rmse', rmse, step)
                writer.add_scalar('val_mape', mape, step)

                val_loss_list.append([mae, rmse, mape])
                if use_nni:
                    nni.report_intermediate_result({'default': mae, 'rmse': rmse, 'mape': mape})
        if current_patience >= patience:
            print("early stop")
            break
        print("epoch: {}, "
              "train_loss: {:.3f}, "
              "train_feature: {:.5f} "
              .format(epoch, total_loss / len(train_loader),
                      total_loss_feature / len(train_loader)))
        if epoch == epochs:
            val_step = 20
        scheduler.step()

    print("best_val_loss: {:.3f}".format(best_val_loss))

    mae, rmse, mape = test_all(device, task, model_name, save_model_path, adj,
                               hidden_feature=hidden_feature,
                               add_history=add_history)
    val_loss_list.append([mae, rmse, mape])
    if use_nni:
        nni.report_final_result({'default': mae, 'rmse': rmse, 'mape': mape})

    import pandas as pd
    pd.DataFrame(val_loss_list).to_csv(loss_file, index=False, header=False)
