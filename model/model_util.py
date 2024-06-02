import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from sklearn.metrics import mean_absolute_error
from torch.utils.tensorboard import SummaryWriter

from model.Gboot import Gboot


def get_model(device, model_name, in_feature, out_feature, node_num, adj,
              hidden_feature=64, edge_mask=0, add_history=True):
    model = None

    if model_name == 'Gboot':
        model = Gboot(in_feature=in_feature, out_feature=out_feature, node_num=node_num,
                      hidden_feature=hidden_feature,
                      edge_mask=edge_mask, add_history=add_history)

        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

        model.backbone_target.load_state_dict(model.backbone_online.state_dict())
        for param in model.backbone_target.parameters():
            param.requires_grad = False

    model.to(device)

    return model


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pyg.seed_everything(seed)


def init_tensorboard_writer(tensorboard_dir):
    import os
    os.makedirs(tensorboard_dir, exist_ok=True)
    for file in os.listdir(tensorboard_dir):
        os.remove(tensorboard_dir + "/" + file)
    writer = SummaryWriter(tensorboard_dir)
    return writer


def cal_loss(out, data, mode='train'):
    loss = F.huber_loss(out, data.y)
    if mode == 'train':
        loss_mae = loss
    else:
        loss_mae = mean_absolute_error(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())

    return loss, loss_mae
