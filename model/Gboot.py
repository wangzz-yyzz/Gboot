import torch
import torch.nn.functional as F

from model.layers.predictor import Backbone
from model.layers.backbone import get_gboot_model as make_my_model

from prepare.data_util import byol_loss_fn, transpose_node_to_graph


class Gboot(torch.nn.Module):
    def __init__(self, in_feature, out_feature, node_num, hidden_feature=64,
                 edge_mask=0, add_history=True):
        super(Gboot, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.node_num = node_num
        self.hidden_feature = hidden_feature
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.backbone_online = make_my_model(node_num=node_num, output_dim=self.hidden_feature, time_num=12,
                                             edge_mask=edge_mask, add_history=add_history)
        self.backbone_target = make_my_model(node_num=node_num, output_dim=self.hidden_feature, time_num=12,
                                             edge_mask=edge_mask, add_history=add_history)

        self.predictor = Backbone(self.hidden_feature, out_feature, node_num,
                                  hidden_feature=256, headers=3, block_num=0)
        # online predictor
        self.dense_online = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_feature, 2 * self.hidden_feature),
            torch.nn.BatchNorm1d(2 * self.hidden_feature),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * self.hidden_feature, self.hidden_feature),
        )

        self.criterion = byol_loss_fn

        self.edges = None
        self.batch_size = None

    def update_moving_average(self, beta=0.99):
        with torch.no_grad():
            for online_params, target_params in zip(self.backbone_online.parameters(),
                                                    self.backbone_target.parameters()):
                target_params.data = beta * target_params.data + (1.0 - beta) * online_params.data

    def forward(self, data):
        # data.x: [N * B, F]
        self.edges = data.edge_index
        self.batch_size = data.num_graphs

        if not hasattr(data, 'aug'):
            return self.predictor(self.backbone_online(data, feature='x'), self.edges)

        features_x = self.backbone_online(data, feature='x')
        features_t = self.backbone_online(data, feature='aug')

        output = self.predictor(features_x, self.edges)

        x_online = self.dense_online(features_x)
        t_online = self.dense_online(features_t)

        with torch.no_grad():
            x_target = self.backbone_target(data, feature='x')
            t_target = self.backbone_target(data, feature='aug')

        x_online = transpose_node_to_graph(x_online, data.num_graphs, self.hidden_feature)
        t_online = transpose_node_to_graph(t_online, data.num_graphs, self.hidden_feature)
        x_target = transpose_node_to_graph(x_target, data.num_graphs, self.hidden_feature)
        t_target = transpose_node_to_graph(t_target, data.num_graphs, self.hidden_feature)

        loss = (self.criterion(F.normalize(x_online), F.normalize(t_target.detach())) +
                self.criterion(F.normalize(t_online), F.normalize(x_target.detach())))

        loss = loss.mean()

        return output, loss
