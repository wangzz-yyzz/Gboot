import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATv2Conv


class Backbone(torch.nn.Module):
    def __init__(self, in_feature, out_feature, node_num, hidden_feature=32, headers=4, block_num=3):
        super(Backbone, self).__init__()

        self.hidden_feature = hidden_feature
        self.headers = headers
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.node_num = node_num
        self.block_num = block_num
        self.add_emb = False

        self.layer1 = GATv2Conv(self.in_feature, self.hidden_feature, heads=self.headers)
        self.layer2 = GATv2Conv(self.hidden_feature * self.headers, self.hidden_feature, heads=self.headers)

        self.dense = nn.Linear(self.hidden_feature * self.headers, self.out_feature, bias=False)

        self.spatial_emb = nn.Parameter(torch.FloatTensor(1, self.node_num, 1))
        self.temporal_emb = nn.Parameter(torch.FloatTensor(1, 1, self.in_feature))

        self.activation = nn.ReLU()

    def forward(self, data, edges):
        # data: [N * B, F]
        features = data

        if self.add_emb:
            features = features.view(-1, self.node_num, self.in_feature)
            features = features + self.temporal_emb + self.spatial_emb
            features = features.view(-1, self.in_feature)

        features = self.layer1(features, edges)
        features = self.activation(features)

        for _ in range(self.block_num):
            features_2 = self.layer2(features, edges)
            features_2 = self.activation(features_2)
            features = features + 0.5 * features_2

        features = self.dense(features)

        return features
