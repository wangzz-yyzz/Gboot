import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GATv2Conv


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, feature_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.feature_dim = feature_dim

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)  # B,F,H,T,T
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, v)  # B,F,H,T,v_dim
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, k_dim, v_dim, headers, feature_dim):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.headers = headers
        self.feature_dim = feature_dim
        self.w_q = nn.Linear(model_dim, k_dim * headers, bias=False)
        self.w_k = nn.Linear(model_dim, k_dim * headers, bias=False)
        self.w_v = nn.Linear(model_dim, v_dim * headers, bias=False)
        self.fc = nn.Linear(v_dim * headers, model_dim, bias=False)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v):
        residual, batch_size = q, q.size(0)

        q = self.w_q(q).view(batch_size, self.feature_dim, -1, self.headers, self.k_dim)  # B,F,T,H,q_dim
        k = self.w_k(k).view(batch_size, self.feature_dim, -1, self.headers, self.k_dim)  # B,F,T,H,k_dim
        v = self.w_v(v).view(batch_size, self.feature_dim, -1, self.headers, self.v_dim)  # B,F,T,H,v_dim

        q = q.transpose(2, 3)  # B,F,H,T,q_dim
        k = k.transpose(2, 3)  # B,F,H,T,k_dim
        v = v.transpose(2, 3)  # B,F,H,T,v_dim

        context = ScaledDotProductAttention(self.k_dim, self.feature_dim)(q, k, v)  # B,F,H,T,v_dim
        context = context.transpose(2, 3).reshape(batch_size, self.feature_dim, -1, self.headers * self.v_dim)  # B,F,T,H*v_dim
        output = self.fc(context)  # B,F,T,model_dim
        output = output + residual

        output = self.layer_norm(output)
        return output


class Embedding(nn.Module):
    def __init__(self, x_len, embedding_dim, features, embedding_type='T'):
        super(Embedding, self).__init__()
        self.x_len = x_len
        self.features = features
        self.embedding_type = embedding_type
        self.embedding_layer = nn.Embedding(x_len, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, batch_size):
        if self.embedding_type == 'T':
            pos = torch.arange(self.x_len, dtype=torch.long).to(x.device)
            pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.features, self.x_len)
            embedding = x.permute(0, 2, 3, 1) + self.embedding_layer(pos)
        else:
            pos = torch.arange(self.x_len, dtype=torch.long).to(x.device)
            pos = pos.unsqueeze(0).expand(batch_size, self.x_len)
            embedding = x + self.embedding_layer(pos)

        embedding = self.norm(embedding)
        return embedding


class GTU(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, 1))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x


class GLU(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(GLU, self).__init__()
        self.in_channels = in_channels
        self.silu = nn.SiLU()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, 1))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x = torch.mul(self.silu(x_p), x_q)
        return x


class GbootBlock(nn.Module):
    def __init__(self,
                 in_features, out_features,
                 node_num, time_num,
                 model_dim, k_dim, v_dim,
                 attention_headers, gat_headers,
                 attention_type,
                 gate_type,
                 if_layer_norm):
        super(GbootBlock, self).__init__()

        self.gat_headers = gat_headers
        self.attention_type = attention_type
        self.if_layer_norm = if_layer_norm

        self.relu = nn.ReLU(inplace=True)

        self.feature_temporal_mixer = nn.Sequential(
            nn.Conv2d(time_num, model_dim, kernel_size=(1, in_features)),
        )

        self.EmbedT = Embedding(time_num, node_num, in_features, 'T')
        self.EmbedS = Embedding(node_num, model_dim, in_features, 'S')

        self.TAt = MultiHeadAttention(node_num, k_dim, v_dim, attention_headers, in_features)
        self.FAt = MultiHeadAttention(node_num, k_dim, v_dim, attention_headers, time_num)

        self.graph_conv = GATv2Conv(model_dim, time_num, heads=gat_headers, dropout=0.01)

        if gate_type == 'GTU':
            self.gate3 = GTU(out_features, 3)
            self.gate5 = GTU(out_features, 5)
            self.gate7 = GTU(out_features, 7)
        else:
            self.gate3 = GLU(out_features, 3)
            self.gate5 = GLU(out_features, 5)
            self.gate7 = GLU(out_features, 7)

        self.residual_conv = nn.Conv2d(in_features, out_features, kernel_size=(1, 1), stride=(1, 1))

        self.dropout = nn.Dropout(p=0.02)
        self.fc = nn.Sequential(
            nn.Linear(3 * time_num - 12, time_num),
            nn.Dropout(0.02),
        )
        self.ln = nn.LayerNorm(out_features)

    def forward(self, data, edges):
        batch_size, node_num, features_num, time_num = data.shape  # B,N,F,T

        if features_num == 3 or features_num == 1:
            x = self.EmbedT(data, batch_size)  # B,F,T,N
        else:
            x = data.permute(0, 2, 3, 1)  # B,F,T,N

        if self.attention_type == 'T':
            x = self.TAt(x, x, x)  # B,F,T,N
        elif self.attention_type == 'F':
            x = x.permute(0, 2, 1, 3)  # B,T,F,N
            x = self.FAt(x, x, x)  # B,T,F,N
            x = x.permute(0, 2, 1, 3)  # B,F,T,N
        elif self.attention_type == 'TF':
            x = self.TAt(x, x, x)
            x = x.permute(0, 2, 1, 3)
            x = self.FAt(x, x, x)
            x = x.permute(0, 2, 1, 3)

        x = self.dropout(x)  # B,F,T,N

        x = x.permute(0, 2, 3, 1)  # B,T,N,F
        x = self.feature_temporal_mixer(x)  # B,model_dim,N,1
        x = x.squeeze(-1)  # B,model_dim,N
        x = x.permute(0, 2, 1)  # B,N,model_dim

        x = self.EmbedS(x, batch_size)  # B,N,model_dim
        x = self.dropout(x)  # B,N,model_dim

        x = x.view(batch_size * node_num, -1)  # B*N,model_dim
        x = self.graph_conv(x, edges)  # B,N,F*T
        x = x.view(batch_size, node_num, self.gat_headers, time_num)  # B,N,F,T

        res_gate = x.permute(0, 2, 1, 3)  # B,F,N,T

        x = [self.gate3(res_gate), self.gate5(res_gate), self.gate7(res_gate)]
        x = torch.cat(x, dim=-1)  # B,F,N,3T-12
        x = self.fc(x)  # B,F,N,T

        if features_num == 3 or features_num == 1:
            x = self.relu(x)  # B,F,N,T
        else:
            x = self.relu(res_gate + 0.5 * x)  # B,F,N,T

        if features_num == 3 or features_num == 1:
            x_residual = self.residual_conv(data.permute(0, 2, 1, 3))  # B,F,N,T
            x = x_residual + x
        else:
            x_residual = data.permute(0, 2, 1, 3)  # B,F,N,T
            x = x_residual + 0.5 * x

        x = F.relu(x).permute(0, 3, 2, 1)  # B,T,N,F

        if self.if_layer_norm:
            x = self.ln(x)

        x = x.permute(0, 2, 3, 1)  # B,N,F,T

        return x


class GbootModule(nn.Module):
    def __init__(self,
                 node_num, time_num,
                 model_dim, k_dim, v_dim,
                 attention_headers, gat_headers,
                 block_num, output_dim,
                 edge_mask, add_history):

        super(GbootModule, self).__init__()

        self.time_num = time_num
        self.output_dim = output_dim
        self.edge_mask = edge_mask
        self.add_history = add_history
        self.in_features = 3 if add_history else 1
        self.BlockList = nn.ModuleList([])
        self.BlockList.append(GbootBlock(
            in_features=self.in_features, out_features=gat_headers,
            node_num=node_num, time_num=time_num,
            model_dim=model_dim, k_dim=k_dim, v_dim=v_dim,
            attention_headers=attention_headers,
            gat_headers=gat_headers,
            attention_type='T', gate_type='GTU',
            if_layer_norm=True))

        self.BlockList.append(GbootBlock(
            in_features=gat_headers, out_features=gat_headers,
            node_num=node_num, time_num=time_num,
            model_dim=model_dim, k_dim=k_dim, v_dim=v_dim,
            attention_headers=attention_headers,
            gat_headers=gat_headers,
            attention_type='F', gate_type='GLU',
            if_layer_norm=True))

        self.final_conv = nn.Conv2d(time_num * block_num, output_dim,
                                    kernel_size=(1, gat_headers))

    def forward(self, data, feature='x'):
        edges = data.edge_index
        batch_size = data.num_graphs

        if feature == 'x':
            features = data.x
        elif feature == 'aug':
            features = data.aug
        else:
            raise ValueError('feature must be x or aug')

        features = features.view(batch_size, -1, self.time_num)
        features = features.unsqueeze(2)

        if self.add_history:
            trend_x = data.trend_x.view(batch_size, -1, 2, self.time_num)
            features = torch.cat([features, trend_x], dim=2)

        need_concat = []
        for index, block in enumerate(self.BlockList):
            features = block(features, edges)
            need_concat.append(features)

        final_x = torch.cat(need_concat, dim=-1)  # B,N,F,T*block_num
        output1 = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)  # B,N,output_dim
        output = output1.view(-1, self.output_dim)  # B*N,output_dim

        return output


def get_gboot_model(node_num, output_dim, time_num, edge_mask, add_history):
    gat_headers = 32
    node_num = node_num
    model_dim = 256
    k_dim = 32
    v_dim = 32
    attention_headers = 3
    edge_mask = edge_mask
    add_history = add_history

    model = GbootModule(node_num, time_num,
                        model_dim, k_dim, v_dim,
                        attention_headers, gat_headers,
                        block_num=2, output_dim=output_dim,
                        edge_mask=edge_mask, add_history=add_history)

    return model
