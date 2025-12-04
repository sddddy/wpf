import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data,Batch


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(padding, 0))

    def forward(self, x):
        # x: [B, C, T, N]
        return self.conv(x)  # -> [B, out_C, T, N]


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TemporalConv(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.gcn = GCNConv(out_channels, out_channels)
        self.temporal2 = TemporalConv(out_channels, out_channels)
        self.num_nodes = num_nodes

    def forward(self, x, graph):
        # x: [B, C_in, T, N]
        x = self.temporal1(x)
        x = self.relu(x)
        B, C, T, N = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, T, N, C]
        x = x.reshape(-1, N, C)

        batch = Batch.from_data_list([Data(x=x[i], edge_index=graph.edge_index) for i in range(x.shape[0])])
        out = self.gcn(batch.x, batch.edge_index)
        x = out.reshape(B, T, N, C).permute(0, 3, 1, 2)  # [B, C, T, N]
        x = self.temporal2(x)  # [B, C, T, N]
        return x


class Model(nn.Module):
    def __init__(self, args, num_layers=2):
        super(Model, self).__init__()
        self.args = args
        self.in_var = args.in_var
        self.out_var = args.out_var
        self.input_len = args.seq_len
        self.output_len = args.pred_len
        self.hidden_dims = args.d_model
        self.capacity = args.capacity
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        input_dim = args.in_var
        for i in range(self.num_layers):
            self.blocks.append(
                STGCNBlock(
                    in_channels=input_dim if i == 0 else self.hidden_dims,
                    out_channels=self.hidden_dims,
                    num_nodes=self.capacity
                )
            )
        self.output_layer = nn.Conv2d(
            self.hidden_dims, self.out_var * self.output_len, kernel_size=(1, 1)
        )

    def forward(self, batch_x, time_x=None, time_y=None, graph=None, feature_graph=None):
        # batch_x: [B, N, T, F] -> [B, F, T, N]
        x = batch_x.permute(0, 3, 2, 1)
        for block in self.blocks:
            x = block(x, graph)

        # x: [B, C, T, N] → 取最后一个时间步
        x = self.output_layer(x)  # [B, out_var * out_len, T, N]
        x = x[:, :, -1, :]  # [B, out_var * out_len, N]
        x = x.permute(0, 2, 1)
        return x
