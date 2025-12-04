import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

from TransGAT.models.AGSTNet import SpatialTemporalConv

DECOMP = 25


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        t_x = x.transpose(1, 2)
        padding = (self.kernel_size - 1) // 2
        mean_x = F.avg_pool1d(t_x, kernel_size=self.kernel_size, stride=1, padding=padding)
        mean_x = mean_x.transpose(1, 2)
        return x - mean_x, mean_x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.in_var = args.in_var
        self.out_var = args.out_var
        self.input_len = args.seq_len
        self.output_len = args.pred_len
        self.hidden_dims = args.d_model
        self.capacity = args.capacity

        self.decompsition = SeriesDecomp(DECOMP)
        self.st_conv_encoder = SpatialTemporalConv(self.input_len, self.capacity, self.in_var, self.hidden_dims)
        self.Linear_Seasonal = nn.Linear(self.input_len, self.output_len)
        self.Linear_Trend = nn.Linear(self.input_len, self.output_len)

    def forward(self, x, time_x, time_y, graph=None, feature_graph=None):
        bz, capacity, input_len, in_var = x.size()
        x = self.st_conv_encoder(x, graph, feature_graph)
        x = x.reshape(bz * capacity, input_len, in_var)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)[:, :, -1]
        x = x.reshape(bz, capacity, self.output_len)
        return x
