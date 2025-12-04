import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch, Data

WIN = 3
DECOMP = 25


class MultiScaleSeriesDecomp(nn.Module):
    def __init__(self, kernel_sizes):
        """
        kernel_sizes: list of odd integers (e.g., [3, 7, 15])
        """
        super().__init__()
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        # x: (B, L, D)
        trends = []
        for k in self.kernel_sizes:
            padding = (k - 1) // 2
            smooth = F.avg_pool1d(x.transpose(1, 2), kernel_size=k, stride=1, padding=padding)
            trends.append(smooth.transpose(1, 2))  # back to (B, L, D)
        trend = sum(trends) / len(trends)
        seasonal = x - trend
        return seasonal, trend


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 d_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=0.1,
                 act_dropout=None):
        super(EncoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = MultiScaleSeriesDecomp(kernel_sizes=[3, 7, 15, 31])
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=attn_dropout)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, src):
        residual = src
        src, _ = self.self_attn(src, src, src)
        src = residual + self.dropout1(src)
        src, _ = self.decomp(src)
        residual = src
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout1(src)
        src, _ = self.decomp(src)
        return src


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 d_feedforward,
                 dropout=0.1,
                 trends_out=134,
                 activation="relu",
                 attn_dropout=0.1,
                 act_dropout=None):
        super(DecoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = MultiScaleSeriesDecomp(kernel_sizes=[3, 7, 15, 31])
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=attn_dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=attn_dropout)

        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.linear_trend = nn.Conv1d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=WIN,
            padding=(WIN // 2),  # To mimic 'SAME' padding
            groups=1)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, src, memory):
        residual = src
        src, _ = self.self_attn(src, src, src)
        src = residual + self.dropout1(src)

        src, trend1 = self.decomp(src)

        residual = src
        src, _ = self.cross_attn(src, memory, memory)
        src = residual + self.dropout1(src)

        src, trend2 = self.decomp(src)

        residual = src
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout1(src)

        src, trend3 = self.decomp(src)
        trend = trend1 + trend2 + trend3
        trend = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)
        return src, trend


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.num_encoder_layer = args.e_layers
        self.dropout = args.dropout
        self.in_var = args.in_var
        self.out_var = args.out_var
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.capacity = args.capacity

        self.enclist = nn.ModuleList()
        self.drop = nn.Dropout(self.dropout)
        for _ in range(self.num_encoder_layer):
            self.enclist.append(
                EncoderLayer(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_feedforward=self.d_model * 2,
                    dropout=self.dropout,
                    activation="gelu",
                    attn_dropout=self.dropout,
                    act_dropout=self.dropout,
                )
            )

    def forward(self, batch_x):
        for lin in self.enclist:
            batch_x = lin(batch_x)
        batch_x = self.drop(batch_x)
        return batch_x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.num_decoder_layer = args.d_layers
        self.dropout = args.dropout
        self.in_var = args.in_var
        self.out_var = args.out_var
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.capacity = args.capacity

        self.declist = nn.ModuleList()
        self.drop = nn.Dropout(self.dropout)

        for _ in range(self.num_decoder_layer):
            self.declist.append(
                DecoderLayer(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_feedforward=self.d_model * 2,
                    trends_out=self.capacity,
                    dropout=self.dropout,
                    activation="gelu",
                    attn_dropout=self.dropout,
                    act_dropout=self.dropout,
                )
            )

    def forward(self, season, trend, enc_output):
        for lin in self.declist:
            season, trend_part = lin(season, enc_output)
            trend += trend_part
        return season, trend


class SpatialTemporalConv(nn.Module):
    def __init__(self, seq_len, capacity, in_var, output_dim, dropout=0.1):
        super(SpatialTemporalConv, self).__init__()
        self.capacity = capacity
        self.in_var = in_var
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.gcn = pyg_nn.SAGEConv(self.in_var, self.in_var)
        self.feature_gcn = pyg_nn.SAGEConv(self.seq_len, self.seq_len)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(
            in_channels=self.in_var,
            out_channels=self.in_var,
            kernel_size=3,
            padding=1,
            bias=False)
        self.conv2 = nn.Conv1d(
            in_channels=self.in_var,
            out_channels=self.in_var,
            kernel_size=5,
            padding=2,
            bias=False
        )
        self.fc = nn.Linear(self.in_var * 3, self.in_var)
        # self.fc = nn.Linear(self.seq_len * 3, self.seq_len)

    def forward(self, src, graph, feature_graph):
        residual = src
        bz, capacity, seq_len, in_var = src.shape
        src_one = src.permute(0, 2, 1, 3).contiguous().view(bz * seq_len, capacity,
                                                            in_var)  # (seq_len * bz, capacity, in_var)

        edge_index = graph.edge_index
        batch = Batch.from_data_list([Data(x=src_one[i], edge_index=edge_index) for i in range(src_one.shape[0])])
        out = self.gcn(batch.x, batch.edge_index)
        spatial_fea = out.view(bz, seq_len, self.capacity, self.in_var)
        spatial_fea = spatial_fea.permute(0, 2, 1, 3)

        feature_edge_index = feature_graph.edge_index
        src_two = src.permute(0, 1, 3, 2).contiguous().view(bz * capacity, in_var, seq_len)
        feature_batch = Batch.from_data_list(
            [Data(x=src_two[i], edge_index=feature_edge_index) for i in range(src_two.shape[0])])
        feature_out = self.feature_gcn(feature_batch.x, feature_batch.edge_index)
        fea_fea = feature_out.view(bz, self.capacity, self.in_var, seq_len)
        fea_fea = fea_fea.permute(0, 1, 3, 2)

        src_three = src.reshape(bz * capacity, self.seq_len, self.in_var)
        conv1_out = self.conv1(src_three.transpose(1, 2))
        conv2_out = self.conv2(src_three.transpose(1, 2))
        temporal_fea = conv1_out + conv2_out
        temporal_fea = temporal_fea.transpose(1, 2)
        temporal_fea = temporal_fea.reshape(bz, capacity, seq_len, in_var)

        com_fea = torch.cat((spatial_fea, fea_fea, temporal_fea), dim=-1)
        com_fea = self.fc(com_fea)

        # com_fea = torch.cat((spatial_fea, fea_fea, temporal_fea), dim=-2)
        # com_fea = self.fc(com_fea.transpose(2,3)).transpose(2,3)

        src = residual + self.dropout(com_fea)
        return src


class DataEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, time_var, seq_len, dropout=0.2):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(input_dim, output_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, output_dim)
        )
        self.temporal_embedding = nn.Linear(time_var, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_feature):
        x = self.value_embedding(x) + self.pos_embedding + self.temporal_embedding(time_feature)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.in_var = args.in_var
        self.out_var = args.out_var
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.d_model = args.d_model
        self.capacity = args.capacity
        self.time_var = args.time_var

        self.decomp = MultiScaleSeriesDecomp(kernel_sizes=[3, 7, 15, 31])
        self.enc_embedding = DataEmbedding(self.in_var, self.d_model, self.time_var, self.seq_len)
        self.dec_embedding = DataEmbedding(self.in_var, self.d_model, self.time_var,
                                           self.seq_len + self.pred_len)
        self.st_conv_encoder = SpatialTemporalConv(self.seq_len, self.capacity, self.in_var, self.d_model)
        self.linear_decoder = nn.Linear(self.capacity * self.in_var, self.d_model)
        self.enc = Encoder(args)
        self.dec = Decoder(args)
        self.pred_nn = nn.Linear(self.d_model, 1)
        self.predy_nn = nn.Linear(self.seq_len + self.pred_len, self.pred_len)

    def forward(self, batch_x, time_x=None, time_y=None, graph=None, feature_graph=None):
        device = batch_x.device
        bz, capacity, seq_len, in_var = batch_x.shape

        batch_x = self.st_conv_encoder(batch_x, graph, feature_graph)

        batch_x = batch_x.reshape(bz * capacity, seq_len, in_var)
        time_x = time_x.reshape(bz * capacity, -1, self.time_var)
        time_y = time_y.reshape(bz * capacity, -1, self.time_var)

        seasonal_init, trend_init = self.decomp(batch_x)
        mean = torch.mean(batch_x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)[:, :, -1].unsqueeze(-1)
        zeros = torch.zeros(bz * self.capacity, self.pred_len, self.in_var, device=device)
        trend_init = torch.cat([trend_init[:, :, -1].unsqueeze(-1), mean], dim=1)
        seasonal_init = torch.cat([seasonal_init, zeros], 1)

        batch_x = self.enc_embedding(batch_x, time_x)
        seasonal_init = self.dec_embedding(seasonal_init, time_y)

        enc_output = self.enc(batch_x)
        pred, trend = self.dec(seasonal_init, trend_init, enc_output)
        pred = self.pred_nn(pred)
        pred_y = pred + trend
        pred_y = pred_y.squeeze()
        pred_y = pred_y.reshape(bz, capacity, -1)
        pred_y = pred_y[:, :, -self.pred_len:]
        return pred_y
