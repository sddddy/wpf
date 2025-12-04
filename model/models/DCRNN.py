import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class DiffusionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K=2):
        super(DiffusionConv, self).__init__(aggr='add')  # 使用加法聚合
        self.K = K
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: [N, in_channels], edge_index: [2, num_edges]
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = x
        for _ in range(self.K):
            out = self.propagate(edge_index=edge_index, x=out)
        return self.linear(out)

    def message(self, x_j):
        return x_j


class DCRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, diffusion_steps=2):
        super(DCRNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.diff_conv = DiffusionConv(input_dim + hidden_dim, hidden_dim, K=diffusion_steps)
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x, h, edge_index):
        # x: [B, N, input_dim], h: [B, N, hidden_dim]
        B, N, _ = x.shape
        combined = torch.cat([x, h], dim=-1)  # [B, N, input_dim + hidden_dim]
        h_new = []

        for b in range(B):
            conv_out = self.diff_conv(combined[b], edge_index)  # [N, hidden_dim]
            h_b = self.gru(x[b], conv_out)  # [N, hidden_dim]
            h_new.append(h_b)

        h = torch.stack(h_new, dim=0)  # [B, N, hidden_dim]
        return h


class Model(nn.Module):
    def __init__(self, args, num_layers=1):
        super(Model, self).__init__()
        self.args = args
        self.in_var = args.in_var
        self.out_var = args.out_var
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.hidden_dim = args.d_model
        self.num_nodes = args.capacity
        self.num_layers = num_layers

        self.rnn_cells = nn.ModuleList([
            DCRNNCell(self.in_var, self.hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(self.hidden_dim, self.out_var)
        self.out2in_proj = nn.Linear(self.out_var, self.in_var)  # 关键：用于将 out 转换回 decoder 的输入维度

    def forward(self, batch_x, time_x=None, time_y=None, graph=None, feature_graph=None):
        """
        Args:
            batch_x: [B, N, T, F]
            graph: PyG Data object，包含 .edge_index
        Returns:
            Tensor: [B, N, pred_len, out_var]
        """
        B, N, T, F = batch_x.shape
        device = batch_x.device
        edge_index = graph.edge_index

        h = torch.zeros(B, N, self.hidden_dim, device=device)

        # Encoder阶段
        for t in range(self.seq_len):
            x_t = batch_x[:, :, t, :]  # [B, N, F]
            for cell in self.rnn_cells:
                h = cell(x_t, h, edge_index)

        # Decoder阶段：auto-regressive
        outputs = []
        decoder_input = torch.zeros(B, N, self.in_var, device=device)
        for _ in range(self.pred_len):
            for cell in self.rnn_cells:
                h = cell(decoder_input, h, edge_index)
            out = self.output_proj(h)  # [B, N, out_var]
            outputs.append(out.unsqueeze(2))  # [B, N, 1, out_var]
            decoder_input = self.out2in_proj(out)  # 关键：投影为 decoder 的下一步输入

        y_pred = torch.cat(outputs, dim=2)  # [B, N, pred_len, out_var]
        return y_pred.squeeze()
