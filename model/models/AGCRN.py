import torch
import torch.nn as nn
import torch.nn.functional as F


class AGCN(nn.Module):
    """Adaptive Graph Convolution Network"""
    def __init__(self, num_nodes, in_dim, out_dim):
        super(AGCN, self).__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_emb = nn.Parameter(torch.randn(num_nodes, 10))  # 可学习嵌入
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: [B, N, in_dim]
        B, N, _ = x.shape
        support = F.softmax(F.relu(torch.matmul(self.node_emb, self.node_emb.T)), dim=-1)  # [N, N]
        x = self.linear(x)  # [B, N, out_dim]
        out = torch.einsum("ij,bjd->bid", support, x)  # [B, N, out_dim]
        return out


class AGCRNCell(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim):
        super(AGCRNCell, self).__init__()
        self.agcn = AGCN(num_nodes, input_dim + hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x, h):
        # x: [B, N, input_dim], h: [B, N, hidden_dim]
        combined = torch.cat([x, h], dim=-1)  # [B, N, input_dim + hidden_dim]
        conv_out = self.agcn(combined)  # [B, N, hidden_dim]
        h = self.gru(x.reshape(-1, x.shape[-1]), conv_out.reshape(-1, conv_out.shape[-1]))
        h = h.view(x.shape[0], x.shape[1], -1)
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
            AGCRNCell(self.num_nodes, self.in_var, self.hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(self.hidden_dim, self.out_var)
        self.out2in_proj = nn.Linear(self.out_var, self.in_var)

    def forward(self, batch_x, time_x=None, time_y=None, graph=None, feature_graph=None):
        """
        Args:
            batch_x: [B, N, T, F]
        Returns:
            Tensor: [B, N, pred_len, out_var]
        """
        B, N, T, F = batch_x.shape
        device = batch_x.device
        h = torch.zeros(B, N, self.hidden_dim, device=device)

        # Encoder
        for t in range(self.seq_len):
            x_t = batch_x[:, :, t, :]  # [B, N, F]
            for cell in self.rnn_cells:
                h = cell(x_t, h)

        # Decoder (auto-regressive)
        outputs = []
        decoder_input = torch.zeros(B, N, self.in_var, device=device)
        for _ in range(self.pred_len):
            for cell in self.rnn_cells:
                h = cell(decoder_input, h)
            out = self.output_proj(h)  # [B, N, out_var]
            outputs.append(out.unsqueeze(2))  # [B, N, 1, out_var]
            decoder_input = self.out2in_proj(out)

        y_pred = torch.cat(outputs, dim=2)  # [B, N, pred_len, out_var]
        return y_pred.squeeze()
