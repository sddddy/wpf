import torch
import torch.nn as nn

from TransGAT.models.AGSTNet import SpatialTemporalConv


class Model(nn.Module):
    def __init__(self, args, num_layers=1):
        super(Model, self).__init__()
        self.args = args
        self.in_var = args.in_var
        self.out_var = args.out_var
        self.input_len = args.seq_len
        self.output_len = args.pred_len
        self.hidden_dims = args.d_model
        self.capacity = args.capacity
        self.num_layers = num_layers

        self.st_conv_encoder = SpatialTemporalConv(self.input_len, self.capacity, self.in_var, self.hidden_dims)
        self.lstm = nn.LSTM(self.in_var, self.hidden_dims, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dims, self.output_len)

    def forward(self, x, time_x, time_y, graph=None, feature_graph=None):
        batch_size, capacity, input_len, features = x.size()
        x = self.st_conv_encoder(x, graph, feature_graph)
        x = x.view(batch_size * capacity, input_len, features)

        h_0 = torch.zeros(self.num_layers, batch_size * capacity, self.hidden_dims).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size * capacity, self.hidden_dims).to(x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_input = lstm_out[:, -1, :]
        pred = self.fc(lstm_input)
        pred = pred.view(batch_size, capacity, self.output_len)
        return pred
