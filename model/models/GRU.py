import torch.nn as nn
import torch


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

        self.gru = nn.GRU(self.in_var, self.hidden_dims, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dims, self.output_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size, capacity, input_len, features = x_enc.size()
        x_enc = x_enc.view(batch_size * capacity, input_len, features)
        h_0 = torch.zeros(self.num_layers, batch_size * capacity, self.hidden_dims).to(x_enc.device)
        gru_out, _ = self.gru(x_enc, h_0)
        gru_input = gru_out[:, -1, :]
        pred = self.fc(gru_input)
        pred = pred.view(batch_size, capacity, self.output_len)
        return pred
