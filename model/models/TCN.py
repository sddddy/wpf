import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()
        self.trim_size = padding

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = out[:, :, :-self.trim_size]

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = out[:, :, :-self.trim_size]

        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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

        num_channels = [128, 128, 128]  # Adjusted number of channels
        kernel_size = 3
        dropout = 0.2

        self.tcn = TemporalConvNet(self.in_var, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], self.output_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size, capacity, input_len, features = x_enc.size()
        x_enc = x_enc.permute(0, 1, 3, 2).contiguous()
        x_enc = x_enc.view(batch_size * capacity, features, input_len)
        tcn_out = self.tcn(x_enc)
        tcn_out = tcn_out[:, :, -1]
        pred = self.fc(tcn_out)
        pred = pred.view(batch_size, capacity, self.output_len)
        return pred
