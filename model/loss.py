import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterMSELoss(nn.Module):
    def __init__(self):
        super(FilterMSELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        # Remove bad input
        cond1 = raw[:, :, col_names["Patv"]] < 0

        cond2 = raw[:, :, col_names["Pab1"]] > 89
        cond2 = torch.logical_or(cond2, raw[:, :, col_names["Pab2"]] > 89)
        cond2 = torch.logical_or(cond2, raw[:, :, col_names["Pab3"]] > 89)

        cond2 = torch.logical_or(cond2,
                                 raw[:, :, col_names["Wdir"]] < -180)
        cond2 = torch.logical_or(cond2, raw[:, :, col_names["Wdir"]] > 180)
        cond2 = torch.logical_or(cond2,
                                 raw[:, :, col_names["Ndir"]] < -720)
        cond2 = torch.logical_or(cond2, raw[:, :, col_names["Ndir"]] > 720)
        cond2 = torch.logical_or(cond2, cond1)

        cond3 = raw[:, :, col_names["Patv"]] == 0
        cond3 = torch.logical_and(cond3,
                                  raw[:, :, col_names["Wspd"]] > 2.5)
        cond3 = torch.logical_or(cond3, cond2)

        cond = torch.logical_not(cond3)
        cond = cond.to(torch.float32)

        return torch.mean(F.mse_loss(pred, gold, reduction='none') * cond)


class MaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        # pred = pred * mask
        # true = true * mask
        # loss = torch.sqrt(self.mse(pred, true)) + self.mae(pred, true)
        loss = self.mse(pred, true)
        return loss

class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, pred, true):
        # pred = pred * mask
        # true = true * mask
        # loss = torch.sqrt(self.mse(pred, true)) + self.mae(pred, true)
        loss = self.mae(pred, true)
        return loss


class MaskedScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        # pred = pred * mask
        # true = true * mask
        _rmse = torch.sqrt(self.mse(pred, true))
        _mae = self.mae(pred, true)
        return _rmse,_mae

