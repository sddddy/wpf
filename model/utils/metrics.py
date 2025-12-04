import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def R2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - (ss_res / ss_tot)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape


# def metric(pred, true):
#     capacity = pred.shape[1]
#     mae_list, mse_list, rmse_list, mape_list, mspe_list, r2_list, corr_list = [], [], [], [], [], [], []
#     for i in range(capacity):
#         mae_list.append(MAE(pred[:, i, :], true[:, i, :]))
#         mse_list.append(MSE(pred[:, i, :], true[:, i, :]))
#         rmse_list.append(RMSE(pred[:, i, :], true[:, i, :]))
#         mape_list.append(MAPE(pred[:, i, :], true[:, i, :]))
#         mspe_list.append(MSPE(pred[:, i, :], true[:, i, :]))
#         r2_list.append(R2(pred[:, i, :], true[:, i, :]))
#         corr_list.append(CORR(pred[:, i, :], true[:, i, :]))
#
#     mae = np.mean(mae_list)
#     mse = np.mean(mse_list)
#     rmse = np.mean(rmse_list)
#     mape = np.mean(mape_list)
#     mspe = np.mean(mspe_list)
#     r2 = np.mean(r2_list)
#     corr = np.mean(corr_list)
#
#     return mae, mse, rmse, mape, mspe, r2, corr
