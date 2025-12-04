import random

from torch.utils.data import DataLoader

from TransGAT.data_provider.data_factory import data_provider
from TransGAT.exp.exp_basic import Exp_Basic
import torch.nn as nn
import numpy as np

from TransGAT.utils.metrics import metric
import time
import torch
import os
import TransGAT.loss as loss_factory
from TransGAT.loss import MAELoss
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from TransGAT.utils.tools import visual


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.graph = None
        self.feature_graph = None
        self.args = args

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # train
    def get_args(self):
        return self.args

    def get_data(self, data, setting, flag):
        data_set, data_loader = data_provider(self.args, data, setting, flag)
        return data_set, data_loader

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), self.args.lr)

    def get_criterion(self):
        criterion = nn.MSELoss(reduction='mean')
        return criterion

    def val(self, graph, feature_graph, val_loader, criterion):
        val_loss = []
        mae_loss = []
        args = self.args
        mae_criterion = MAELoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i, (batch_x, batch_y, time_x, time_y) in enumerate(val_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            time_x = time_x.float().to(self.device)
            time_y = time_y.float().to(self.device)

            if not args.input_flag:
                dec_inp = torch.zeros_like(batch_y[:, :, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_x[:, :, -args.label_len:, :], dec_inp], dim=2).float().to(device)
                time_y = torch.cat([time_x[:, :, -args.label_len:, :], time_y[:, :, -args.pred_len:, :]], dim=2)
                pred_y = self.model(batch_x, time_x, dec_inp, time_y)
            else:
                pred_y = self.model(batch_x, time_x, time_y, graph, feature_graph)
            gold_y = batch_y[:, :, :, -1]
            loss = criterion(pred_y, gold_y)
            mae = mae_criterion(pred_y, gold_y)
            val_loss.append(loss.item())
            mae_loss.append(mae.item())

        val_loss = np.average(val_loss)
        mae_loss = np.average(mae_loss)
        return val_loss, mae_loss

    def test(self, df_raw, setting):
        args = self.args
        capacity = self.args.capacity
        model_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")

        self.model.load_state_dict(torch.load(model_path))

        test_data, test_loader = self.get_data(df_raw, setting, flag="test")

        # visual path
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pred_batch = []
        gold_batch = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, time_x, time_y) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                time_x = time_x.float().to(self.device)
                time_y = time_y.float().to(self.device)

                if not args.input_flag:
                    dec_inp = torch.zeros_like(batch_y[:, :, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_x[:, :, -args.label_len:, :], dec_inp], dim=2).float().to(self.device)
                    time_y = torch.cat([time_x[:, :, -args.label_len:, :], time_y[:, :, -args.pred_len:, :]], dim=2)
                    # (batch_size,capacity,pred_len)
                    pred_y = self.model(batch_x, time_x, dec_inp, time_y)
                else:
                    pred_y = self.model(batch_x, time_x, time_y, self.graph, self.feature_graph)

                gold_y = batch_y[:, :, :, -1]

                pred_y = pred_y.detach().cpu().numpy()
                gold_y = gold_y.detach().cpu().numpy()

                pred_batch.append(pred_y)
                gold_batch.append(gold_y)

                if i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    random_number = random.randint(0, capacity - 1)
                    gt = np.concatenate((input_x[0, random_number, :, -1], gold_y[0, random_number, :]), axis=0)
                    pd = np.concatenate((input_x[0, random_number, :, -1], pred_y[0, random_number, :]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = np.concatenate(pred_batch, axis=0)
        trues = np.concatenate(gold_batch, axis=0)
        mae, mse, rmse, mape = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def train_and_val(self, df_raw, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_data, train_loader = self.get_data(df_raw, setting, flag="train")
        val_data, val_loader = self.get_data(df_raw, setting, flag="val")
        test_data, test_loader = self.get_data(df_raw, setting, flag="test")

        self.graph = train_data.graph.to(self.device)
        self.feature_graph = train_data.feature_graph.to(self.device)

        mae_criterion = MAELoss()
        criterion = getattr(loss_factory, self.args.loss)()
        optimizer = self.get_optimizer(self.model)
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        # min_lr = 5e-05
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

        early_stopping = EarlyStopping(self.args)
        global_step = 0
        epoch_start_time = start_train_time = time.time()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            mae_loss = []
            self.model.train()
            for i, (batch_x, batch_y, time_x, time_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                time_x = time_x.float().to(self.device)
                time_y = time_y.float().to(self.device)
                optimizer.zero_grad()
                gold_y = batch_y[:, :, :, -1]

                if not self.args.input_flag:
                    dec_inp = torch.zeros_like(batch_y[:, :, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_x[:, :, -self.args.label_len:, :], dec_inp], dim=2).float().to(
                        self.device)
                    time_y = torch.cat([time_x[:, :, -self.args.label_len:, :], time_y[:, :, -self.args.pred_len:, :]],
                                       dim=2)
                    pred_y = self.model(batch_x, time_x, dec_inp, time_y)
                else:
                    pred_y = self.model(batch_x, time_x, time_y, self.graph, self.feature_graph)

                loss = criterion(pred_y, gold_y)
                mae = mae_criterion(pred_y, gold_y)
                train_loss.append(loss.item())
                mae_loss.append(mae.item())
                loss.backward()
                optimizer.step()
                global_step += 1
                # if global_step % args["log_per_steps"] == 0:
                #     print(
                #         f"Step {global_step} Train MSE-Loss: {loss.item()} RMSE-Loss: {torch.sqrt(loss).item()} MAE-Loss:{mae.item()}")
            self.model.eval()
            with torch.no_grad():
                val_loss, val_mae_loss = self.val(self.graph, self.feature_graph, val_loader, criterion)
                test_loss, test_mae_loss = self.val(self.graph, self.feature_graph, test_loader, criterion)

            if self.args.is_debug:
                train_loss = np.average(train_loss)
                mae_loss = np.average(mae_loss)
                epoch_end_time = time.time()
                print(
                    "Epoch: {}, \nTrain Loss: {:.8f},Train mae: {:.8f} \nValidation Loss: {:.8f},Validation mae: {:.8f}\nTest Loss: {:.8f},Test mae: {:.8f}".format(
                        epoch, train_loss, mae_loss,
                        val_loss, val_mae_loss, test_loss, test_mae_loss))
                print("Elapsed time for epoch-{}: {:.2f} secs".format(epoch, epoch_end_time - epoch_start_time))
                epoch_start_time = epoch_end_time

            current_lr = optimizer.param_groups[0]['lr']
            # if current_lr >= min_lr:
            scheduler.step()
            print(f"Epoch {epoch}, Learning rate: {optimizer.param_groups[0]['lr']}")

            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("training early stop!")
                break

        if self.args.is_debug:
            end_train_time = time.time()
            print("\nTotal time in training {} turbines is "
                  "{:.2f} secs".format(self.args.capacity, end_train_time - start_train_time))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


class EarlyStopping:
    def __init__(self, args):
        self.patience = args.patience
        self.verbose = args.verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = args.delta
        self.best_model = False
        self.use_graph = args.use_graph

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        self.best_model = True
        self.val_loss_min = val_loss
        pathname = "checkpoint.pth"
        path = os.path.join(path, pathname)
        torch.save(model.state_dict(), path)
