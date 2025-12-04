from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import numpy as np
from torch_geometric.data import Data
import pywt
import pickle
import os
from sklearn.feature_selection import mutual_info_regression


class Dataset_Common(Dataset):
    def __init__(self,
                 data,
                 args,
                 flag="train",
                 input_len=36,
                 output_len=72,
                 target="Patv",
                 in_features=None,
                 day_len=144,
                 train_days=15,
                 val_days=3,
                 test_days=3,
                 total_days=21,
                 capacity=134,
                 k_neighbor=5,
                 feature_neighbor=3,
                 setting=None
                 ):
        super().__init__()

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type = type_map[flag]
        self.data = data
        self.args = args
        self.total_days = total_days
        self.val_days = val_days
        self.train_days = train_days
        self.unit_size = day_len
        self.target = target
        self.output_len = output_len
        self.input_len = input_len
        self.capacity = capacity
        self.in_feature = in_features
        self.flag = flag
        self.k_neighbor = k_neighbor
        self.feature_neighbor = feature_neighbor
        self.setting = setting
        self.path = os.path.join(self.args.checkpoints, self.setting)

        self.total_size = total_days * self.unit_size
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.test_size = test_days * self.unit_size

        self.feature_scaler = None
        self.target_scaler = None

        self.__read_data__()

    def process_data(self):
        return 0

    def build_turbine_graph_mic(self, data_x):
        data_edge = data_x[:, :, -1]
        num_nodes = data_edge.shape[0]
        # 计算互信息矩阵
        mi_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    mi_matrix[i, j] = mutual_info_regression(
                        data_edge[i].reshape(-1, 1), data_edge[j]
                    )[0]

        k = self.k_neighbor + 1
        top_k_indices = np.argpartition(mi_matrix, -k, axis=1)[:, -k:]
        rows, _ = np.indices((mi_matrix.shape[0], k))
        k_min = mi_matrix[rows, top_k_indices].min(axis=1).reshape([-1, 1])
        row, col = np.where(mi_matrix >= k_min)
        mask = row != col
        row, col = row[mask], col[mask]
        edges = np.concatenate([row.reshape([-1, 1]), col.reshape([-1, 1])], -1)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        graph = Data(edge_index=edge_index, num_nodes=mi_matrix.shape[0])
        return graph

    def build_turbine_graph(self, data_x):
        data_edge = data_x[:, :, -1]
        edge_w = np.corrcoef(data_edge)

        k = self.k_neighbor + 1
        top_k_indices = np.argpartition(edge_w, -k, axis=1)[:, -k:]
        rows, _ = np.indices((edge_w.shape[0], k))
        k_min = edge_w[rows, top_k_indices].min(axis=1).reshape([-1, 1])
        row, col = np.where(edge_w >= k_min)

        mask = row != col
        row, col = row[mask], col[mask]

        edges = np.stack([row, col], axis=0)
        edge_index = torch.tensor(edges, dtype=torch.long)

        edge_weight = edge_w[row, col]
        edge_attr = torch.tensor(edge_weight, dtype=torch.float)

        graph = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=edge_w.shape[0])
        return graph

    def build_feature_correlation_graph(self, data_x):
        import numpy as np
        import torch
        from torch_geometric.data import Data

        data_edge = data_x.reshape(-1, len(self.in_feature)).T
        edge_w = np.corrcoef(data_edge)
        edge_w = np.abs(np.nan_to_num(edge_w))

        k = self.feature_neighbor + 1
        top_k_indices = np.argpartition(edge_w, -k, axis=1)[:, -k:]
        rows, _ = np.indices((edge_w.shape[0], k))
        k_min = edge_w[rows, top_k_indices].min(axis=1).reshape([-1, 1])
        row, col = np.where(edge_w >= k_min)

        mask = row != col
        row, col = row[mask], col[mask]

        edges = np.stack([row, col], axis=0)
        edge_index = torch.tensor(edges, dtype=torch.long)

        edge_weight = edge_w[row, col]
        edge_attr = torch.tensor(edge_weight, dtype=torch.float)

        feature_graph = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=edge_w.shape[0])
        return feature_graph

    def __read_data__(self):
        df_data = self.data

        time_features = self.args.time_features
        time = df_data[time_features].values
        df_data = df_data.drop(columns=time_features)

        data = df_data.values
        # data=[134,35280,5] mask=[134,35280,5]
        data = np.reshape(data, [self.capacity, -1, len(self.in_feature)])
        time = np.reshape(time, [self.capacity, -1, len(time_features)])

        border1s = [
            0, self.train_size - self.input_len,
               self.train_size + self.val_size - self.input_len
        ]
        border2s = [
            self.train_size, self.train_size + self.val_size,
                             self.train_size + self.val_size + self.test_size
        ]
        border1 = border1s[self.type]
        border2 = border2s[self.type]
        time = time[:, border1:border2, :]
        data_x = data[:, border1:border2, :]

        if self.flag == "train":
            data_x = self.apply_denoising(data_x)
            data_x = self.standardize_training_data(data_x)
        else:
            data_x = self.standardize_val_test_data(data_x)

        self.data_x = data_x
        self.time = time

        if self.flag == "train":
            # 构建134个风机的关系图(一张图)
            graph = self.build_turbine_graph(data_x)
            # 属性联系图
            feature_graph = self.build_feature_correlation_graph(data_x)
            self.feature_graph = feature_graph
            self.graph = graph

    def wavelet_denoising(self, data, wavelet='db4', level=1, thresholding='soft'):
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(data)))
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], uthresh, mode=thresholding)
        return pywt.waverec(coeffs, wavelet)

    def apply_denoising(self, data_x):
        # 小波去噪
        for i in range(self.capacity):
            data_x[i, :, -1] = self.wavelet_denoising(data_x[i, :, -1], wavelet='db4', level=2, thresholding='soft')
        return data_x

    def standardize_training_data(self, data_x):
        # 对训练集进行标准化
        in_feature_indices = [self.in_feature.index(f) for f in self.in_feature if f != self.target]
        target_index = self.in_feature.index(self.target)

        # 对特征和标签分别拟合标准化器
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # 标准化特征
        data_x[:, :, in_feature_indices] = self.feature_scaler.fit_transform(
            data_x[:, :, in_feature_indices].reshape(-1, len(in_feature_indices))).reshape(data_x.shape[0],
                                                                                           data_x.shape[1],
                                                                                           len(in_feature_indices))

        # 标准化标签
        data_x[:, :, target_index] = self.target_scaler.fit_transform(
            data_x[:, :, target_index].reshape(-1, 1)).reshape(data_x.shape[0], data_x.shape[1])

        # 保存标准化器
        with open(self.path + '/feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(self.path + '/target_scaler.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)

        return data_x

    def standardize_val_test_data(self, data_x):
        # 对验证集和测试集进行标准化，使用训练集拟合的标准化器
        in_feature_indices = [self.in_feature.index(f) for f in self.in_feature if f != self.target]
        target_index = self.in_feature.index(self.target)

        # 加载训练集的标准化器
        with open(self.path + '/feature_scaler.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(self.path + '/target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)

        # 使用训练集的标准化参数对验证集和测试集进行标准化
        data_x[:, :, in_feature_indices] = self.feature_scaler.transform(
            data_x[:, :, in_feature_indices].reshape(-1, len(in_feature_indices))).reshape(data_x.shape[0],
                                                                                           data_x.shape[1],
                                                                                           len(in_feature_indices))
        data_x[:, :, target_index] = self.target_scaler.transform(data_x[:, :, target_index].reshape(-1, 1)).reshape(
            data_x.shape[0], data_x.shape[1])

        return data_x

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[:, s_begin:s_end, :]
        seq_y = self.data_x[:, r_begin:r_end, :]
        time_x = self.time[:, s_begin:s_end, :]
        time_y = self.time[:, s_begin:r_end, :]

        seq_x = torch.from_numpy(seq_x).float()
        seq_y = torch.from_numpy(seq_y).float()
        time_x = torch.from_numpy(time_x).float()
        time_y = torch.from_numpy(time_y).float()

        return seq_x, seq_y, time_x, time_y

    def __len__(self):
        return self.data_x.shape[1] - self.input_len - self.output_len + 1
