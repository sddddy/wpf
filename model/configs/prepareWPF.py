from TransGAT.configs.prepareBase import PrepareBase

in_features = ['Pab_max', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Prtv', 'Patv']
drop_cols = ['TurbID', 'Tmstamp', 'Pab1', 'Pab2', 'Pab3']
time_features = ['Day', 'Hour', 'Minute']


class PrepareWPF(PrepareBase):
    """WPF 数据集的配置"""

    def __init__(self):
        super().__init__()
        self.common_settings.update({
            "data": "WPF",
            "data_path": "../data/WPF",
            "filename": "wtbdata.csv",
            "target": "Patv",
            "in_features": in_features,
            "time_features": time_features,
            "drop_cols": drop_cols,
            "capacity": 10,
            "seq_len": 72,
            "pred_len": 36,
            "label_len": 36,
            "in_var": len(in_features),
            "time_var": len(time_features),
            "day_len": 144,
            "train_days": 172,
            "val_days": 24,
            "test_days": 49,
            "total_days": 245,
            "k_neighbor": 8,
            "feature_neighbor": 5,
            "freq": "d",
        })

    def __call__(self):
        return self.get_settings()
