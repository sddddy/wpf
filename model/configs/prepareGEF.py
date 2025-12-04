from TransGAT.configs.prepareBase import PrepareBase

in_features = ['u', 'v', 'ws', 'wd', 'wp']
drop_cols = ['date', 'hors']
time_features = ['Year', 'Month', 'Day', 'Hour']


class PrepareGEF(PrepareBase):
    """GEF 数据集的配置"""
    def __init__(self):
        super().__init__()
        self.common_settings.update({
            "data": "GEF",
            "data_path": "../data/GEF",
            "filename": "GEF2012.csv",
            "target": "wp",
            "in_features": in_features,
            "time_features": time_features,
            "drop_cols": drop_cols,
            "capacity": 7,
            "seq_len": 48,
            "pred_len": 12,
            "label_len": 24,
            "in_var": len(in_features),
            "time_var": len(time_features),
            "day_len": 48,
            "train_days": 1529,
            "val_days": 218,
            "test_days": 437,
            "total_days": 2184,
            "k_neighbor": 3,
            "feature_neighbor": 2,
            "freq": "h",
        })

    def __call__(self):
        return self.get_settings()