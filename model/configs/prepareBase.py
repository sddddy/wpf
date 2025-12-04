from types import SimpleNamespace


class PrepareBase:
    """基类，包含公共参数"""

    def __init__(self):
        self.common_settings = {
            "task_name": "long_term_forecast",
            "features": "MS",
            "model": "iTransformer",
            "des": "test",
            #
            "target": None,
            "in_features": None,
            "time_features": None,
            "drop_cols": None,
            #
            "capacity": None,
            "seq_len": None,
            "pred_len": None,
            "label_len": None,
            "in_var": None,
            "time_var": None,
            "out_var": 1,
            "day_len": None,
            "train_days": None,
            "val_days": None,
            "test_days": None,
            "total_days": None,
            "k_neighbor": None,
            "feature_neighbor": None,
            #
            "num_workers": 0,
            "train_epochs": 30,
            "batch_size": 16,
            "patience": 2,
            "itr": 3,
            "delta": 0,
            "verbose": True,
            "lr": 1e-4,
            "lradj": "type1",
            "use_gpu": True,
            "use_graph": True,
            "gpu": 0,
            "use_multi_gpu": False,
            "is_debug": True,
            # 是否加入图神经
            "input_flag": True,
            "log_per_steps": 20,
            "loss": "MaskedLoss",
            #
            "framework": "pytorch",
            "checkpoints": "checkpoints",
            #
            "embedding_dims": 8,
            "d_model": 512,
            "n_heads": 8,
            "contamination": 0.10,
            "dropout": 0.05,
            "e_layers": 2,
            "d_layers": 1,
            "factor": 1,
            "embed": "timeF",
            "activation": "gelu",
            "d_ff": 2048,
            "distil": "store_false",
            "patch_len": 16,
            "use_norm": 1,
        }

    def get_settings(self):
        """返回配置"""
        return SimpleNamespace(**self.common_settings)
