import numpy as np
import random

from TransGAT.configs.config_factory import config_factory
import time
import pandas as pd
import os
from TransGAT.exp.exp_forecasting import Exp_Long_Term_Forecast
from TransGAT.utils.preprocess import preprocess_provider

if __name__ == "__main__":
    fix_seed = 8888
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    WPF = 'WPF'
    GEF = 'GEF'
    settings = config_factory(WPF)
    start_time = end_time = time.time()
    df_raw = pd.read_csv(os.path.join(settings.data_path, settings.filename))

    capacity = settings.capacity
    total_size = capacity * settings.total_days * settings.day_len
    df_raw = df_raw.iloc[:total_size]
    df_raw = preprocess_provider(df_raw, settings)

    for (k, v) in [[72, 36], [144, 72], [144, 144]]:
        for name in ['AGSTNetMultiDecomp']:
            settings.input_flag = True  # 是否加入图神经
            settings.model = name
            settings.seq_len = k
            settings.pred_len = v
            settings.label_len = v

            for ii in range(settings.itr):
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dr{}_df{}_{}_{}'.format(
                    settings.task_name,
                    settings.model,
                    settings.data,
                    settings.features,
                    settings.seq_len,
                    settings.label_len,
                    settings.pred_len,
                    settings.d_model,
                    settings.n_heads,
                    settings.e_layers,
                    settings.d_layers,
                    settings.dropout,
                    settings.d_ff,
                    settings.des, ii)

                print(setting)
                exp = Exp_Long_Term_Forecast(settings)
                exp.train_and_val(df_raw, setting)
                exp.test(df_raw, setting)
