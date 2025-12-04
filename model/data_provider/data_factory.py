from TransGAT.data_provider.data_loader_weight import Dataset_Common
from torch.utils.data import DataLoader

data_dict = {
    'WPF': Dataset_Common,
    'GEF': Dataset_Common
}


def data_provider(args, data, setting, flag):
    Data = data_dict[args.data]
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True

    data_set = Data(
        data=data,
        args=args,
        flag=flag,
        input_len=args.seq_len,
        output_len=args.pred_len,
        target=args.target,
        in_features=args.in_features,
        day_len=args.day_len,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        total_days=args.total_days,
        capacity=args.capacity,
        k_neighbor=args.k_neighbor,
        feature_neighbor=args.feature_neighbor,
        setting=setting
    )
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers
    )

    return data_set, data_loader
