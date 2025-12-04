import numpy as np


def preprocess_GEF(df, args):
    # 设置无效数据为 NaN
    df = df.bfill().ffill().fillna(0)
    df['predict_date'] = df['predict_date'].astype(str)

    df['Year'] = df['predict_date'].str[:4].astype(int)
    df['Month'] = df['predict_date'].str[4:6].astype(int)
    df['Day'] = df['predict_date'].str[6:8].astype(int)
    df['Hour'] = df['predict_date'].str[8:].astype(int)

    # 获取每列的最大值和最小值，进行归一化到 [-0.5, 0.5]
    df['Year'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min()) - 0.5
    df['Month'] = (df['Month'] - df['Month'].min()) / (df['Month'].max() - df['Month'].min()) - 0.5
    df['Day'] = (df['Day'] - df['Day'].min()) / (df['Day'].max() - df['Day'].min()) - 0.5
    df['Hour'] = (df['Hour'] - df['Hour'].min()) / (df['Hour'].max() - df['Hour'].min()) - 0.5

    df = df.drop(columns=['predict_date'])

    drop_cols = args.drop_cols
    df = df.drop(columns=drop_cols, errors='ignore')

    return df


def preprocess_WPF(df, args):
    nan_rows = df.isnull().any(axis=1)
    invalid_cond = (df['Patv'] < 0) | \
                   ((df['Patv'] == 0) & (df['Wspd'] > 2.5)) | \
                   ((df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89)) | \
                   ((df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) |
                    (df['Ndir'] > 720))

    # 设置无效数据为 NaN
    df.loc[invalid_cond, list(set(df.columns) - {'TurbID'})] = np.nan
    # df['mask'] = np.where(invalid_cond | nan_rows, 0, 1)
    df = df.bfill().ffill().fillna(0)

    df['Day'] = (df['Day'] - 1) / 143 - 0.5
    df['Hour'] = df['Tmstamp'].str.split(':').str[0].astype(int)
    df['Minute'] = df['Tmstamp'].str.split(':').str[1].astype(int)

    df['Hour'] = df['Hour'] / 23 - 0.5
    df['Minute'] = df['Minute'] / 59 - 0.5
    df = df.drop(columns=['Tmstamp'])

    df['Prtv'] = df['Prtv'].abs()
    pab_max = df[['Pab1', 'Pab2', 'Pab3']].max(axis=1)
    df.insert(loc=0, column='Pab_max', value=pab_max)

    drop_cols = args.drop_cols
    df = df.drop(columns=drop_cols, errors='ignore')
    return df


preprocess_dict = {
    'WPF': preprocess_WPF,
    'GEF': preprocess_GEF
}


def preprocess_provider(df, args):
    df = preprocess_dict[args.data](df, args)
    return df
