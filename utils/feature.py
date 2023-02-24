import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

def erase_onbase_who(df):
    # erase on_base playerid
    for col in ('on_3b', 'on_2b', 'on_1b'):
        df[col] = np.where(df[col].isna(), 0, 1) # 맞는지 체크 필요

def read_info(fname='config/full_feature.tsv'):
    '''
    read tsv file and return dataframe with selected columns
    '''

    # read config/full_feature.tsv
    info = pd.read_csv(
        fname, header=None, 
        delim_whitespace=True,
        names=['column_name','data_type']
    )
    
    for data in info.itertuples():
        if data.data_type == '-':
            print(f'{data.column_name} is not used')
            continue

        ctype = {
            'n' : 'numeric',
            'c' : 'categorical',
            'b' : 'binary',
            'cy' : 'label(category)'
        }[data.data_type]

        print(f'{data.column_name} is {ctype} type.')
    
    return info

def get_number_of_features(df, info):
    # breakpoint()
    for col in df.columns:
        col_type = info.loc[info.column_name==col,'data_type'].item()
        n_type = len(df[col].unique())
        print(col, col_type, n_type)
        # breakpoint()
        if n_type <= 30:
            print(df[col].unique())

def get_dummies(df, info):
    new_df_x = pd.DataFrame()
    new_df_y = pd.DataFrame()
    for col in tqdm(df.columns):
        if info[info.column_name==col].data_type.item() == 'c': # category
            col_one_hot = pd.get_dummies(df[col], prefix=col)
            new_df_x = pd.concat([new_df_x, col_one_hot], axis=1)
        if info[info.column_name==col].data_type.item() == 'b': # binary
            col_one_hot = pd.get_dummies(df[col], prefix=col)
            new_df_x = pd.concat([new_df_x, col_one_hot], axis=1)
        elif info[info.column_name==col].data_type.item() == 'n': # numeric
            new_df_x = pd.concat([new_df_x, df[col]], axis=1)
        elif info[info.column_name==col].data_type.item() == 'cy': # category label
            col_one_hot = pd.get_dummies(df[col], prefix=col)
            new_df_y = pd.concat([new_df_y, df[col]], axis=1)
        # breakpoint()

    breakpoint()

    return new_df_x, new_df_y


if __name__ == '__main__':
    df = pd.read_csv('data_csv/sorted-2015-to-2021(named,alpha)_v3.csv')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    erase_onbase_who(df)

    info = read_info()
    info_with_n_feautres = get_number_of_features(df, info)

    # for given features, make tensor
    new_df = get_dummies(df, info)