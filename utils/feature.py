import numpy as np
import pandas as pd
from collections import Counter

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

if __name__ == '__main__':
    info = read_info()