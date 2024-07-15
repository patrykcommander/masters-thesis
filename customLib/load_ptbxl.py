import os, ast
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

def load_ptbxl(path: str, sampling_rate=500, files_num=""):
    ptxbl_csv = os.path.join(path, 'ptbxl_database.csv')
    Y = pd.read_csv(ptxbl_csv, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # drop the rows with labels other than 'NORM' -> normal ECG
    Y = Y[Y['scp_codes'].apply(lambda x: 'NORM' in x)]

    if sampling_rate == 500:
        file_names = Y['filename_hr'].tolist()
    else:
        file_names = Y['filename_lr'].tolist()

    if files_num != 'all':
        file_names = file_names[:5]

    path += "\\"
    data = [(wfdb.rdsamp(path+f, channels=[1]), f.split('/')[-1],) for f in tqdm(file_names)]

    data = [signal.reshape((-1, )) for (signal, meta), f in data]
    return data