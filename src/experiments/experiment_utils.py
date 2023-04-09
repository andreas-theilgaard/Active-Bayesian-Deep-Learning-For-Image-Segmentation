import pandas as pd
import os
import numpy as np


def post_process_ensemble(df):
    cols = df.columns
    for col in cols:
        df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
    return df


def post_process(df):
    cols = df.columns
    for col in cols:
        df[col] = df[col].apply(lambda x: np.array(x).flatten() if isinstance(x, list) else x)
    return df


def arrayify_results(data_to_store, save_path):
    file_exists = os.path.isfile(f"{save_path}.json")
    if file_exists:
        stored_res = pd.read_json(f"{save_path}.json")
        # assert len([x for x in stored_res.columns if x not in list(data_to_store.keys())])==0
        stored_res.loc[len(stored_res)] = data_to_store.values()  # append new row
        stored_res = post_process(stored_res)
        stored_res.to_json(f"{save_path}.json")  # update file
        return stored_res
    else:
        # res = pd.DataFrame(columns=list(data_to_store.keys()),data=[[list(data_to_store.values())]])
        res = pd.DataFrame(data_to_store)
        res = post_process(res)
        res.to_json(f"{save_path}.json")
        return res
