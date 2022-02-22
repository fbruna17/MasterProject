import pandas as pd
from datetime import datetime


def clean_data(filepath: str):
    data = load_data(filepath)
    return data


def load_data(filepath: str):
    if 'xlsx' in filepath:
        return pd.read_excel(filepath)
    elif 'csv' in filepath:
        return pd.read_csv(filepath)
    else:
        return pd.read_pickle(filepath)
