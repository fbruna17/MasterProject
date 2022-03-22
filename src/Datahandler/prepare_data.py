import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def split_data(df):
    split_train = int(len(df) * 0.8)
    split_val = split_train + int(len(df) * 0.1)

    train = df[:split_train]
    val = df[split_train:split_val]
    test = df[split_val:]
    return train, val, test