import pandas as pd


def clean_data(filepath: str):
    data = load_data(filepath)
    data = remove_columns(data)
    data = to_datetime(data)
    data = clean_data_types(data)
    data = set_index(data)
    data.to_pickle("C:/Users/Bruno/Documents/GitHub/MasterProject/data/processed/cleaned_data.pkl")
    return data


def load_data(filepath: str):
    if 'xlsx' in filepath:
        return pd.read_excel(filepath)
    elif 'csv' in filepath:
        return pd.read_csv(filepath)
    else:
        return pd.read_pickle(filepath)


def remove_columns(data: pd.DataFrame):
    data = data[["HourDK", "SpotPriceDKK"]]
    return data


def to_datetime(data: pd.DataFrame):
    data['HourDK'] = pd.to_datetime(data['HourDK'])
    return data


def clean_data_types(data):
    data['SpotPriceDKK'] = data['SpotPriceDKK'].astype(int)
    return data


def set_index(data: pd.DataFrame):
    data = data.set_index('HourDK')
    return data
