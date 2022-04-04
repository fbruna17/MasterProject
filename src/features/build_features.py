from pandas import DataFrame
from numpy import sin, cos, pi


def generate_leads(df, columns, number_of_leads: int = 5):
    for col in columns:
        for i in range(1, number_of_leads):
            df[f"{col} t+{i}"] = df[col].shift(-i)
    return df


def encode_cycle(df, columns):
    for col in columns:
        df[f"{col}_sin"] = sin(2 * pi * df[col] / df[col].max())
        df[f"{col}_cos"] = cos(2 * pi * df[col] / df[col].max())
        df.drop(columns=col, inplace=True)
    return df


def build_features(df):
    lead_feature = ['Azimuth', 'Zenith']
    cyclic_features = ['Hour', 'Year', 'Day', 'WindDir', 'Zenith', 'Azimuth']
    number_of_leads = 4

    df = generate_leads(df, lead_feature, number_of_leads=number_of_leads)
    df = encode_cycle(df, cyclic_features)
    return df
