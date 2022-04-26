from numpy import sin, cos, pi


def generate_leads(df, columns, number_of_leads: int = 5):
    for col in columns:
        for i in range(1, number_of_leads):
            df[f"{col}t{i}"] = df[col].shift(-i)
    return df


def encode_cycle(df, columns):
    for col in columns:
        df[f"{col}_sin"] = sin(2 * pi * df[col] / df[col].max())
        df[f"{col}_cos"] = cos(2 * pi * df[col] / df[col].max())
    return df


def build_features(df):
    cyclic_features = ['Month','Hour', 'Year', 'Day', 'WindDir', 'Zenith', 'Azimuth']
    #number_of_leads = 4
    #lead_feature = ['Azimuth_sin', 'Azimuth_cos', 'Zenith_sin', 'Zenith_cos']
    df = encode_cycle(df, cyclic_features)

    #df = generate_leads(df, lead_feature, number_of_leads=number_of_leads)
    df = df.drop(columns=cyclic_features)
    return df
