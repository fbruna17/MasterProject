from pandas import Series, DataFrame
from numpy import maximum, random

def generate_noise(series: Series):
    min_value = series.min()
    max_value = series.max()
    noise = series.apply(lambda x: x - 0.015 * ((min_value - max_value) * random.random() + min_value))
    noise = maximum(noise, min_value)
    return noise


def generate_leads(df: DataFrame, series: Series, col_name, number_of_leads: int = 5):
    for i in range(1, number_of_leads):
        df[f"{col_name} t+{i}"] = series.shift(-i)

def clean_data(df):
    weather_copy = df[['Cloudopacity', 'DewPoint', 'Pressure', 'WindDir', 'WindVel', 'Pw', 'Tamb']]
    df = df.loc[(df.Hour < 22) & (df.Hour > 5)]
    return df
