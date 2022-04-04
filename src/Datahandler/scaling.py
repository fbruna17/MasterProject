import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class TimeSeriesTransformer:
    def __init__(self,
                 target_col: str = "GHI",
                 ):
        self.standardizer = None
        self.target_col = target_col
        self.ys = None
        self.target_mean = None
        self.target_std = None

    def standardizer_fit(self, data):
        ys = data[self.target_col]
        self.target_mean = ys.mean()
        self.target_std = ys.std(ddof=0)
        self.standardizer = StandardScaler()
        self.standardizer.fit(data)

    def standardizer_transform(self, data):
        return pd.DataFrame(
            data=self.standardizer.transform(data),
            columns=self.standardizer.get_feature_names_out())

    def inverse_transform_target(self, y, y_hat):
        return torch.tensor((y.numpy() * self.target_std) + self.target_mean).float(), \
               torch.tensor((y_hat.numpy() * self.target_std) + self.target_mean).float()


def scale_data(train, val, test):
    transformer = TimeSeriesTransformer()
    transformer.standardizer_fit(train)
    train = transformer.standardizer_transform(train)
    val = transformer.standardizer_transform(val)
    test = transformer.standardizer_transform(test)
    return train, val, test, transformer

def scale_train(train):
    transformer = TimeSeriesTransformer()
    transformer.standardizer_fit(train)
    train = transformer.standardizer_transform(train)
    return train