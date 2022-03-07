from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.compose import ColumnwiseTransformer
from sktime.transformations.series.detrend import Detrender


class TimeSeriesTransformer:
    def __init__(self,
                 data: pd.DataFrame,
                 target_col: str = "GHI",
                 ):
        self.data = data
        self.standardizer = None
        self.target_col = target_col
        self.ys = data[target_col]
        self.target_mean = self.ys.mean()
        self.target_std = self.ys.std(ddof=0)


    def standardizer_fit(self):
        self.standardizer = StandardScaler()
        self.standardizer.fit(self.data)

    def standardizer_transform(self, data):
        return pd.DataFrame(
            data=self.standardizer.transform(data),
            columns=self.standardizer.get_feature_names_out())

    def inverse_transform_target(self, y, y_hat):
        return (y.numpy() * self.target_std) + self.target_mean, (y_hat.numpy() * self.target_std) + self.target_mean

