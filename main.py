import pandas as pd
import darts
from darts.dataprocessing.transformers import Scaler, Mapper, InvertibleMapper
from darts.models import TCNModel, TFTModel, LinearRegressionModel

makeSolarradiationData = False

def main():
    df = pd.read_pickle("data/raw/irradiance_data_NL_2007_2022.pkl")
    target = df['GHI']
    df = df.drop(columns=["DHI", "DNI", "EBH", "GHI", "Month", "Day", "Hour"])
    target_ts = darts.TimeSeries.from_series(target)
    past_covariances_ts = [darts.TimeSeries.from_series(df[x]) for x in df.columns]
    first_covs = past_covariances_ts[0]
    last_covs = past_covariances_ts[1:]
    for cov in last_covs:
        first_covs = first_covs.concatenate(cov, axis=1)

    first_covs = past_covariances_ts[0]
    last_covs = past_covariances_ts[1:]
    for cov in last_covs:
        first_covs = first_covs.concatenate(cov, axis=1)

    train, val = target_ts.split_after(0.8)
    past_cov_train, past_cov_val = first_covs.split_after(0.8)
    target_scaler = Scaler()
    target_scaler.fit(train)
    rescaled_target_train = target_scaler.transform(train)
    rescaled_target_val = target_scaler.transform(val)
    past_covs_scaler = Scaler()
    past_covs_scaler.fit(past_cov_train)
    rescaled_past_covs_train = past_covs_scaler.transform(past_cov_train)
    rescaled_past_covs_val = past_covs_scaler.transform(past_cov_val)
    model2 = LinearRegressionModel(lags=1, lags_past_covariates=48)
    model2.fit(series=rescaled_target_train, past_covariates=rescaled_past_covs_train)
    model1 = TCNModel(input_chunk_length=48, output_chunk_length=4, n_epochs=10)

    model1.fit(series=rescaled_target_train, val_series=rescaled_target_val, past_covariates=rescaled_past_covs_train, val_past_covariates=rescaled_past_covs_val,
              verbose=True)

    print("hi")


if __name__ == '__main__':
    main()
