# %%

import pandas as pd
from darts.metrics import mape, rmse, smape
from darts.utils.likelihood_models import QuantileRegression
from matplotlib import pyplot as plt

from src.features import build_features as bf
from darts import TimeSeries
from src.models.darts_implementation import SolarFlare
from darts.dataprocessing.transformers import Scaler
# %%
path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:8000]
df = df.drop(columns="Minute")
df = bf.build_features(df)
memory = 24
horizon = 5
batch = 128

df.insert(0, 'GHI', df.pop('GHI'))

# Split data
target_ts = TimeSeries.from_series(df["GHI"])
target_train, target_val = target_ts.split_after(0.8)
target_val, target_test = target_val.split_after(0.5)

past_covariates = ["Tamb", "DewPoint", "Pw", "Cloudopacity", "Pressure", "AlbedoDaily", "WindVel", "DHI", "DNI", "EBH", "WindDir_cos", "WindDir_sin"]
future_covariates = ["Year_sin", "Year_cos", "Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos", "Azimuth_sin", "Azimuth_cos", "Zenith_sin", "Zenith"]

past_covar_ts = TimeSeries.from_dataframe(df[df.columns.to_list()[1:]])
past_covar_train, past_covar_val = past_covar_ts.split_after(0.8)
past_covar_val, past_covar_test = past_covar_val.split_after(0.5)

# Scale data
target_ts_scaler = Scaler()
target_train = target_ts_scaler.fit_transform(target_train)
target_val = target_ts_scaler.transform(target_val)
target_test = target_ts_scaler.transform(target_test)
series_transformed = target_ts_scaler.transform(target_ts)

covar_ts_scaler = Scaler()
covar_train = covar_ts_scaler.fit_transform(past_covar_train)
covar_val = covar_ts_scaler.transform(past_covar_val)
covar_test = covar_ts_scaler.transform(past_covar_test)

# %%

model_air = SolarFlare(input_chunk_length=memory,
                       kernel_size=2,
                       dilation_base=2,
                       output_chunk_length=horizon,
                       likelihood=QuantileRegression(quantiles=[0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]),
                       batch_size=batch,
                       hidden_size=56,
                       nr_epochs_val_period=1,
                       log_tensorboard=True)


# %%

# model_air.fit(
#     series=target_train,
#     past_covariates=covar_train,
#     val_series=target_val,
#     val_past_covariates=covar_val,
#     verbose=True)



# %%


model = SolarFlare.load_model("solarflaremodel.pth.tar")

def eval_model(model, n, actual_series, val_series, num_samples=600):
    pred_series = model.predict(n=n, num_samples=num_samples, series=target_test)
    # plot actual series
    plt.figure(figsize=(9,6))
    actual_series[pred_series.end_time() - pred_series.freq * 12: pred_series.end_time()].plot(label="actual")
    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=0.01, high_quantile=0.99, label=f"{int(0.01 * 100)}-{int(0.99 * 100)}th percentiles"
    )
    pred_series.plot(low_quantile=0.1, high_quantile=0.9, label=f"{int(0.1 * 100)}-{int(0.9 * 100)}th percentiles")
    plt.title("MAPE: {:.2f}%".format(smape(val_series, pred_series)))
    plt.legend()

eval_model(model, 4, series_transformed[:-27], target_val[:-27])
plt.show()