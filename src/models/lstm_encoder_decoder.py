import pandas as pd
from numpy import stack, mean
from src.Datahandler.scaling import scale_train
from src.models.architecture import LSTMEncoderDecoder, EncoderPrediction, PredictionNet
from src.Datahandler.SequenceDataset import make_torch_dataset, SubSequenceDataset, SequenceDataset
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt


train_new_model = False
path = '../../data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:30000]
columns = ['Year', 'Month', 'Day', 'Hour', 'GHI']
external_features = ['Tamb', 'Azimuth', 'Cloudopacity',
                     'DewPoint', 'Pw', 'Pressure', 'WindDir',
                     'WindVel', 'AlbedoDaily', 'Zenith']
training_df = df[columns]
external_features_df = df[external_features]

memory = 48
horizon = 4
encoder_output = 1
hidden_size_1 = 12
hidden_size_2 = 8
encoder_input = len(df.columns)
num_layer = 1
batch = 192
target = "GHI"
learning_rate = 0.001
EPOCHS = 20
decoder_input = encoder_output
dropout = 0.4
if(train_new_model):
    # SETTING TARGET VALUE TO FIRST COLUMN
    df.insert(0, 'GHI', df.pop("GHI"))

    df = scale_train(df)
    data = DataLoader(SequenceDataset(dataframe=df, memory=memory, horizon=horizon, features=columns, target=target),
                      batch_size=batch, drop_last=True)

    model = LSTMEncoderDecoder(
        encoder_input=encoder_input,
        encoder_output=encoder_output,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        decoder_input=decoder_input,
        num_layers=num_layer,
        horizon=horizon,
        dropout=dropout,
        decoder_hs_1=12, decoder_hs_2=24, decoder_output=12)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(EPOCHS):
        loss_1 = []
        for x1, y in data:
            y_hat = model(x1)
            optimizer.zero_grad()
            loss = torch.sqrt(mse_loss(y_hat, y))
            loss.backward()
            optimizer.step()
            loss_1.append(loss.cpu().detach().numpy())
        print(mean(stack(loss_1)))
        if epoch % 5 == 0:
            plt.plot(y[::4].flatten())
            plt.plot(y_hat[::4].flatten().detach())
            plt.show()

    torch.save(model, 'pretrained.pkl')

pretrained = torch.load('pretrained.pkl')
encoder_pretrained = pretrained.encoder
params = {
    'n_extracted_features': encoder_output,
    'n_external_features': len(df.columns),
    'predict_hidden_1': 288,
    'predict_hidden_2': 64,
    'n_output_steps': horizon,
}

encoder_pred = EncoderPrediction(encoder_pretrained)

prediction_net = PredictionNet(params, dropout)

df = scale_train(df)
training_df = df[columns]
external_features_df = df[external_features]

#df_pretraining = scale_train(training_df)


df1 = training_df
ghi = df[['GHI']]

encoder_input_data = SequenceDataset(df1, target='GHI', features=df1.columns, memory=memory, horizon=horizon)
univariate_external_data = SequenceDataset(df1, target='Hour', features=df1.columns, memory=memory, horizon=horizon)
prediction_net_data = SequenceDataset(external_features_df, target='Tamb', features=external_features_df.columns, memory=memory, horizon=horizon)

encoder_input_dataloader = DataLoader(encoder_input_data, batch_size=batch)
univariate_external_dataloader = DataLoader(univariate_external_data, batch_size=batch)
prediction_net_data_dataloader = DataLoader(prediction_net_data, batch_size=batch)


prediction_net.train()
optimizer = torch.optim.Adam(params=prediction_net.parameters(), lr=learning_rate)
for epoch in range(EPOCHS):
    loss_1 = []
    for (x, y), (external, _), (pred_data, _) in zip(encoder_input_dataloader, univariate_external_dataloader, prediction_net_data_dataloader):
        encoder_output = encoder_pred((x, external))
        y_hat = prediction_net(pred_data, encoder_output)
        optimizer.zero_grad()
        loss = torch.sqrt(mse_loss(y_hat, y))
        loss.backward()
        optimizer.step()
        loss_1.append(loss.cpu().detach().numpy())
    print(mean(stack(loss_1)))
    if epoch % 5 == 0:
        plt.plot(y[::4].flatten())
        plt.plot(y_hat[::4].flatten().detach())
        plt.show()

print("hi")



