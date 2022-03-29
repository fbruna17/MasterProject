import pandas as pd
from numpy import stack, mean
from src.Datahandler.scaling import scale_train
from src.models.architecture import LSTMEncoderDecoder, Predict
from src.Datahandler.SequenceDataset import make_torch_dataset, SubSequenceDataset, SequenceDataset
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt

path = '../../data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:40000]
#
# columns = ['GHI']
memory = 32
horizon = 4
encoder_output = 6
hidden_size_1 = 32
hidden_size_2 = 8
encoder_input = memory
num_layer = 1
batch = 128
target = "GHI"
learning_rate = 0.005
EPOCHS = 50
train_new_model = False
decoder_input = horizon + encoder_output
dropout = 0.2

# df = df[columns]
# df = scale_train(df)
# data = DataLoader(SequenceDataset(dataframe=df, memory=memory, horizon=horizon, features=columns, target=target),
#                   batch_size=batch, drop_last=True)
#
# model = LSTMEncoderDecoder(
#     encoder_input=encoder_input,
#     encoder_output=encoder_output,
#     hidden_size_1=hidden_size_1,
#     hidden_size_2=hidden_size_2,
#     decoder_input=decoder_input,
#     num_layers=num_layer,
#     horizon=horizon,
#     dropout=dropout,
#     decoder_hs_1=12, decoder_hs_2=6, decoder_output=4)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
#
# model.train()
# for epoch in range(EPOCHS):
#     loss_1 = []
#     for x1, y in data:
#         y_hat = model(x1.permute(0, 2, 1))
#         optimizer.zero_grad()
#         loss = torch.sqrt(mse_loss(y_hat.squeeze(), y))
#         loss.backward()
#         optimizer.step()
#         loss_1.append(loss.cpu().detach().numpy())
#     print(mean(stack(loss_1)))
#     if epoch % 5 == 0:
#         plt.plot(y[::4].flatten())
#         plt.plot(y_hat[::4].flatten().detach())
#         plt.show()
#
# torch.save(model, 'pretrained.pkl')

pretrained = torch.load('pretrained.pkl')

params = {
    'n_extracted_features': encoder_output,
    'n_external_features': len(df.columns) - 1,
    'predict_hidden_1': 128,
    'predict_hidden_2': 64,
    'n_output_steps': horizon,
}

pred = Predict(params, dropout, pretrained.encoder)

df1 = df.drop(columns='GHI')
ghi = df[['GHI']]

external_data = SequenceDataset(df1, target='Tamb', features=df1.columns, memory=memory, horizon=horizon)
ghi_data = SequenceDataset(ghi, target='GHI', features=['GHI'], memory=memory, horizon=horizon)

external_dataloader = DataLoader(external_data, batch_size=batch)
ghi_dataloader = DataLoader(ghi_data, batch_size=batch)

model = pred
model.train()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
for epoch in range(EPOCHS):
    loss_1 = []
    for (x, y), (external, _) in zip(ghi_dataloader, external_dataloader):
        y_hat = model((x, external))
        optimizer.zero_grad()
        loss = torch.sqrt(mse_loss(y_hat.squeeze(), y))
        loss.backward()
        optimizer.step()
        loss_1.append(loss.cpu().detach().numpy())
    print(mean(stack(loss_1)))
    if epoch % 5 == 0:
        plt.plot(y[::4].flatten())
        plt.plot(y_hat[::4].flatten().detach())
        plt.show()
