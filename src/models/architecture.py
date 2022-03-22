import torch.nn as nn


class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size):
        super(BayesianLSTM, self).__init__()
