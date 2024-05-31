from torch import nn
import torch


class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.features_len = configs.features_len
        self.window_size = configs.window_size
        self.device = device
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.project = configs.project

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.final_out_channels * self.features_len, self.final_out_channels * self.features_len // self.project),
            nn.BatchNorm1d(self.final_out_channels * self.features_len // self.project),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * self.features_len // self.project, 2),
        )
        self.logits = nn.Linear(self.final_out_channels * self.features_len, 2)

    def forward(self, x_in):
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        # 1D CNN feature extraction
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        # Encoder
        hidden = x.permute(0, 2, 1)
        hidden = hidden.reshape(hidden.size(0), -1)
        logits = self.projection_head(hidden)
        # logits = self.logits(hidden)

        return logits

