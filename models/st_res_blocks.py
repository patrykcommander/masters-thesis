import torch
from torch import nn

class ParallelConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ParallelConv, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same")
        self.conv_2 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x)

        x_1 = self.relu(x_1)
        x_2 = self.relu(x_2)

        return x_1 + x_2

class ResPath(nn.Module):
    def __init__(self, in_channels, out_channels, num_parallel_convs):
        super(ResPath, self).__init__()
        self.parallel_convs = nn.ModuleList([ParallelConv(in_channels, out_channels) for _ in range(num_parallel_convs)])
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.parallel_convs:
            x = layer(x)
            x = self.relu(x)
        return x

class ST(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.25):
        super(ST, self).__init__()

        if int(out_channels/2) < 1:
            hidden_size = 1
        else:
            hidden_size = int(out_channels/2)

        self.bi_lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.input_conv_3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same", )
        self.input_conv_7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding="same")

        self.conv_kernel_3 = nn.ModuleList([nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same") for _ in range(2)])
        self.conv_kernel_7 = nn.ModuleList([nn.Conv1d(out_channels, out_channels, kernel_size=7, padding="same") for _ in range(1)])

        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv1d expects input of shape (batch_size, channels, sequence_length)

        x_1 = self.input_conv_3(x)
        x_1 = self.relu(x_1)

        x_2 = self.input_conv_7(x)
        x_2 = self.relu(x_2)

        for layer in self.conv_kernel_3:
            x_1 = layer(x_1)
            x_1 = self.relu(x_1)

        for layer in self.conv_kernel_7:
            x_2 = layer(x_2)
            x_2 = self.relu(x_2)

        x = x.permute(0, 2, 1)
        x_3, _ = self.bi_lstm(x)
        x_3 = self.relu(x_3)
        
        x_3 = x_3.permute(0, 2, 1)

        f = x_1 + x_2 + x_3
        p = self.max_pool(f)

        return f, self.relu(p)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dropout=0.3):
        super(UpSampleBlock, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=2)
        self.st_block = ST(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, conv_features):
        x = self.conv1d_transpose(x)
        x = torch.cat([x, conv_features], dim=1)
        x = self.dropout(x)
        return self.st_block(x)