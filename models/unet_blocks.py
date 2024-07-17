import torch
from torch import nn, cat
from torch.nn import Conv1d, ConvTranspose1d, MaxPool1d, Dropout, ReLU, Sequential, Sigmoid

# data shape [batch_size, channels, samples]

class Conv1DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size):
    super(Conv1DBlock, self).__init__()
    self.conv1d = Conv1d(in_channels, out_channels, kernel_size, padding="same")
    self.relu = ReLU(inplace=True)

  def forward(self, x):
    x = self.conv1d(x)
    x = self.relu(x)
    return x

class DoubleConv1DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size):
    super(DoubleConv1DBlock, self).__init__()
    self.double_conv = Sequential(
      Conv1DBlock(in_channels, out_channels, kernel_size),
      Conv1DBlock(out_channels, out_channels, kernel_size)
    )

  def forward(self, x):
    return self.double_conv(x)


class DownSample1DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3):
    super(DownSample1DBlock, self).__init__()
    self.double_conv = DoubleConv1DBlock(in_channels, out_channels, kernel_size)
    self.maxpool_1d = MaxPool1d(kernel_size=2)
    self.dropout = Dropout(dropout)

  def forward(self, x):
    x = self.double_conv(x)
    pool = self.maxpool_1d(x)
    return x, self.dropout(pool)


class UpSample1DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3):
    super(UpSample1DBlock, self).__init__()
    self.conv1d_transpose = ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
    self.double_conv = DoubleConv1DBlock(in_channels, out_channels, kernel_size)
    self.dropout = Dropout(dropout)

  def forward(self, x, conv_features):
    x = self.conv1d_transpose(x)
    x = cat([x, conv_features], dim=1)
    x = self.dropout(x)
    return self.double_conv(x)