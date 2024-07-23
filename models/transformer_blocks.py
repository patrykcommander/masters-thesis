import torch
from torch import nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same")
        self.conv_3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.relu(x1)

        x2 = self.conv_3(x)
        x2 = self.relu(x2)

        x3 = self.conv_5(x)
        x3 = self.relu(x3)

        x4 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x4 = self.branch_pool(x4)
        x4 = self.relu(x4)

        return torch.cat([x1, x2, x3, x4], dim=1)
    
class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilationBlock, self).__init__()

        self.dil_1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, dilation=1, padding=2)
        self.dil_2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, dilation=4, padding=8)
        self.dil_3 = nn.Conv1d(in_channels, out_channels, kernel_size=5, dilation=8, padding=16)
        self.dil_4 = nn.Conv1d(in_channels, out_channels, kernel_size=5, dilation=16, padding=32)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.dil_1(x)
        x1 = self.relu(x1)

        x2 = self.dil_2(x)
        x2 = self.relu(x2)

        x3 = self.dil_3(x)
        x3 = self.relu(x3)

        x4 = self.dil_4(x)
        x4 = self.relu(x4)

        return torch.cat([x1, x2, x3, x4], dim=1)
    
class OneDCNNNorm(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(OneDCNNNorm, self).__init__()
        self.conv_1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv_1d(x)
        x = self.relu(x)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, nhead, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout)
        self.inception = InceptionBlock(in_channels=input_dim, out_channels=input_dim)
        #self.dilatation = DilationBlock(in_channels=input_dim, out_channels=input_dim)
        self.dropout = nn.Dropout(dropout)

        self.oneD_cnn_norm_1 = OneDCNNNorm(in_channels=input_dim*4, out_channels=input_dim, dropout=0.2)
        #self.oneD_cnn_norm_2 = OneDCNNNorm(in_channels=input_dim*4, out_channels=input_dim, dropout=0.2)

        self.linear = nn.Linear(input_dim, input_dim)
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)


    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src = src.permute(1,2,0)
        src2 = self.inception(src)
        src2 = self.oneD_cnn_norm_1(src2)
        #src2 = self.dilatation(src2)
        #src2 = self.oneD_cnn_norm_2(src2)
        src2 = src2.permute(2,0,1)
        src2 = self.linear(src2)
        src2 = src2.permute(1,2,0)

        src = src + src2
        src = src.permute(2,0,1)
        src = self.norm2(src)

        return src


    


