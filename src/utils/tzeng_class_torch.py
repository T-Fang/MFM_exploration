import torch.nn as nn


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, bottleneck_channels):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Linear(in_channels, bottleneck_channels)
        self.restore = nn.Linear(bottleneck_channels, out_channels)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.restore(x)
        return x
