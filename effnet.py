import torch
import torch.nn as nn
from math import ceil

base_model = [
    # expand ratio, channels, repeats (layers), stride (down-sample), kernel size (conv)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # version : (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2), # alpha, beta, gamma (e.g. depth = alpha ** phi)
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups = 1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups = groups, # if group = 1: normal conv, if group = in_channels: depthwise conv
            bias = False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.99)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C*H*W -> C*1*1 (squeezing)
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)
    
class InvertedResidualBlock(nn.Module): #MBConv Block
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio,
                 reduction = 4, # squeeze excitation (down-sampling)
                 survival_prob = 0.8 # stochastic depth
                 ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1 # don't use skip connections when we down-sample
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=1, stride=1, padding=0,
            )

        self.conv = nn.Sequential(
            CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias = False),
            nn.BatchNorm2d(out_channels, momentum = 0.99),
        )

    def stochastic_depth(self, x):
        if not self.training: # don't prune layers during testing
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device = x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs # adds the input of the earlier layer (if dims match)
        else:
            return self.conv(x)
        
class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        depth_factor, width_factor, dropout = self.get_factors(version) # depth, width and drop out rate
        last_channels = ceil(1280 * width_factor) # last layer's channels
        self.pool = nn.AdaptiveAvgPool2d(1) # last layer pooling
        self.features = self.create_features(depth_factor, width_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channels, num_classes)
        )

    def get_factors(self, version, alpha = 1.2, beta = 1.1): # following the paper
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return depth_factor, width_factor, drop_rate

    def create_features(self, depth_factor, width_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride = 2, padding = 1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4) # make the output channels always divisible by 4, our reduction factor
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride = stride if layer == 0 else 1, # down sample only for the first layer of each stage
                        kernel_size=kernel_size,
                        padding = kernel_size//2, # floor division: if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )

                in_channels = out_channels # the no. input channels for next layer is no. output channels at current layer

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1)) # flatten before sending to linear layer