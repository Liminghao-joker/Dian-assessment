import torch.nn as nn

# BasicBlock for ResNet-18
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # the first Residual Block
        # probably dimension not match
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # the second Residual Block
        # dimension keeps same
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # the first convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # the second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # to solve conv3-5 dimension mismatch
        if self.downsample is not None:
            identity = self.downsample(x)

        # residual connection
        out += identity
        out = self.relu(out) # activation function
        return out


class Bottleneck(nn.Module):
    expansion = 4
    # Bottleneck block for ResNet-50/101
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels

        # 1x1 convolution to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 convolution to restore dimensions
        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out