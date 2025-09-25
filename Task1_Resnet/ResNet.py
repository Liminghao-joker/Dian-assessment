import torch
import torch.nn as nn
from Block import Bottleneck, BasicBlock

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, ):
        super(ResNet, self).__init__()
        """
        after conv1 layer
        input:(3, 224, 224)
        output:(64, 56, 56)
        """
        self.in_channels = 64
        # Cov1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        """
        after residual layers
        input:(64, 56, 56)
        output:(512, 7, 7)
        """
        # residual layers
        self.layer1 = self._make_layer(block, 64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # only focus on the first block
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # the rest blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        # conv1 layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        # residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # fully connected layer
        x = self.fc(x)

        return x

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


# simple test
def test_resnet():
    model18 = resnet18(num_classes=10)
    model50 = resnet50(num_classes=10)

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out18_eval = model18(x)
        out50_eval = model50(x)

        print(f"input shape:{x.shape}")
        print(f"output shape for ResNet-18:{out18_eval.shape}")
        print(f"output shape for ResNet-50:{out50_eval.shape}")
        print(f"Parameter numbers for ResNet-18:{sum(p.numel() for p in model18.parameters())}")
        print(f"Parameter numbers for ResNet-50:{sum(p.numel() for p in model50.parameters())}")

    model18.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model18.parameters(), lr=0.01)

    optimizer.zero_grad()
    output18 = model18(x)
    loss = criterion(output18, torch.tensor([1, 0]))
    print("loss.requires_grad:", loss.requires_grad)
    loss.backward()
    optimizer.step()

    print("test pass!")


if __name__ == "__main__":
    test_resnet()
