import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)

    def internal_states(self, x):
        states = []
        identity = self.shortcut(x)
        if isinstance(self.shortcut, nn.Sequential):
            states.append(identity)
        out = self.conv1(x); states.append(out)
        out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); states.append(out)
        out = self.bn2(out); out = out + identity
        out = self.relu(out)
        return out, states

class LightweightResNet18(nn.Module):
    """Team's CIFAR-sized ResNet-18-style model; defaults match their code."""
    def __init__(self, num_classes=10, channels=(10, 20, 40, 80)):
        super().__init__()
        self.channels = tuple(channels)
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(channels[0], channels[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def predict_internal_states(self, x):
        states = []
        x = self.conv1(x); states.append(x)
        x = self.bn1(x); x = self.relu(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x = block(x)
                states.append(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        x = self.fc(x); states.append(x)
        return states

Net = LightweightResNet18

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
