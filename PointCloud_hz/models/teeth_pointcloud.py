import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models


class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # x = self.model.avgpool(x)

        # x = x.view(x.size(0), x.size(1))
        return x

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class Model(nn.Module):
    def __init__(self, embed_dim=64, bias=False, activation='relu'):
        super(Model, self).__init__()
        self.embedding = ConvBNReLU1D(9, embed_dim, bias=bias, activation=activation)

    def forward(self, x):
        x = self.embedding(x)

        return x

if __name__ == '__main__':
    # model = base_resnet()
    model = Model()
    vectors = torch.rand(16, 4096, 3, 3)
    normals = torch.rand(16, 4096, 3, 1)
    vectors = vectors.reshape(16, 4096, -1).permute(0, 2, 1)
    x = model(vectors)
    print(1)