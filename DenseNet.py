import torch
from torch import nn


def conv(in_channels: int, out_channels: int, kernel_size: int, stride: int):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=1
        )  # in_channels, out_channels, kernel_size
    )


def DenseLayer(in_channels: int, growth_rate: int):
    return nn.Sequential(
        conv(in_channels, growth_rate * 4, 1, 1),
        conv(growth_rate * 4, growth_rate, 3, 1),
    )

def TransitionLayer(in_channels: int, growth_rate: int):
    return nn.Sequential(
        conv(in_channels=in_channels, out_channels=growth_rate, kernel_size=1, stride=1),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def ClassificationLayer(in_channel: int, out_channels: int):
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=7, stride=7, padding=0),
        nn.Linear(in_features=in_channel, out_features=out_channels, bias=True),
        nn.Softmax(dim=0)
    )

class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, lay_num: int):
        super(DenseBlock, self).__init__()
        self.lay_num = lay_num
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(self.lay_num)]
        )

    def forward(self, feature):
        # print(self.layers)
        for i in range(self.lay_num):
            feature = torch.cat([feature, self.layers[i](feature)], dim = 1)
        return feature

class DenseNet(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, *blocks):
        super(DenseNet, self).__init__()
        self.convolution = conv(in_channels, growth_rate * 2, kernel_size=7, stride=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        self.dense_block_1 = DenseBlock(growth_rate * 2, growth_rate, blocks[0])
        self.transition_layer_1 = TransitionLayer(blocks[0] * growth_rate + 2*growth_rate, growth_rate)
        for i in range(1, len(blocks)):
            self.add_module('dense_block_' + str(i + 1), DenseBlock(growth_rate, growth_rate, blocks[i]))
            self.add_module('transition_layer_' + str(i + 1),
                            TransitionLayer(blocks[i] * growth_rate + growth_rate, growth_rate))
        self.classification_layer = ClassificationLayer(blocks[-1] * growth_rate + growth_rate, 1000)

    def forward(self, img):
        feature = self.convolution(img)
        print(feature.size())
        feature = self.pooling(feature)
        print(feature.size())
        feature = self.dense_block_1(feature)
        print(feature.size())
        feature = self.transition_layer_1(feature)
        print(feature.size())
        feature = self.dense_block_2(feature)
        print(feature.size())
        feature = self.transition_layer_2(feature)
        print(feature.size())
        feature = self.dense_block_3(feature)
        print(feature.size())
        feature = self.transition_layer_3(feature)
        print(feature.size())
        feature = self.dense_block_4(feature)
        print(feature.size())
        output = self.classification_layer[0](feature)
        output = output.view(output.size(0), -1)
        output = self.classification_layer[1:](output)
        print('output', output.size(),output)
        return output
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DenseNet(3, 32, 6, 12, 24, 16).to(device)
    # net = DenseBlock(64, 32, 6).to(device)
    x = torch.rand(3,3,224,224)
    x = x.to(device)
    y = net(x)

