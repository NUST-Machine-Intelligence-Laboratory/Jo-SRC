import torch
import torch.nn as nn
import torchvision
from utils.cnn import CNN
from utils.utils import init_weights


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden, projection_size, init_method='He'):
        super().__init__()

        mlp_hidden_size = round(mlp_hidden * in_channels)
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )
        init_weights(self.mlp_head, init_method)

    def forward(self, x):
        return self.mlp_head(x)


class Encoder(nn.Module):
    def __init__(self, arch='cnn', num_classes=200, pretrained=True):
        super().__init__()
        if arch.startswith('resnet') and arch in torchvision.models.__dict__.keys():
            resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = resnet.fc.in_features
        elif arch.startswith('cnn'):
            cnn = CNN(input_channel=3, n_outputs=num_classes)
            self.encoder = nn.Sequential(*list(cnn.children())[:-1])
            self.feature_dim = cnn.classifier.in_features
        else:
            raise AssertionError(f'{arch} is not supported!')

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.shape[0], -1)


class Model(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, mlp_hidden=2, pretrained=True):
        super().__init__()
        self.encoder = Encoder(arch, num_classes, pretrained)
        self.classifier = MLPHead(self.encoder.feature_dim, mlp_hidden, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
