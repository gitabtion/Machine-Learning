import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


def mlp(n_classes=2):
    model = Net(n_classes)
    return model
