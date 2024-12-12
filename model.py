import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )

        # CONV Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),  # 28x28x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.01)
        )

        # Transition Block 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2)  # 14x14x32
        )

        # CONV Block 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, 1),  # Channel reduction
            nn.Conv2d(16, 32, 3, padding=1),  # 14x14x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.01)
        )

        # Transition Block 2
        self.trans2 = nn.Sequential(
            nn.MaxPool2d(2, 2)  # 7x7x32
        )

        # Output Block
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # 7x7x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )
        
        # Final Layer
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 1x1x16
            nn.Conv2d(16, 10, 1)  # 1x1x10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.conv3(x)
        x = self.trans2(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1) 