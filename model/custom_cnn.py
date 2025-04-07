import torch.nn as nn
import torch.nn.functional as F
import src.config as config
class CustomCNN(nn.Module):
    def __init__(self, num_classes = 101):
        super().__init__()

        # Original [3, 224, 224]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # dimensionality reduction [64, 112, 112]
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Second layer [128, 56, 56]
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third layer [256, 28, 28]
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Fourth layer [512, 14, 14]
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # adaptive pooling: transforms [batch, 512, H, W] to [batch, 512, 1, 1]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout
        self.dropout = nn.Dropout(p=config.DROPOUT)

        # Fully connected layer
        # [batch, 512, 1, 1]
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Note that deeper we go the more complex features model can recognize
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.adaptive_pool(x) #[batch, 512, 1, 1]
        # Flatten [batch, 512]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x= self.dropout(x)
        x = self.fc2(x)

        return x