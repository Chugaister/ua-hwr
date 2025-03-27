import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, hidden_size=256, num_lstm_layers=2):
        super(CRNN, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, H, W)
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (B, 64, H/2, W/2)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (B, 128, H/4, W/4)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (B, 256, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (B, 256, H/4, W/4)
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # (B, 256, H/8, W/4)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (B, 512, H/8, W/4)
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (B, 512, H/8, W/4)
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1)),  # (B, 512, H/16, W/4)

            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # (B, 512, H/16, W/4)
            nn.ReLU(),
        )

        # Recurrent (LSTM) layer
        self.lstm = nn.LSTM(512, hidden_size, num_layers=num_lstm_layers, bidirectional=True, batch_first=True)

        # Fully connected layer to predict characters
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, 512, H/16, W/4)

        x = x.permute(0, 3, 1, 2)  # (B, W/4, 512, H/16)
        x = x.mean(dim=3)  # Average over height dimension (B, W/4, 512)

        x, _ = self.lstm(x)  # Now x has correct shape: (B, W/4, 512)
        x = self.fc(x)  # (B, W/4, num_classes)

        return x

