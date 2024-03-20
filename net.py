from torch import nn
from torchsummary import summary

FACE_SHAPE = (64, 64)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class EmotionNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionNet, self).__init__()

        self.conv1 = ConvBlock(1, 16)
        self.conv2_1 = ConvBlock(16, 16)
        self.conv2_2 = ConvBlock(16, 32)
        self.conv2_3 = ConvBlock(16, 64)
        self.conv3_1 = ConvBlock(16, 16)
        self.conv3_2 = ConvBlock(32, 128)
        self.conv3_3 = ConvBlock(64, 32)
        self.conv4_2 = ConvBlock(128, 16)
        self.conv4_3 = ConvBlock(32, 16)
        self.conv5 = ConvBlock(16, 32, stride=2)
        self.conv6_1 = ConvBlock(32, 16, stride=2)
        self.conv6_2 = ConvBlock(32, 16)
        self.conv7_2 = ConvBlock(16, 16, stride=2)
        self.conv8_2 = ConvBlock(16, 16)
        self.conv9 = ConvBlock(16, 32)
        self.conv10_1 = ConvBlock(32, 32)
        self.conv10_2 = ConvBlock(32, 64)
        self.conv10_3 = ConvBlock(32, 16)
        self.conv11_1 = ConvBlock(32, 16)
        self.conv11_2 = ConvBlock(64, 32)
        self.conv12_2 = ConvBlock(32, 16)
        self.conv13 = ConvBlock(16, 16)
        self.conv14_1 = ConvBlock(16, 32)
        self.conv14_2 = ConvBlock(16, 16)
        self.conv14_3 = ConvBlock(16, 16)
        self.conv15_2 = ConvBlock(16, 32)
        self.conv15_3 = ConvBlock(16, 16)
        self.conv16_3 = ConvBlock(16, 32)
        self.conv17 = ConvBlock(32, 16)
        self.conv18 = ConvBlock(16, 32)
        self.conv19 = ConvBlock(32, 64)
        self.conv20 = ConvBlock(64, 128)
        self.conv21 = ConvBlock(128, 128)
        self.conv22_1 = ConvBlock(128, 64)
        self.conv22_2 = ConvBlock(128, 128)
        self.conv22_3 = ConvBlock(128, 32)
        self.conv23_1 = ConvBlock(64, 32)
        self.conv23_2 = ConvBlock(128, 64)
        self.conv24_2 = ConvBlock(64, 32)
        self.conv25 = ConvBlock(32, 16)

        self.bridge_conv5_13 = ConvBlock(32, 16, stride=2)
        self.bridge_conv5_17 = ConvBlock(32, 16, stride=2)
        self.bridge_conv9_17 = ConvBlock(32, 16)
        self.bridge_conv5_25 = ConvBlock(32, 16, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x_2_1 = self.conv2_1(x1)
        x_2_2 = self.conv2_2(x1)
        x_2_3 = self.conv2_3(x1)
        x_3_1 = self.conv3_1(x_2_1)
        x_3_2 = self.conv3_2(x_2_2)
        x_3_3 = self.conv3_3(x_2_3)
        x_4_2 = self.conv4_2(x_3_2)
        x_4_3 = self.conv4_3(x_3_3)

        x = x_4_2 + x_3_1 + x_4_3 + x1 + x
        x5 = self.conv5(x)
        bridge_5_13 = self.bridge_conv5_13(x5)
        bridge_5_17 = self.bridge_conv5_17(x5)
        x_6_1 = self.conv6_1(x5)
        x_6_2 = self.conv6_2(x5)
        x_7_2 = self.conv7_2(x_6_2)
        x_8_2 = self.conv8_2(x_7_2)
        x = x_8_2 + x_6_1
        x9 = self.conv9(x)
        bridge_9_17 = self.bridge_conv9_17(x9)
        x_10_1 = self.conv10_1(x9)
        x_10_2 = self.conv10_2(x9)
        x_10_3 = self.conv10_3(x9)
        x_11_1 = self.conv11_1(x_10_1)
        x_11_2 = self.conv11_2(x_10_2)
        x_12_2 = self.conv12_2(x_11_2)
        x = x_11_1 + x_12_2 + x_10_3
        x13 = self.conv13(x) + bridge_5_13
        x_14_1 = self.conv14_1(x13)
        x_14_2 = self.conv14_2(x13)
        x_14_3 = self.conv14_3(x13)
        x_15_2 = self.conv15_2(x_14_2)
        x_15_3 = self.conv15_3(x_14_3)
        x_16_3 = self.conv16_3(x_15_3)
        x17 = self.conv17(x_14_1 + x_15_2 + x_16_3) + bridge_5_17 + bridge_9_17
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        x21 = self.conv21(x20)
        x_22_1 = self.conv22_1(x21)
        x_22_2 = self.conv22_2(x21)
        x_22_3 = self.conv22_3(x21)
        x_23_1 = self.conv23_1(x_22_1)
        x_23_2 = self.conv23_2(x_22_2)
        x_24_2 = self.conv24_2(x_23_2)
        x25 = self.conv25(x_23_1 + x_24_2 + x_22_3)
        x_5_25 = self.bridge_conv5_25(x5)
        x = self.flatten(x25 + x_5_25)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 7分类

    net = EmotionNet(num_classes=7).to('cuda:0')
    summary(net, (1, FACE_SHAPE[0], FACE_SHAPE[1]))
