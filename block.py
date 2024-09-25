import torch
import torch.nn as nn
import torch.nn.functional as F

# 인코더 블록 정의
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # 너비 차원이 1 이하로 너무 작은 경우, 풀링을 건너뜁니다.
        if x.size(-1) > 1:
            x_pooled = self.pool(x)
        else:
            x_pooled = x  # 풀링을 생략하고 그대로 전달

        return x_pooled, x


# 디코더 블록 정의
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        
        # 업샘플된 텐서와 스킵 커넥션의 크기 맞추기
        if x.size(2) != skip_connection.size(2) or x.size(3) != skip_connection.size(3):
            x = F.interpolate(x, size=(skip_connection.size(2), skip_connection.size(3)), mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection
        x = torch.cat((x, skip_connection), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# 전체 U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 인코더 경로
        x1_pooled, x1 = self.encoder1(x)
        x2_pooled, x2 = self.encoder2(x1_pooled)
        x3_pooled, x3 = self.encoder3(x2_pooled)
        x4_pooled, x4 = self.encoder4(x3_pooled)

        # Bottleneck
        x_bottleneck = self.bottleneck(x4_pooled)

        # 디코더 경로
        x = self.decoder4(x_bottleneck, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)

        # 최종 출력
        out = self.final_conv(x)
        return out

class EncoderBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # 너비 차원이 1 이하로 너무 작은 경우, 풀링을 건너뜁니다.
        if x.size(-1) > 1:
            x_pooled = self.pool(x)
        else:
            x_pooled = x  # 풀링을 생략하고 그대로 전달

        return x_pooled, x  # 스킵 연결을 위해 풀링 전 값을 반환

# 1D 디코더 블록 정의
class DecoderBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock1D, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        
        # 업샘플된 텐서와 스킵 커넥션의 크기를 맞추기
        if x.size(2) != skip_connection.size(2):
            x = F.interpolate(x, size=skip_connection.size(2), mode='linear', align_corners=True)
        
        # 스킵 연결과 Concatenate
        x = torch.cat((x, skip_connection), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# 1D U-Net 클래스 정의 (A 구성)
class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1D, self).__init__()
        self.encoder1 = EncoderBlock1D(in_channels, 64)
        self.encoder2 = EncoderBlock1D(64, 128)
        self.encoder3 = EncoderBlock1D(128, 256)
        self.encoder4 = EncoderBlock1D(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = DecoderBlock1D(1024, 512)
        self.decoder3 = DecoderBlock1D(512, 256)
        self.decoder2 = DecoderBlock1D(256, 128)
        self.decoder1 = DecoderBlock1D(128, 64)

        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 인코더 경로
        x1_pooled, x1 = self.encoder1(x)
        x2_pooled, x2 = self.encoder2(x1_pooled)
        x3_pooled, x3 = self.encoder3(x2_pooled)
        x4_pooled, x4 = self.encoder4(x3_pooled)

        # Bottleneck
        x_bottleneck = self.bottleneck(x4_pooled)

        # 디코더 경로
        x = self.decoder4(x_bottleneck, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)

        # 최종 출력
        out = self.final_conv(x)
        return out