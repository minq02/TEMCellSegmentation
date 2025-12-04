import torch
import torch.nn as nn
import torch.nn.functional as F

# attention positional data check

class DoubleConv(nn.Module):
    """
    (Conv -> BN -> ReLU) * 2
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """
    Attention gate from "Attention U-Net":
    - g: gating signal from decoder (coarse, low-res)
    - x: skip connection from encoder (fine, high-res)
    Produces attention coefficients to modulate x.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # linear transformations for gating and skip feature maps
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: [B, F_g, H, W] (decoder)
        x: [B, F_l, H, W] (encoder skip)

        output: attended skip connection, same shape as x
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)        # [B, 1, H, W]
        return x * psi             # broadcast along channel dim


class UpBlockWithAttention(nn.Module):
    """
    Up-convolution + Attention gate + DoubleConv.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        """
        in_ch: channels from previous decoder level
        skip_ch: channels from encoder skip connection
        out_ch: output channels after DoubleConv
        """
        super().__init__()
        # upsample
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # attention gate uses:
        #   F_g = out_ch (after upsample)
        #   F_l = skip_ch
        #   F_int = out_ch // 2 (intermediate)
        self.attention = AttentionBlock(F_g=out_ch, F_l=skip_ch, F_int=out_ch // 2)

        # after concat(attended_skip, upsampled), channels = out_ch + skip_ch
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)  # upsampled decoder feature

        # in case of slight size mismatches due to odd dimensions
        if x.size()[2:] != skip.size()[2:]:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        # apply attention to skip connection
        skip_att = self.attention(g=x, x=skip)
        x = torch.cat([skip_att, x], dim=1)
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    """
    Full-scale Attention U-Net.
    Default: 4 down-sampling steps (depth 5 including bottleneck).
    Filters: 64, 128, 256, 512, 1024.
    """
    def __init__(self, in_ch=1, n_classes=2, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch)          # 64
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)    # 128
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)  # 256
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)  # 512
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)  # 1024

        # Decoder with attention
        self.up4 = UpBlockWithAttention(
            in_ch=base_ch * 16,   # 1024
            skip_ch=base_ch * 8,  # 512
            out_ch=base_ch * 8,   # 512
        )
        self.up3 = UpBlockWithAttention(
            in_ch=base_ch * 8,    # 512
            skip_ch=base_ch * 4,  # 256
            out_ch=base_ch * 4,   # 256
        )
        self.up2 = UpBlockWithAttention(
            in_ch=base_ch * 4,    # 256
            skip_ch=base_ch * 2,  # 128
            out_ch=base_ch * 2,   # 128
        )
        self.up1 = UpBlockWithAttention(
            in_ch=base_ch * 2,    # 128
            skip_ch=base_ch,      # 64
            out_ch=base_ch,       # 64
        )

        # Output head
        self.out_conv = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool1(x1)

        x2 = self.enc2(x2)
        x3 = self.pool2(x2)

        x3 = self.enc3(x3)
        x4 = self.pool3(x3)

        x4 = self.enc4(x4)
        x5 = self.pool4(x4)

        # Bottleneck
        x5 = self.bottleneck(x5)

        # Decoder with attention gates
        x = self.up4(x5, x4)
        x = self.up3(x,  x3)
        x = self.up2(x,  x2)
        x = self.up1(x,  x1)

        logits = self.out_conv(x)
        return logits


if __name__ == "__main__":
    # quick sanity check
    model = AttentionUNet(in_ch=1, n_classes=2)
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    print("Input:", x.shape, "Output:", y.shape)

class AdaptedAttentionUNet(nn.Module):
    """
    Tiny learnable input adapter + Attention U-Net.
    Adapter is a single conv layer that can learn to fix
    brightness/contrast/low-frequency issues in raw images.
    """
    def __init__(self, in_ch=1, n_classes=5, base_ch=64):
        super().__init__()
        # 1-layer "adapter" as suggested by your professor
        self.adapter = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        # your original Attention U-Net
        self.unet = AttentionUNet(in_ch=in_ch, n_classes=n_classes, base_ch=base_ch)

    def forward(self, x):
        x = self.adapter(x)
        return self.unet(x)
