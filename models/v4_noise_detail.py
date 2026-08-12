import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# V4 NOISE-AWARE DETAIL RESTORATION
# ============================================================


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class DetailBlock(nn.Module):
    """
    Extracts local and wider-context detail features.
    """

    def __init__(self, channels=64):
        super().__init__()

        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )

        self.fuse = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        local = self.act(self.local(x))
        wide = self.act(self.dilated(x))

        out = torch.cat([local, wide], dim=1)

        return self.act(self.fuse(out))


class NoiseEstimator(nn.Module):
    """
    Estimates a soft noise map.

    Output:
        0 -> mostly clean/detail
        1 -> likely noisy region
    """

    def __init__(self, channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DetailEstimator(nn.Module):
    """
    Learns where genuine fine detail should be preserved.
    """

    def __init__(self, channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class V4NoiseAwareRestoration(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        channels=64,
        num_blocks=6
    ):
        super().__init__()

        self.channels = channels

        # ----------------------------------------------------
        # Shallow feature extraction
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Main residual body
        # ----------------------------------------------------

        self.residual_body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # ----------------------------------------------------
        # Detail branch
        # ----------------------------------------------------

        self.detail_block1 = DetailBlock(channels)

        self.detail_block2 = DetailBlock(channels)

        self.detail_estimator = DetailEstimator(channels)

        # ----------------------------------------------------
        # Noise estimation branch
        # ----------------------------------------------------

        self.noise_estimator = NoiseEstimator(channels)

        # ----------------------------------------------------
        # Feature fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Upsampling ×2
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        self.residual_head = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Learnable residual scaling
        # ----------------------------------------------------

        self.residual_scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, x):

        # ====================================================
        # Bicubic baseline
        # ====================================================

        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False
        )

        # ====================================================
        # Feature extraction
        # ====================================================

        features = self.head(x)

        # ====================================================
        # Main structure features
        # ====================================================

        structure = self.residual_body(features)

        # ====================================================
        # Detail extraction
        # ====================================================

        detail = self.detail_block1(structure)

        detail = self.detail_block2(detail)

        detail = self.detail_estimator(detail)

        # ====================================================
        # Noise estimation
        # ====================================================

        noise_map = self.noise_estimator(structure)

        # ====================================================
        # Noise-aware detail gating
        #
        # High noise -> suppress detail
        # Low noise  -> preserve detail
        # ====================================================

        clean_detail = detail * (1.0 - noise_map)

        # ====================================================
        # Fuse structure + clean detail
        # ====================================================

        fused = torch.cat(
            [structure, clean_detail],
            dim=1
        )

        fused = self.fusion(fused)

        # ====================================================
        # Upsample
        # ====================================================

        up = self.upsample(fused)

        # ====================================================
        # Residual reconstruction
        # ====================================================

        residual = self.residual_head(up)

        # ====================================================
        # Global residual learning
        # ====================================================

        output = base + self.residual_scale * residual

        return output


# ============================================================
# ALIAS
# ============================================================

V4DetailRestoration = V4NoiseAwareRestoration


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("V4 NOISE-AWARE DETAIL RESTORATION")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = V4NoiseAwareRestoration().to(device)

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    x = torch.randn(
        4,
        1,
        128,
        128,
        device=device
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():
        y = model(x)

    print(f"\nInput shape : {x.shape}")
    print(f"Output shape: {y.shape}")

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --------------------------------------------------------
    # Output verification
    # --------------------------------------------------------

    assert x.shape == (
        4,
        1,
        128,
        128
    )

    assert y.shape == (
        4,
        1,
        256,
        256
    )

    print("\n✓ Input shape correct")
    print("✓ Output shape correct")
    print("✓ V4 model test successful")

    print("\n" + "=" * 60)