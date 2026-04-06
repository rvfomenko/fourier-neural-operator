import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes_x, n_modes_y):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y # by symmetry N//2+1 max. other modes found by conjugate (signal is real)

        # Scale factor for weight initialisation; each frequency starts "weak"
        A = 1 / (in_channels * out_channels)

        # KERNEL: Complex weights that decide how the fourier modes contributes to determine the (untransformed) output
        self.weights1 = nn.Parameter(
            A * torch.rand(
                size=(in_channels, out_channels, n_modes_x, n_modes_y),
                dtype=torch.cfloat
            )
        )

        self.weights2 = nn.Parameter(
            A * torch.rand(
                size=(in_channels, out_channels, n_modes_x, n_modes_y),
                dtype=torch.cfloat
            )
        )

    def projection(self, input, weights):
        # Project input (b, i, x, y) to output (b, o, x, y) using weights (i, o, x, y) (implied summation over i)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape # batch, in_channels, grid points along x, grid points along y

        # Forward FFT
        x_ft = torch.fft.rfft2(x) # transformed shape: (B, in_channels, H, W//2+1); unique modes remaining (others can be found through conjugate)

        # Transform the modes using the weights (again due to real a[x,y] theres symmetry along W (or H))
        out_ft = torch.zeros(
            (B, self.out_channels, H, W // 2 + 1),
            dtype=torch.cfloat,
            device=x.device
        )

        # Lower-left corner of the frequency domain
        out_ft[:, :, :self.n_modes_x, :self.n_modes_y] = self.projection(
            x_ft[:, :, :self.n_modes_x, :self.n_modes_y], self.weights1
        )

        # Upper-left corner (negative frequencies along first axis)
        out_ft[:, :, -self.n_modes_x:, :self.n_modes_y] = self.projection(
            x_ft[:, :, -self.n_modes_x:, :self.n_modes_y], self.weights2
        )

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(H, W)) # transformed shape: (B, out_channels, H, W) (internal conjugation to get W) (ensure size H, W)
        return x

class FNOBlock2d(nn.Module):
    def __init__(self, channels, n_modes_x, n_modes_y):
        super().__init__()
        # Spectral convolution to capture global interactions in the frequency domain
        self.spectral = SpectralConv2d(channels, channels, n_modes_x, n_modes_y)
        # 1×1 convolution to combine channel features at each gridpoint
        self.skip = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Preservation of local pointwise information using "skip"
        return self.spectral(x) + self.skip(x)


class FNO2dUQ(nn.Module):
    def __init__(self, n_modes_x=12, n_modes_y=12, width=32, n_layers=4, in_channels=3):
        super().__init__()
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
        self.width = width
        self.padding = 9 # pad the domain to account for non-periodic boundaries
        self.n_layers = n_layers

        # Lift network to expanded representation of [width] channels, from initial [in_channels]
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # List of FNO blocks, each doing SpecConv2D + Channel mixing skip
        self.blocks = nn.ModuleList([FNOBlock2d(width, n_modes_x, n_modes_y) for _ in range(n_layers)])

        # Projection: first widening capacity for nonlinear mixing, then collapsing to one value
        self.proj_mu = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.proj_sigma = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.lift(x)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for block in self.blocks[:-1]:
            x = F.gelu(block(x)) # (B, width, H, W)

        x = self.blocks[-1](x) # No GELU after last block
        x = x[..., :-self.padding, :-self.padding]

        mu = self.proj_mu(x) # (B, 1, H, W)
        sigma = F.softplus(self.proj_sigma(x)) # (B, 1, H, W)

        return mu, sigma


def add_grid(a): # shape: (B, 1, H, W)
    B, _, H, W = a.shape # has only the a[x,y] channel for [B] example
    device = a.device

    # Grids accross [0,1] reshaped to include the other dimensions (view) then copied into them (expand)
    grid_x = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    grid_y = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

    return torch.cat([a, grid_x, grid_y], dim=1) # concatenate a + positional encoding in dim 1 (C); shape: (B, 3, H, W)
