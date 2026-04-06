import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes_x, n_modes_y, n_modes_t):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_x = n_modes_x      
        self.n_modes_y = n_modes_y      
        self.n_modes_t = n_modes_t # modes throughout time, by symmetry N//2+1 max.

        # Scale factor for weight initialisation; each frequency starts "weak"
        A = 1 / (in_channels * out_channels)

        # KERNEL: Complex weights that decide how the fourier modes contributes to determine the (untransformed) output
        # 4 weights because there are 2 extra corners in the frequency space now that the redundant half is in time, not space.
        self.weights1 = nn.Parameter(
            A * torch.rand(
                size=(in_channels, out_channels, n_modes_x, n_modes_y, n_modes_t),
                dtype=torch.cfloat
            )
        )

        self.weights2 = nn.Parameter(
            A * torch.rand(
                size=(in_channels, out_channels, n_modes_x, n_modes_y, n_modes_t),
                dtype=torch.cfloat
            )
        )

        self.weights3 = nn.Parameter(
            A * torch.rand(
                size=(in_channels, out_channels, n_modes_x, n_modes_y, n_modes_t),
                dtype=torch.cfloat
            )
        )

        self.weights4 = nn.Parameter(
            A * torch.rand(
                size=(in_channels, out_channels, n_modes_x, n_modes_y, n_modes_t),
                dtype=torch.cfloat
            )
        )

    def projection(self, input, weights):
        # Project input (b, i, x, y, t) to output (b, o, x, y, t) using weights (i, o, x, y, t) (implied summation over i)
        return torch.einsum("bixyt,ioxyt->boxyt", input, weights) # (b)atch, (i)nput, (o)utput, (x)-modes, (y)-modes, (t)ime
         
    def forward(self, x):
        B, C, H, W, T = x.shape # batch, in_channels, grid points along (x, y, t)
        
        # Forward FFT
        x_ft =  torch.fft.rfftn(x, dim=(-3, -2, -1)) # transformed shape: (B, in_channels, H, W, T//2+1); unique modes remaining (others can be found through conjugate)

        # Transform the modes using the weights (again due to real a[x,y] theres symmetry that can be exploited on an axis
        out_ft = torch.zeros(
            (B, self.out_channels, H, W, T// 2 + 1),
            dtype=torch.cfloat,
            device=x.device
        )
        
        # Four corners now because the redudant half is in kt, not ky dimension. 
        out_ft[:, :, :self.n_modes_x, :self.n_modes_y, :self.n_modes_t] = self.projection(
            x_ft[:, :, :self.n_modes_x, :self.n_modes_y, :self.n_modes_t], self.weights1
        )

        out_ft[:, :, -self.n_modes_x:, :self.n_modes_y, :self.n_modes_t] = self.projection(
            x_ft[:, :, -self.n_modes_x:, :self.n_modes_y, :self.n_modes_t], self.weights2
        )

        out_ft[:, :, :self.n_modes_x, -self.n_modes_y:, :self.n_modes_t] = self.projection(
            x_ft[:, :, :self.n_modes_x, -self.n_modes_y:, :self.n_modes_t], self.weights3
        )

        out_ft[:, :, -self.n_modes_x:, -self.n_modes_y:, :self.n_modes_t] = self.projection(
            x_ft[:, :, -self.n_modes_x:, -self.n_modes_y:, :self.n_modes_t], self.weights4
        )

        # Inverse FFT
        x = torch.fft.irfftn(out_ft, s=(H, W, T), dim=(-3, -2, -1))
        return x

class FNOBlock3d(nn.Module):
    def __init__(self, channels, n_modes_x, n_modes_y, n_modes_t):
        super().__init__()
        # Spectral convolution to capture global interactions in the frequency domain
        self.spectral = SpectralConv3d(channels, channels, n_modes_x, n_modes_y, n_modes_t)
        # 1×1 convolution to combine channel features at each gridpoint
        self.skip = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        # preservation of local pointwise information using "skip"
        return self.spectral(x) + self.skip(x)


class FNO3d(nn.Module):
    def __init__(self, n_modes_x=12, n_modes_y=12, n_modes_t=12, width=20, n_layers=4, T_in=10):
        super().__init__()
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
        self.n_modes_t = n_modes_t
        self.T_in = T_in
        self.width = width
        self.padding = 6 # pad the domain to account for non-periodic temporal boundaries
        self.n_layers = n_layers

        # Lift network to expanded representation of [width] channels, from initial [T_in + 3] (3 extra for grid) channels
        self.lift = nn.Conv3d(T_in + 3, width, kernel_size=1) # lifting (x, y, t) + input timesteps dims.

        # List of FNO blocks, each doing SpecConv3d + skip
        self.blocks = nn.ModuleList([FNOBlock3d(width, n_modes_x, n_modes_y, n_modes_t) for _ in range(n_layers)])

        # Projection: first widening capacity for nonlinear mixing, then collapsing to one value
        self.proj = nn.Sequential(
            nn.Conv3d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.lift(x)
        x = F.pad(x, [0, self.padding]) # pad only time dimension, NS runs in periodic spatial domain

        for block in self.blocks[:-1]:
            x = F.gelu(block(x)) # (B, width, H, W, T)

        x = self.blocks[-1](x) # No GELU after last block
        x = x[..., :-self.padding]
        x = self.proj(x) # (B, 1, H, W, T)

        return x

# Shape is odd, this is because T in the last dimension responds to T_out,
# and is a trick to let the model know how many time steps it needs to predict.
# Training data initially comes in as (T, H, W, B)
def add_grid(a):  # shape: (B, T_in, H, W, T)
    B, _, H, W, T = a.shape
    device = a.device

    grid_x = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1, 1).expand(B, 1, H, W, T)
    grid_y = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W, 1).expand(B, 1, H, W, T)
    grid_t = torch.linspace(0, 1, T, device=device).view(1, 1, 1, 1, T).expand(B, 1, H, W, T)

    return torch.cat([a, grid_x, grid_y, grid_t], dim=1)  # (B, T_in+3, H, W, T)