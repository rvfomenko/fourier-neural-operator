import scipy.io
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Normalizer that standardizes data to zero mean and unit variance
class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):
    # ASSUMING BATCH FIRST DIMENSION
        self.mean = x.mean(0)
        self.std = x.std(0, unbiased=False) # use population std
        self.eps = eps

    # Standardize input to zero mean and unit variance
    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    # Decode output back to original space
    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    # Decode just the scale (for UQ model)
    def decode_scale(self, x):
        return x * (self.std + self.eps)

    # Move mean and std to the same device as the model
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

# Utils for loading data, all variables for each mode in one function
def get_mode_config(mode):
    case = ["241", "421"][mode]
    res_original = [241, 421][mode]
    res_working = [60, 85][mode]
    return {
        "difficulty": case,
        "res_original": res_original,
        "res_working": res_working,
        "train_path": f"data/piececonst_r{case}_N1024_smooth1.mat",
        "test_path": f"data/piececonst_r{case}_N1024_smooth2.mat"
    }

# Loading .mat data, subsampling to working resolution, and adding channel dimension
def load_data(path, n_samples, res_original, res_working):
    data = scipy.io.loadmat(path)

    # 'a' is input, 'u' is output (even though in 3D 'u' is also the input)
    a = torch.tensor(data["coeff"], dtype=torch.float32) # float32 to save memory
    u = torch.tensor(data["sol"], dtype=torch.float32)

    # Subsample the data to the working resolution (e.g. 64x64)
    # by skipping points in the original resolution (e.g. 241x241)
    sub = (res_original - 1) // (res_working - 1)
    a = a[:n_samples, ::sub, ::sub][:, :res_working, :res_working]
    u = u[:n_samples, ::sub, ::sub][:, :res_working, :res_working]

    # Add channel dimension
    a = a.unsqueeze(1)
    u = u.unsqueeze(1)
    return a, u

def load_data_3D(path, n_train, n_test, T_in=10, T_out=40, res=64):
    # Note the terminology: in 3D, "a" is the input, "u" is the output
    # but ALL are vorticities
    with h5py.File(path, "r") as f:
        data = torch.tensor(np.array(f["u"]), dtype=torch.float32)

    # Fix ordering to (N, S, S, T)
    data = data.permute(3, 1, 2, 0)

    # Split into train and test
    u_train_full = data[:n_train]
    u_test_full = data[-n_test:]

    # Sample T_in for input, T for output
    a_train = u_train_full[:, :, :, :T_in]
    u_train = u_train_full[:, :, :, T_in:T_in+T_out]
    a_test = u_test_full[:, :, :, :T_in]
    u_test = u_test_full[:, :, :, T_in:T_in+T_out]

    # Repeat input across output dimension
    a_train = a_train.reshape(n_train, res, res, 1, T_in).repeat(1, 1, 1, T_out, 1)
    a_test = a_test.reshape(n_test,  res, res, 1, T_in).repeat(1, 1, 1, T_out, 1)

    # Channels-first for add_grid and model
    a_train = a_train.permute(0, 4, 1, 2, 3)
    a_test = a_test.permute(0, 4, 1, 2, 3)

    return a_train, u_train, a_test, u_test

# Relative L2 error metric for evaluation.
# This way the error is normalized by magnitude of signal
def relative_l2(pred, target):
    batch_size = pred.shape[0]
    # Norm of per-pixel difference, norm of target, then average over batch
    diff = torch.norm(pred.reshape(batch_size, -1) - target.reshape(batch_size, -1), dim=1)
    tgt = torch.norm(target.reshape(batch_size, -1), dim=1)
    return (diff / tgt).mean()

# Build dataloaders for training and testing
# Raw 2D: (B, X, Y); Raw 3D: (T, X, Y, B) << PERMUTE
def build_dataloaders(mode, n_train, n_test, batch_size, add_grid):
    # Get config for the mode (e.g. resolution, paths) and load the training and test data
    config = get_mode_config(mode)
    a_train, u_train = load_data(config["train_path"], n_train, config["res_original"], config["res_working"])
    a_test, u_test = load_data(config["test_path"], n_test, config["res_original"], config["res_working"])

    # Normalize the data
    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)
    u_normalizer = UnitGaussianNormalizer(u_train)
    u_train_encoded = u_normalizer.encode(u_train)

    x_train, x_test = add_grid(a_train), add_grid(a_test)

    train_loader = DataLoader(
        TensorDataset(x_train, u_train_encoded),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, u_test),
        batch_size=batch_size, shuffle=False # no need to shuffle test data
    )
    return train_loader, test_loader, u_normalizer, config
