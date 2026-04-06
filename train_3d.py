import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model_3d import FNO3d, add_grid
from timeit import default_timer
from train_utils import load_data_3D, relative_l2

# Configuration
data_path = "data/ns_V1e-3_N5000_T50.mat"
n_train = 200
n_test = 50
batch_size = 10
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.5

# T_in: input timesteps, T: output timesteps, S: resolution
T_in = 10
T = 40
S = 64

# Set up device, paths, and load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
latest_path = "outputs/fno3d_latest.pt"
final_path = "outputs/fno3d_final.pt"
loss_path = "outputs/losses_3d.npy"

print("Data is being loaded and preprocessed...")
a_train, u_train, a_test, u_test = load_data_3D(data_path, n_train, n_test, T_in, T, S)

# add positional and temporal embedding
x_train = add_grid(a_train)
x_test  = add_grid(a_test)

# Note that for 3D no normalization is applied
train_loader = DataLoader(
    TensorDataset(x_train, u_train),
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(x_test, u_test),
    batch_size=batch_size, shuffle=False # no need to shuffle test data
)

# Preparing the model, optimizer, and scheduler
model = FNO3d(n_modes_x=8, n_modes_y=8, n_modes_t=8, width=20, T_in=T_in).to(device) # 8, 8, 8, 20

# Count parameters for reference
n = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n:,}")

# Adam has weight decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING
print("\nEpoch   train L2    test L2    time per epoch (s)")
print("—" * 50)

train_losses = []
test_losses  = []

for ep in range(1, epochs + 1):
    # Time the epoch to monitor training speed
    t1 = default_timer()

    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x).squeeze(1) # remove channel dim for loss calculation; shape
        loss = relative_l2(pred, y)
        loss.backward()  # backpropagate the loss to compute gradients
        optimizer.step() # update model parameters using the optimizer
        train_loss += loss.item()

    scheduler.step()

    # Evaluation on test set without gradient tracking
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(1) # remove channel dim for loss calculation
            test_loss += relative_l2(pred, y).item()

    # Average losses over batches
    train_loss /= len(train_loader)
    test_loss  /= len(test_loader)

    # Time the epoch
    t2 = default_timer()

    # Record losses for diagnostics
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if ep % 10 == 0 or ep == 1:
        print(f"  {ep:>4}   {train_loss:.4f}      {test_loss:.4f}     {t2-t1:.1f}s")

    if ep % 100 == 0:
        torch.save(model.state_dict(), latest_path)
        print(f"  updated latest model to {latest_path}")

print("\nTraining has concluded.")
torch.save(model.state_dict(), final_path)
print(f"Final model saved to {final_path}")

# Save learning curves for later analysis
np.save(loss_path, {"train": train_losses, "test": test_losses})
print(f"Loss history saved to {loss_path}")
