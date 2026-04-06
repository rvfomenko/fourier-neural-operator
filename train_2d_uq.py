import numpy as np
import torch
from timeit import default_timer
from model_2d_uq import FNO2dUQ, add_grid
from train_utils import build_dataloaders, relative_l2

# Configuration
mode = 0 # 0 for easier (241x241), 1 for harder (421x421)
n_train = 1000
n_test = 200
batch_size = 20
learning_rate = 0.001
epochs = 500

# Scheduler: every [step_size] epochs, multiply LR by [gamma]
step_size = 100
gamma = 0.5

# Set up device, paths, and load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
print("Data is being loaded and preprocessed...")

train_loader, test_loader, y_normalizer, config = build_dataloaders(
    mode=mode,
    n_train=n_train,
    n_test=n_test,
    batch_size=batch_size,
    add_grid=add_grid,
)
difficulty = config["difficulty"]
latest_path = f"outputs/fno2d_uq_{difficulty}_latest.pt"
final_path = f"outputs/fno2d_uq_{difficulty}_final.pt"
loss_path = f"outputs/losses_2d_uq_{difficulty}.npy"   
y_normalizer = y_normalizer.to(device)

# Define negative log-likelihood loss for training sigma and mu
def gaussian_nll(mu, sigma, target):
    sigma = sigma.clamp_min(1e-6) # prevent log(0) and division by zero
    return (0.5 * torch.log(sigma**2) + 0.5 * ((target - mu) / sigma)**2).mean()

# Initialize the model, optimizer, and scheduler
model = FNO2dUQ(n_modes_x=12, n_modes_y=12, width=32).to(device) # 12, 12, 32 

# Count parameters for reference
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# Adam has weight decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING
# Also tracking sigma to monitor uncertainty estimates
print("\nEpoch   train NLL   test NLL   train L2    test L2   mean sigma   time per epoch (s)")
print("—" * 86)

train_losses = []
test_losses = []

for ep in range(1, epochs + 1):
    # Time the epoch
    t1 = default_timer()

    model.train()
    train_nll = 0.0
    train_l2 = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        mu, sigma = model(x)
        mu = y_normalizer.decode(mu)
        sigma = y_normalizer.decode_scale(sigma)
        y = y_normalizer.decode(y)

        loss = gaussian_nll(mu, sigma, y)
        loss.backward()
        optimizer.step()

        train_nll += loss.item()
        train_l2 += relative_l2(mu, y).item()

    scheduler.step()

    # Evaluate on test set without gradient tracking
    model.eval()
    test_nll = 0.0
    test_l2 = 0.0
    sigma_acc = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mu, sigma = model(x)
            mu = y_normalizer.decode(mu)
            sigma = y_normalizer.decode_scale(sigma)
            test_nll += gaussian_nll(mu, sigma, y).item()
            test_l2 += relative_l2(mu, y).item()
            sigma_acc += sigma.mean().item()

    # Average losses over batches
    train_nll /= len(train_loader)
    train_l2 /= len(train_loader)
    test_nll /= len(test_loader)
    test_l2 /= len(test_loader)
    sigma_acc /= len(test_loader)

    # Time the epoch
    t2 = default_timer()

    # Record losses for diagnostics
    train_losses.append(train_l2)
    test_losses.append(test_l2)

    if ep % 10 == 0 or ep == 1:
        print(
            f"  {ep:>4}   {train_nll:.4f}    {test_nll:.4f}     {train_l2:.4f}"
            f"     {test_l2:.4f}     {sigma_acc:.6f}     {t2-t1:.1f}s"
        )

    if ep % 100 == 0:
        torch.save(model.state_dict(), latest_path)
        print(f"updated latest model to {latest_path}")

print("\nTraining has concluded.")
torch.save(model.state_dict(), final_path)
print(f"Final model saved to {final_path}")

np.save(f"outputs/losses_2d_uq_{difficulty}.npy", {"train": train_losses, "test": test_losses})
print(f"Loss history saved to {loss_path}")
