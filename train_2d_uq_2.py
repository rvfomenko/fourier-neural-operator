import numpy as np
import torch
from timeit import default_timer

from model_2d_uq import FNO2dUQ, add_grid
from train_utils import build_dataloaders, relative_l2

# Configuration
mode = 0  # 0 for easier (241x241), 1 for harder (421x421)
n_train = 1000
n_test = 200
batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.5

# Beta parameter for beta-NLL loss
beta = 0.2

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
beta_tag = f"b{str(beta).replace('.', 'p')}"
latest_path = f"outputs/fno2d_uq2_{beta_tag}_{difficulty}_latest.pt"
final_path = f"outputs/fno2d_uq2_{beta_tag}_{difficulty}_final.pt"
loss_path = f"outputs/losses_2d_uq2_{beta_tag}_{difficulty}.npy"
y_normalizer = y_normalizer.to(device)

# Define beta-NLL loss for training sigma and mu
def beta_nll(mu, sigma, target, beta=0.5):
    sigma = sigma.clamp_min(1e-6)
    nll = 0.5 * torch.log(sigma ** 2) + 0.5 * ((target - mu) / sigma) ** 2
    weight = (sigma ** (2 * beta)).detach()
    return (weight * nll).mean()

# Initialize the model, optimizer, and scheduler
model = FNO2dUQ(n_modes_x=12, n_modes_y=12, width=32).to(device)

# Count parameters for reference
n = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n:,}")
print(f"Beta-NLL beta: {beta}")

# Adam has weight decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING
# Also tracking sigma to monitor uncertainty estimates
print("\nEpoch   train BNLL   test BNLL    train L2   test L2   mean sigma   time per epoch (s)")
print("—" * 88)

train_losses = []
test_losses = []

for ep in range(1, epochs + 1):
    # Time the epoch
    t1 = default_timer()

    model.train()
    train_bnll = 0.0
    train_l2 = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        mu_norm, sigma_norm = model(x)
        sigma_norm = sigma_norm.clamp_min(1e-6)
        loss = beta_nll(mu_norm, sigma_norm, y, beta=beta)
        loss.backward()
        optimizer.step()

        mu = y_normalizer.decode(mu_norm.detach())
        sigma = y_normalizer.decode_scale(sigma_norm.detach()).clamp_min(1e-6)
        y_phys = y_normalizer.decode(y)
        train_bnll += loss.item()
        train_l2 += relative_l2(mu, y_phys).item()

    scheduler.step()

    # Evaluate on test set without gradient tracking
    model.eval()
    test_bnll = 0.0
    test_l2 = 0.0
    sigma_acc = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mu, sigma = model(x)
            mu = y_normalizer.decode(mu)
            sigma = y_normalizer.decode_scale(sigma)
            test_bnll += beta_nll(mu, sigma, y, beta=beta).item()
            test_l2 += relative_l2(mu, y).item()
            sigma_acc += sigma.mean().item()

    # Average losses over batches
    train_bnll /= len(train_loader)
    train_l2 /= len(train_loader)
    test_bnll /= len(test_loader)
    test_l2 /= len(test_loader)
    sigma_acc /= len(test_loader)

    # Time the epoch
    t2 = default_timer()

    # Record losses for diagnostics
    train_losses.append(train_l2)
    test_losses.append(test_l2)

    if ep % 10 == 0 or ep == 1:
        print(
            f"  {ep:>4}   {train_bnll:<11.4f} {test_bnll:<10.4f} {train_l2:9.4f}"
            f"     {test_l2:.4f}    {sigma_acc:.6f}     {t2-t1:.1f}s"
        )

    if ep % 100 == 0:
        torch.save(model.state_dict(), latest_path)
        print(f"updated rolling checkpoint -> {latest_path}")

print("\nTraining has concluded.")
torch.save(model.state_dict(), final_path)
print(f"Final model saved to {final_path}")

# Save learning curves for later analysis
np.save(loss_path, {"train": train_losses, "test": test_losses})
print(f"Loss history saved to {loss_path}")
