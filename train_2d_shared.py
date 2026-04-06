import numpy as np
import torch
from timeit import default_timer
from model_2d_shared import FNO2dShared, add_grid
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
y_normalizer = y_normalizer.to(device)
latest_path = f"outputs/fno2d_shared_{difficulty}_latest.pt"
loss_path = f"outputs/losses_2d_shared_{difficulty}.npy"
final_path = f"outputs/fno2d_shared_{difficulty}_final.pt"

# Initialize the model, optimizer, and scheduler
model = FNO2dShared(n_modes_x=12, n_modes_y=12, width=32).to(device)

# Count parameters for reference
n= sum(p.numel() for p in model.parameters())
print(f"Parameters: {n:,}  (weight-sharing across 4 layers)")

# Adam has weight decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING --- TRAINING
print("\nEpoch   train L2    test L2    time per epoch (s)")
print("—" * 50)

train_losses = []
test_losses = []

for ep in range(1, epochs + 1):
    # Time the epoch
    t1 = default_timer()

    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = y_normalizer.decode(model(x)) # decode to unnormalized space for loss calculation
        y = y_normalizer.decode(y)

        loss = relative_l2(pred, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    scheduler.step()

    # Evaluate on test set without gradient tracking
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = y_normalizer.decode(model(x)) # decode to unnormalized space for loss calculation
            test_loss += relative_l2(pred, y).item()

    # Average losses over batches
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)

    # Time the epoch
    t2 = default_timer()

    # Record losses for diagnostics
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if ep % 10 == 0 or ep == 1:
        print(f"  {ep:>4}   {train_loss:.4f}      {test_loss:.4f}     {t2-t1:.1f}s")

    if ep % 100 == 0:
        torch.save(model.state_dict(), latest_path)
        print(f"updated latest model to {latest_path}")

print("\nTraining has concluded.")
torch.save(model.state_dict(), final_path)
print(f"Final model saved to {final_path}")

# Save learning curves for later analysis
np.save(loss_path, {"train": train_losses, "test": test_losses})
print(f"Loss history saved to {loss_path}")