import os
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

from data.mnist import get_mnist_loaders


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("train.log")],
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=3, delta=1e-3, path="models/best.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = None
        self.state = None
        self.counter = 0
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def step(self, loss, model: nn.Module) -> bool:
        # Yaxşılaşma varmı? (ilk dəfə həmişə bəli)
        if self.best_loss is None or self.best_loss - loss > self.delta:
            logger.info("New best val_loss: %s -> %.4f", self.best_loss, loss)
            self.best_loss = loss
            self.counter = 0
            self.state = deepcopy(model.state_dict())
            torch.save(self.state, self.path)
        else:
            self.counter += 1
            logger.info("EarlyStopping: no improvement, counter %d/%d",
                        self.counter, self.patience)

        return self.counter >= self.patience


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(32, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # 28 -> 14
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # 14 -> 7
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))              # 7x7
        return self.fc(self.flatten(x))


# ---------- Data ----------
train_loader, val_loader, test_loader, classes = get_mnist_loaders(512)
logger.info("Data loaded | batches: train=%d, val=%d, test=%d | classes=%d",
            len(train_loader), len(val_loader), len(test_loader), len(classes))

# ---------- Model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
num_epochs = 15
early_stopping = EarlyStopping(patience=3)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.01,
    total_iters=num_epochs * len(train_loader),
)

num_params = sum(p.numel() for p in model.parameters())
logger.info("Model: CNN | params=%s | device=%s", f"{num_params:,}", device)

# ---------- Training ----------
# logging_redirect_tqdm routes log lines through tqdm.write so they don't break the progress bars
with logging_redirect_tqdm():
    for epoch in trange(num_epochs, desc="Epochs"):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_acc = running_loss / total, correct / total

        # validation pass
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Val {epoch + 1}/{num_epochs}", leave=False):
                xb, yb = xb.to(device), yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_running_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += xb.size(0)

        val_loss, val_acc = val_running_loss / val_total, val_correct / val_total

        logger.info("Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | lr=%.2e",
                    epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc,
                    scheduler.get_last_lr()[0])

        if early_stopping.step(val_loss, model):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

logger.info("Training finished")

# Ən yaxşı val loss olan çəkiləri (weights) geri yüklə
model.load_state_dict(early_stopping.state)

# ---------- Test: bütün test set üzərində ----------
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0

with torch.no_grad():
    for x_t, y_t in test_loader:
        x_t, y_t = x_t.to(device), y_t.to(device)
        out = model(x_t)

        test_loss += criterion(out, y_t).item() * x_t.size(0)
        test_correct += (out.argmax(dim=1) == y_t).sum().item()
        test_total += x_t.size(0)

logger.info("Test | loss=%.4f acc=%.4f", test_loss / test_total, test_correct / test_total)

# ---------- Vizuallaşdırma: 1 batch ----------
xb, yb = next(iter(test_loader))
xb, yb = xb.to(device), yb.to(device)

with torch.no_grad():
    preds = model(xb).argmax(dim=1)

print("Batch accuracy:", (preds == yb).float().mean().item())

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(xb[i].squeeze().cpu().numpy(), cmap="gray")
    pred = classes[preds[i].item()]
    true = classes[yb[i].item()]
    ax.set_title(f"Təxmin: {pred} | Həqiqi: {true}", color="green" if pred == true else "red")
    ax.axis("off")
plt.tight_layout()
plt.savefig("predictions.png", dpi=120)
plt.show()