import torch
import torch.nn as nn
from metrics import recall_report
from data import get_loaders
from model import ECGNet

USE_WEIGHTS = False
EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, y_train = get_loaders()
model = ECGNet().to(device)

if USE_WEIGHTS:
    counts = torch.bincount(y_train).float()
    weights = (counts.sum() / counts).to(device)  # rare class -> bigger weight
else:
    weights = None

loss_fn = nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"epoch {epoch + 1}: loss {total / len(train_loader):.4f}")

model.eval()
preds, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()

names = ["normal", "supraventricular", "ventricular", "fusion", "unknown"]
recall_report(labels, preds, names)