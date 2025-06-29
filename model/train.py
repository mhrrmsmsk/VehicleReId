import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model.lfft import LFFT  # senin daha önce oluşturduğun model
from tqdm import tqdm
import os

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
train_data = datasets.ImageFolder("../data/vehicle_dataset", transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = LFFT().to(device)

# Loss: Triplet + ID
id_loss = nn.CrossEntropyLoss()
triplet_loss = nn.TripletMarginLoss(margin=1.0)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Label mapping
class_to_idx = train_data.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        feats = model(imgs)  # output: [B, D]

        # ID loss
        logits = feats[:, :len(class_to_idx)]  # varsayım
        loss_id = id_loss(logits, labels)

        # Triplet loss (örnekleme ile yapılmalı)
        a, p, n = feats[::3], feats[1::3], feats[2::3]
        if len(a) < 1:
            continue
        loss_tri = triplet_loss(a, p, n)

        loss = loss_id + loss_tri
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), f"lfft_epoch{epoch+1}.pt")
