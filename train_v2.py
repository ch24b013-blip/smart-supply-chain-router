import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import BananaRotDataset
from model import BananaNet

# --- V2 Hyperparameters ---
BATCH_SIZE = 32
INITIAL_LR = 0.001
EPOCHS = 100 

def train_v2():
    print("🔥 Booting up the V2 Training Engine...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware locked: {device}")

    dataset = BananaRotDataset(csv_file="banana_labels.csv", root_dir="./banana_dataset")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = BananaNet().to(device)

    # 1. UPGRADE: Huber Loss (Handles outliers flawlessly)
    criterion = nn.HuberLoss(delta=1.0)

    # 2. UPGRADE: Adam Optimizer with Weight Decay (Prevents memorization)
    optimizer = optim.Adam(model.backbone.fc.parameters(), lr=INITIAL_LR, weight_decay=1e-4)

    # 3. UPGRADE: The Smart Sensor (Learning Rate Scheduler)
    # If the loss doesn't improve for 2 epochs, cut the learning rate in half (factor=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    print("\n🚀 Commencing V2 Neural Network Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device).view(-1, 1).float() 

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        
        # Grab the current speed of the AI to monitor it
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | Huber Loss: {avg_loss:.4f}")

        # Trigger the sensor to check if we need to hit the brakes!
        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "banana_net_v2.pth")
    print("\n✅ V2 TRAINING COMPLETE. Brain saved as 'banana_net_v2.pth'")

if __name__ == "__main__":
    train_v2()