import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the custom SDE files you built!
from dataset import BananaRotDataset
from model import BananaNet

# --- Hyperparameters (The SDE Control Panel) ---
BATCH_SIZE = 32      # How many bananas the GPU looks at at once
LEARNING_RATE = 0.001 # How fast the AI changes its mind
EPOCHS = 10          # How many times it reads the entire textbook (dataset)

def train_model():
    print("🔥 Booting up the Training Engine...")
    
    # 1. Hardware Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware locked: {device}")

    # 2. Load the Data Factory
    dataset = BananaRotDataset(csv_file="banana_labels.csv", root_dir="./banana_dataset")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} augmented images into the pipeline.")

    # 3. Load the Brain
    model = BananaNet().to(device)

    # 4. Define the Grader (Loss) and the Optimizer
    # MSELoss = Mean Squared Error (Standard for predicting exact numbers)
    # Adam = The industry standard optimization algorithm
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.backbone.fc.parameters(), lr=LEARNING_RATE)

    # 5. The Actual Training Loop
    print("\n🚀 Commencing Neural Network Training...")
    
    for epoch in range(EPOCHS):
        model.train() # Tell the model it's gym time, not test time
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move the images and answers to the RTX card
            images = images.to(device)
            # Reshape labels from [32] to [32, 1] to match model output
            labels = labels.to(device).view(-1, 1).float() 

            # A. Zero the calculus gradients (clear the whiteboard)
            optimizer.zero_grad()

            # B. Forward Pass: The model takes a guess
            predictions = model(images)

            # C. Calculate the Error: How wrong was the guess?
            loss = criterion(predictions, labels)

            # D. Backward Pass: Calculate how to fix the error (Calculus)
            loss.backward()

            # E. Step: Update the brain's weights
            optimizer.step()

            running_loss += loss.item()

        # Print the progress at the end of every Epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Average Error (Loss): {avg_loss:.4f}")

    # 6. Save the trained brain to your hard drive
    torch.save(model.state_dict(), "banana_net_v1.pth")
    print("\n✅ TRAINING COMPLETE. Brain saved as 'banana_net_v1.pth'")

if __name__ == "__main__":
    train_model()