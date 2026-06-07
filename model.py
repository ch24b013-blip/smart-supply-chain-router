import torch
import torch.nn as nn
from torchvision import models

class BananaNet(nn.Module):
    def __init__(self):
        super(BananaNet, self).__init__()
        
        # 1. Load the pre-trained ResNet18 backbone
        # We use the modern PyTorch weights API to grab the best version
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        
        # 2. Freeze the early layers (SDE Optimization)
        # This stops the model from forgetting how to see shapes/colors
        # and makes training 10x faster on your RTX card.
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Swap the final classification head for a Regression Head
        num_ftrs = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevents overfitting to the background table
            nn.Linear(128, 1) # Final output: 1 single float (Days to rot)
        )

    def forward(self, x):
        return self.backbone(x)

# --- SDE Unit Test ---
if __name__ == "__main__":
    print("Initializing BananaNet Architecture...")
    
    # Auto-detect your specific RTX GPU!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model onto hardware: {device}")
    
    # Load the model and send it to the hardware
    model = BananaNet().to(device)
    
    # Create a fake image tensor (1 image, 3 colors, 224x224) and send to hardware
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Push the fake image through the brain
    output = model(dummy_image)
    
    print(f"SUCCESS! Output shape: {output.shape}") 
    print(f"Predicted days to death (random initialized weights): {output.item():.2f}")