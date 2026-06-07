import torch
import os
import random
from torchvision import transforms
from PIL import Image
from model import BananaNet

# 1. Setup the exact same vision pipeline (BUT NO RANDOM AUGMENTATIONS!)
# When testing, we don't want to flip or zoom the image. We just resize and normalize.
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_rot(image_path, model_path="banana_net_v1.pth"):
    print("\n🧠 Waking up the trained Neural Network...")
    
    # 2. Hardware and Brain Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BananaNet().to(device)
    
    # Load the memories you just saved!
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # CRITICAL SDE STEP: Turn off "training mode" (disables Dropout)
    model.eval() 

    # 3. Load and prep the image
    img = Image.open(image_path).convert("RGB")
    # Add a fake "batch" dimension because the model expects [Batch, Colors, Height, Width]
    img_tensor = test_transform(img).unsqueeze(0).to(device) 

    # 4. Make the prediction
    # Turn off calculus engine to save GPU memory and run 10x faster
    with torch.no_grad(): 
        prediction = model(img_tensor)

    predicted_days = prediction.item()
    print(f"🍌 AI PREDICTION: This banana will rot in {predicted_days:.2f} days.")
    return predicted_days

if __name__ == "__main__":
    # SDE Auto-Tester: Grab a random image from your dataset folder
    base_dir = "./banana_dataset"
    all_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_dir) for f in filenames if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if all_images:
        test_image = random.choice(all_images)
        print(f"📸 Selected random test image: {test_image}")
        predict_rot(test_image)
    else:
        print("ERROR: Could not find any images to test.")