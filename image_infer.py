# src/image_infer.py

import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Temperature scaling (1.0 = Raw probabilities)
TEMPERATURE = 1.0

# Decision threshold
IMAGE_FAKE_THRESHOLD = 0.75

# ---------------- Model ----------------
model = resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, 2)

try:
    state_dict = torch.load("models/image_model.pth", map_location=DEVICE, weights_only=True)
except TypeError:
    state_dict = torch.load("models/image_model.pth", map_location=DEVICE)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# ---------------- Preprocess ----------------
# FIXED: Added Normalize to match the training script.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def infer_image(image_path: str) -> dict:
    """
    Image deepfake inference with calibrated confidence and decision.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return {"input_type": "image", "is_fake": False, "confidence": 0.0}

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        scaled_logits = logits / TEMPERATURE
        probs = F.softmax(scaled_logits, dim=1)

    # Assuming Class 0 = Real, Class 1 = Fake (Standard ImageFolder behavior)
    fake_prob = float(probs[0][1].item())
    
    is_fake = fake_prob >= IMAGE_FAKE_THRESHOLD

    return {
        "input_type": "image",
        "is_fake": is_fake,
        "confidence": round(fake_prob, 3)
    }