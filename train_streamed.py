# src/train_streamed.py

import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import glob

# ---------------- CONFIG ----------------
# Use absolute paths to avoid "current working directory" confusion
BASE_DIR = os.path.abspath(os.getcwd())
VIDEO_DIR = os.path.join(BASE_DIR, "data", "videos")
EPOCHS = 8
LR = 5e-5
BATCH_SIZE = 64
FPS = 12
INPUT_SIZE = 224
MODEL_PATH = os.path.join(BASE_DIR, "models", "image_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

print(f"Training on device: {DEVICE}")
print(f"Looking for data in: {VIDEO_DIR}")

torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)
torch.set_num_threads(os.cpu_count())

# ---------------- TRANSFORM ----------------
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- MODEL ----------------
model = resnet18(weights="IMAGENET1K_V1")
for p in model.parameters(): p.requires_grad = False
for p in model.layer4.parameters(): p.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 2)
for p in model.fc.parameters(): p.requires_grad = True
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scaler = GradScaler()

# ---------------- VIDEO TRAINING ----------------
def train_on_video(video_path: str, label: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Error] Could not open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30
    interval = max(int(video_fps // FPS), 1)
    
    frames, labels = [], []
    frame_id = 0
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                margin = int(0.1 * w)
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
                face_crop = frame[y1:y2, x1:x2]
                
                try:
                    img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    with torch.no_grad():
                        frames.append(train_transform(img))
                        labels.append(label)
                        frames_processed += 1
                except Exception:
                    pass
                break 

            if len(frames) == BATCH_SIZE:
                process_batch(frames, labels)
                frames.clear()
                labels.clear()

        frame_id += 1

    if frames:
        process_batch(frames, labels)

    cap.release()
    # Debug: Check if faces were actually found in this video
    if frames_processed == 0:
        print(f"  [Warning] No faces detected in: {os.path.basename(video_path)}")

def process_batch(frames, labels):
    x = torch.stack(frames).pin_memory().to(DEVICE, non_blocking=True)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

    optimizer.zero_grad()
    with autocast():
        out = model(x)
        loss = criterion(out, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# ---------------- TRAIN LOOP ----------------
def train():
    model.train()
    print("\n--- Starting Training Process ---")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        total_videos = 0

        for label, cls in enumerate(["real", "fake"]):
            vid_dir = os.path.join(VIDEO_DIR, cls)
            
            if not os.path.isdir(vid_dir):
                print(f"[Error] Directory not found: {vid_dir}")
                print(f"        Please create '{cls}' folder inside 'data/videos'")
                continue

            # Case-insensitive recursive search for video files
            extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(vid_dir, '**', ext), recursive=True))
            
            # Remove duplicates if filesystem is case-insensitive
            files = sorted(list(set(files)))

            print(f"Found {len(files)} videos in '{cls}' class")
            total_videos += len(files)
            
            for i, video_path in enumerate(files):
                print(f"  [{cls.upper()}] Processing {i+1}/{len(files)}: {os.path.basename(video_path)}")
                train_on_video(video_path, label)

        if total_videos == 0:
            print("\n[CRITICAL ERROR] No videos found! Check your paths.")
            return

        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Checkpoint saved to {MODEL_PATH}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)