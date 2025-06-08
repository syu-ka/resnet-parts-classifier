# このスクリプトは、data/train および data/val を使って ResNet18 を学習し、
# ../experiments_train/以下に日時付きのフォルダを作成して、学習済みモデルと設定情報（train_config.txt）を保存します。

import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch import nn, optim
import os
from datetime import datetime

# --- パス設定 ---
train_dir = "../data/train"
val_dir = "../data/val"

# --- 出力先フォルダを作成 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
exp_dir = os.path.join("../experiments_train", timestamp)
os.makedirs(exp_dir, exist_ok=True)
model_output_path = os.path.join(exp_dir, "resnet18.pth")

# --- クラス数（自動取得） ---
classes = sorted(os.listdir(train_dir))
num_classes = len(classes)

# --- 前処理 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- データセットとデータローダ ---
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデル定義（ResNet18） ---
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# --- 損失関数と最適化 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# --- 学習ループ ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # バリデーション精度
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# --- モデル保存 ---
torch.save(model.state_dict(), model_output_path)
print(f"✅ モデル保存完了: {model_output_path}")

# --- train_config.txt に設定を記録 ---
config_path = os.path.join(exp_dir, "train_config.txt")
with open(config_path, "w", encoding="utf-8") as cfg:
    cfg.write(f"日時: {timestamp}\n")
    cfg.write(f"エポック数: {num_epochs}\n")
    cfg.write(f"最適化手法: Adam (lr=0.001)\n")
    cfg.write(f"損失関数: CrossEntropyLoss\n")
    cfg.write("使用クラス（学習画像枚数）:\n")
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        count = len([
            f for f in os.listdir(cls_dir)
            if os.path.isfile(os.path.join(cls_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        cfg.write(f" - {cls}（{count}）\n")
