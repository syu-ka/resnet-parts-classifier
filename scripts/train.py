# raw_images から分割された data/train および data/val を使って ResNet18 を学習します。
# 学習済みモデルは ../models/resnet18.pth に保存されます。

import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch import nn, optim
import os

# --- パス設定 ---
train_dir = "../data/train"
val_dir = "../data/val"
model_output_path = "../models/resnet18.pth"

# --- クラス数（自動取得） ---
num_classes = len(os.listdir(train_dir))

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

# --- 学習ループ ---
num_epochs = 10
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
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
torch.save(model.state_dict(), model_output_path)
print(f"✅ モデル保存完了: {model_output_path}")
