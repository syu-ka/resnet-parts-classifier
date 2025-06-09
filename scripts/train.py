# このスクリプトは、data/train および data/val を使って ResNet18 を学習し、
# ../experiments_train/以下に日時付きのフォルダを作成して、学習済みモデルと設定情報（train_config.txt、train_images.txt）を保存します。

import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch import nn, optim
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import argparse

# --- BASE_DIR を scripts/ に設定 ---
BASE_DIR = Path(__file__).resolve().parent

# --- 引数処理 ---
parser = argparse.ArgumentParser(description="ResNet18 の学習スクリプト")
parser.add_argument("--seed", type=int, help="乱数シードを指定（例: --seed 42）")
parser.add_argument("--expname", help="学習実験名（例: bright_800lux）")
args = parser.parse_args()

# --- 乱数シードの設定（固定 or ランダム） ---
if args.seed is not None:
    print(f"🔒 乱数シードを {args.seed} に固定します")
else:
    args.seed = random.randint(0, 999999)
    print(f"🔄 シードが指定されていないため、ランダムに {args.seed} を使用します")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- パス設定 ---
train_dir = (BASE_DIR / "../data/train").resolve()
val_dir = (BASE_DIR / "../data/val").resolve()

# --- 出力先フォルダを作成 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
if args.expname:
    exp_dir = (BASE_DIR / f"../experiments_train/{timestamp}_{args.expname}").resolve()
else:
    exp_dir = (BASE_DIR / f"../experiments_train/{timestamp}").resolve()
exp_dir.mkdir(parents=True, exist_ok=True)

model_output_path = exp_dir / "resnet18.pth"
config_path = exp_dir / "train_config.txt"
images_list_path = exp_dir / "train_images.txt"

# --- クラス数（ディレクトリ名で自動取得） ---
classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
num_classes = len(classes)

# --- 前処理 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- データセットとデータローダ ---
train_dataset = datasets.ImageFolder(str(train_dir), transform=transform)
val_dataset = datasets.ImageFolder(str(val_dir), transform=transform)

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
with config_path.open("w", encoding="utf-8") as cfg:
    cfg.write(f"日時: {timestamp}\n")
    if args.expname:
        cfg.write(f"モデル名: {args.expname}\n")
    cfg.write(f"全クラス数: {num_classes}\n")
    cfg.write(f"エポック数: {num_epochs}\n")
    cfg.write(f"最適化手法: Adam (lr=0.001)\n")
    cfg.write(f"損失関数: CrossEntropyLoss\n")
    cfg.write(f"シード: {args.seed}\n")
    cfg.write("使用クラス（学習画像枚数）:\n")

    total_images = 0
    for cls in classes:
        cls_dir = train_dir / cls
        count = len([f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        cfg.write(f" - {cls}（{count}）\n")
        total_images += count
    cfg.write(f"全画像数: {total_images}\n")

# --- train_images.txt にファイル一覧を出力 ---
with images_list_path.open("w", encoding="utf-8") as f:
    for cls in classes:
        cls_dir = train_dir / cls
        for img in sorted(cls_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                f.write(str(img.resolve()) + "\n")
