# 指定フォルダ以下の画像に対して推論を行い、正解ラベル（フォルダ名）と比較して表示します。
# クラス名は ../data/train/ から取得（辞書順）

import torch
from torchvision import models, transforms
from PIL import Image
import os
import sys

# --- クラス名取得（../data/train のフォルダ名） ---
train_dir = "../data/train"
classes = sorted(os.listdir(train_dir))  # ImageFolderと同じラベル順

# --- モデル準備（ResNet18） ---
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("../models/resnet18.pth", map_location="cpu"))
model.eval()

# --- 前処理定義（学習時と同じ） ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- 実行引数チェック ---
if len(sys.argv) < 2:
    print("📂 推論対象のフォルダパスを指定してください")
    sys.exit(1)

folder_path = sys.argv[1]

if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
    print("❌ 指定されたフォルダが存在しないか、ディレクトリではありません")
    sys.exit(1)

# --- 推論実行 ---
print(f"\n📊 推論と正解比較（{folder_path} 以下）:")
correct_count = 0
total_count = 0

for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, filename)

            # --- 正解ラベル（親フォルダ名）を取得 ---
            true_label_name = os.path.basename(os.path.dirname(img_path))

            # --- 画像読み込みと前処理 ---
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0)

            # --- 推論 ---
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_label_name = classes[predicted.item()]

            # --- 表示 ---
            mark = "✅" if predicted_label_name == true_label_name else "❌"
            print(f"{mark} {img_path} | 正解: {true_label_name} | 予測: {predicted_label_name}")

            # --- カウント更新 ---
            total_count += 1
            if predicted_label_name == true_label_name:
                correct_count += 1

# --- 集計表示 ---
accuracy = 100 * correct_count / total_count if total_count else 0
print(f"\n🎯 正解数: {correct_count}/{total_count}（正解率: {accuracy:.2f}%）")
