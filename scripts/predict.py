# 単一の画像ファイルを指定して推論し、分類結果（クラス名）を出力します。
# コマンドライン引数に画像パスを指定して使用します。

import torch
from torchvision import models, transforms
from PIL import Image
import sys
import os

# --- クラス名（ImageFolderのルール：フォルダ名の辞書順） ---
classes = sorted([
    "magnet_clip_small",
    "magnet_clip_big",
    "clip",
    "ring",
    "hook_ring"
])

# 必ず辞書順に合わせる！
# 実際にモデルが使うラベル順は以下のようになります：
# label 0 → clip
# label 1 → hook_ring
# label 2 → magnet_clip_big
# label 3 → magnet_clip_small
# label 4 → ring

# --- モデルの準備 ---
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("../models/resnet18.pth", map_location="cpu"))
model.eval()

# --- 前処理 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- 画像ファイルの取得 ---
if len(sys.argv) < 2:
    print("画像ファイルのパスを指定してください。")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("指定されたファイルが存在しません。")
    sys.exit(1)

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# --- 推論 ---
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_label = classes[predicted.item()]
    print(f"予測結果: {predicted_label}")
