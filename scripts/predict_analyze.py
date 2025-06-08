# このスクリプトは、指定フォルダ（未記入だと../data/val）以下の画像に対して
# 画像分類モデルを用いて推論を行い、
# 各画像の正解ラベル（フォルダ名）と予測ラベルを比較して結果を出力します。
# 全件表示・誤分類のみ表示・CSV出力に対応しており、学習条件や撮影条件が異なる実験ごとに
# 結果を整理して保存できるよう、--experiment オプションにより experiments/ 以下に
# 実験名付きのフォルダを作成して保存されます。
# --experiment オプションを使用しない場合は日付＋時刻による自動命名、
# 使用した場合はそこで指定した実験名（例: light_normal）に
# 日付＋時刻が自動で先頭に付けられます（例: 20250604_2338_light_normal）。
# また、実行時の設定や使用クラスを記録した config.txt が実験フォルダに保存され、
# 誤分類された画像は misclassified_images/ フォルダにコピー保存されます。
# 使用するモデルは ../experiments_train/ 以下の最新の resnet18.pth を自動で読み込み、
# その学習条件（train_config.txt）も config.txt に記録されます。

import torch
from torchvision import models, transforms
from PIL import Image
import os
import sys
import argparse
import csv
from datetime import datetime
import shutil

# --- クラス名を ../data/train から取得 ---
train_dir = "../data/train"
classes = sorted(os.listdir(train_dir))

# --- 最新の学習モデルを取得 ---
train_exp_dir = "../experiments_train"
subdirs = [
    d for d in os.listdir(train_exp_dir)
    if os.path.isdir(os.path.join(train_exp_dir, d)) and d[:8].isdigit()
]
latest_train_dir = sorted(subdirs)[-1]
model_path = os.path.join(train_exp_dir, latest_train_dir, "resnet18.pth")

# --- モデル準備 ---
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# --- 前処理 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- 引数処理 ---
parser = argparse.ArgumentParser(description="画像分類の結果を表示・CSV出力")
parser.add_argument("folder", nargs="?", default="../data/val", help="推論対象のフォルダパス")
parser.add_argument("--filter", choices=["wrong"], help="誤分類のみ表示")
parser.add_argument("--csv", default="result.csv", help="保存するCSVファイル名（例: result.csv）")
parser.add_argument("--experiment", help="保存先 experiments/ のサブフォルダ名（例: 20240603_light_normal）")
args = parser.parse_args()

# --- 入力フォルダ確認 ---
if not os.path.exists(args.folder) or not os.path.isdir(args.folder):
    print("❌ 指定されたフォルダが存在しないか、ディレクトリではありません")
    sys.exit(1)

# --- 実験フォルダ名の決定 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
if args.experiment:
    exp_name = f"{timestamp}_{args.experiment}"
else:
    exp_name = timestamp

exp_dir = os.path.join("../experiments", exp_name)
os.makedirs(exp_dir, exist_ok=True)

# --- 誤分類画像保存用フォルダを作成 ---
misclassified_dir = os.path.join(exp_dir, "misclassified_images")
os.makedirs(misclassified_dir, exist_ok=True)

# --- 推論処理開始 ---
results = []
total = 0
correct = 0
print(f"\n📊 推論結果（{args.folder}）:")
for root, _, files in os.walk(args.folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, filename)
            true_label = os.path.basename(os.path.dirname(img_path))

            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_label = classes[predicted.item()]

            is_correct = (predicted_label == true_label)
            total += 1
            if is_correct:
                correct += 1

            if not is_correct:
                dst_path = os.path.join(misclassified_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dst_path)

            results.append({
                "path": img_path,
                "true": true_label,
                "pred": predicted_label,
                "correct": is_correct,
            })

# --- 表示 ---
for r in results:
    if args.filter == "wrong" and r["correct"]:
        continue
    mark = "✅" if r["correct"] else "❌"
    print(f"{mark} {r['path']} | 正解: {r['true']} | 予測: {r['pred']}")

# --- 統計 ---
accuracy = 100 * correct / total if total else 0
print(f"\n🎯 正解数: {correct}/{total} (正解率: {accuracy:.2f}%)")

# --- config.txt を保存 ---
config_path = os.path.join(exp_dir, "config.txt")
with open(config_path, "w", encoding="utf-8") as cfg:
    cfg.write(f"日時: {timestamp}\n")
    cfg.write(f"正解数: {correct}/{total} (正解率: {accuracy:.2f}%)\n")
    cfg.write(f"推論対象: {args.folder}\n")
    cfg.write(f"フィルタ: {'誤分類のみ' if args.filter == 'wrong' else '全件'}\n")
    cfg.write(f"CSVファイル名: {args.csv}\n")
    cfg.write(" 使用クラス(推論画像枚数):\n")
    for cls in classes:
        predict_path = os.path.join(args.folder, cls)
        predict_count = len([
            f for f in os.listdir(predict_path)
            if os.path.isfile(os.path.join(predict_path, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]) if os.path.exists(predict_path) else 0

        cfg.write(f" - {cls} ({predict_count})\n")

    shooting_path = os.path.join("..", "config", "shooting.txt")
    if os.path.exists(shooting_path):
        cfg.write("\n撮影条件:\n")
        with open(shooting_path, "r", encoding="utf-8") as shoot:
            cfg.write(f"{shoot.read()}\n")
    else:
        cfg.write("\n撮影条件: shooting.txt が見つかりません\n")

    # --- 使用モデルの学習条件 ---
    train_config_path = os.path.join(train_exp_dir, latest_train_dir, "train_config.txt")
    if os.path.exists(train_config_path):
        cfg.write("\n使用モデルの学習条件:\n")
        with open(train_config_path, "r", encoding="utf-8") as f:
            cfg.write(f.read())
    else:
        cfg.write("\n使用モデルの学習条件: train_config.txt が見つかりません\n")

# --- CSV保存 ---
output_path = os.path.join(exp_dir, args.csv)
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "true", "pred", "correct"])
    writer.writeheader()
    for r in results:
        if args.filter == "wrong" and r["correct"]:
            continue
        writer.writerow(r)

print(f"📁 CSV出力完了: {output_path}")
