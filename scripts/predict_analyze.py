# このスクリプトは、指定フォルダ（未記入だと../data/val）以下の画像に対して
# 画像分類モデルを用いて推論を行い、
# 各画像の正解ラベル（フォルダ名）と予測ラベルを比較して結果を出力します。
# 学習条件や撮影条件が異なる実験ごとに結果を整理して保存できるよう、
# --experiment オプションにより experiments/ 以下に日時付きの推論名フォルダを作成して、
# その中にconfig.txt、misclassified_images/、2種類のcsvが保存されます。
# --experiment オプションを使用しない場合は日時のみのフォルダになります。
# config.txt：実行時の設定や使用したモデルの学習条件を記録します。
# 使用するモデルは ../experiments_train/ 以下の最新の resnet18.pth です。
# misclassified_images/ フォルダ：誤分類された画像がここにコピー保存されます。
# 2種類のCSVファイル：
# - result_all.csv：全ての画像の結果を保存します。
# - result_wrong.csv：誤分類された画像のみの結果を保存します。

import torch
from torchvision import models, transforms
from PIL import Image
import os
import sys
import argparse
import csv
from datetime import datetime
import shutil
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F # モデルの出力をSoftmaxでスコアに変換するため

# --- スクリプトのあるディレクトリを基準に絶対パスを構築 ---
BASE_DIR = Path(__file__).resolve().parent

# --- パス設定（絶対パス） ---
train_dir = (BASE_DIR / "../data/train").resolve()
train_exp_dir = (BASE_DIR / "../experiments_train").resolve()

# --- クラス名を train_dir から取得 ---
classes = sorted(os.listdir(train_dir))
num_classes = len(classes)

# --- 最新の学習モデルを取得 ---
subdirs = [
    d for d in os.listdir(train_exp_dir)
    if os.path.isdir(os.path.join(train_exp_dir, d)) and d[:8].isdigit()
]
latest_train_dir = sorted(subdirs)[-1]
model_path = train_exp_dir / latest_train_dir / "resnet18.pth"

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
parser = argparse.ArgumentParser(description="画像分類の結果を表示・CSV出力（全件 + 誤分類）")
parser.add_argument("folder", nargs="?", default="../data/val", help="推論対象のフォルダパス（省略可）")
parser.add_argument("--expname", help="検証実験名を指定. experiments/ のサブフォルダ名（接尾辞）にもなる（例: --expname imageCount_100）")
args = parser.parse_args()
# --- args.folder を絶対パスに変換（引数で指定されても、されなくても） ---
args.folder = (BASE_DIR / args.folder).resolve()

# --- 入力フォルダ確認 ---
if not os.path.exists(args.folder) or not os.path.isdir(args.folder):
    print("❌ 指定されたフォルダが存在しないか、ディレクトリではありません")
    sys.exit(1)

# --- 実験フォルダ名の決定 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
if args.expname:
    exp_name = f"{timestamp}_{args.expname}"
else:
    exp_name = timestamp

exp_dir = (BASE_DIR / "../experiments" / exp_name).resolve()
os.makedirs(exp_dir, exist_ok=True)

# --- 誤分類画像保存用フォルダを作成 ---
misclassified_dir = exp_dir / "misclassified_images"
os.makedirs(misclassified_dir, exist_ok=True)

# --- 推論処理開始 ---
results = []
total = 0
correct = 0
print(f"\n📊 検証結果（{args.folder}）:")
for root, _, files in os.walk(args.folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = Path(root) / filename
            true_label = true_label = img_path.parent.name

            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image)
                probs = F.softmax(outputs, dim=1)
                confidence = probs.max().item()
                _, predicted = torch.max(outputs, 1)
                predicted_label = classes[predicted.item()]

            is_correct = (predicted_label == true_label)
            total += 1
            if is_correct:
                correct += 1

            if not is_correct:
                dst_path = misclassified_dir / img_path.name
                shutil.copy2(img_path, dst_path)

            results.append({
                "path": img_path,
                "true": true_label,
                "pred": predicted_label,
                "correct": is_correct,
                "confidence": round(confidence, 3)  # 小数第3位までに丸めて見やすく
            })


# --- 混同行列を作成・保存 ---
true_labels = [r["true"] for r in results]
pred_labels = [r["pred"] for r in results]

if true_labels and pred_labels:
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # 保存
    confusion_path = os.path.join(exp_dir, "confusion_matrix.png")
    plt.savefig(confusion_path)
    print(f"🖼️ 混同行列を保存しました: {confusion_path}")

    # 表示
    # plt.show()
else:
    print("⚠️ 混同行列の作成に必要なデータが不足しています")


# --- 表示 ---
for r in results:
    mark = "✅" if r["correct"] else "❌"
    print(f"{mark} {Path(r['path']).as_posix()} | 正解: {r['true']} | 予測: {r['pred']} | 確信度: {r['confidence']:.3f}")

# --- 統計 ---
accuracy = 100 * correct / total if total else 0
print(f"\n🎯 正解数: {correct}/{total} (正解率: {accuracy:.2f}%)")

# --- config.txt を保存 ---
config_path = os.path.join(exp_dir, "config.txt")
with open(config_path, "w", encoding="utf-8") as cfg:
    if args.expname:
        cfg.write(f"検証実験名: {exp_name}\n")
    else:
        cfg.write(f"検証実験名: {exp_name}（自動命名）\n")
    cfg.write(f"日時: {timestamp}\n")
    cfg.write(f"正解数: {correct}/{total} (正解率: {accuracy:.2f}%)\n")
    cfg.write(f"検証対象: {args.folder}\n")
    cfg.write("出力ファイル:\n")
    cfg.write(" - result_all.csv（全件）\n")
    cfg.write(" - result_wrong.csv（誤分類）\n")
    cfg.write(f"全クラス数: {num_classes}\n")
    cfg.write("使用クラス(検証画像枚数):\n")
    total_predict_images = 0
    for cls in classes:
        predict_path = os.path.join(args.folder, cls)
        predict_count = len([
            f for f in os.listdir(predict_path)
            if os.path.isfile(os.path.join(predict_path, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]) if os.path.exists(predict_path) else 0
        total_predict_images += predict_count
        cfg.write(f" - {cls} ({predict_count})\n")
    cfg.write(f"全推論画像数: {total_predict_images} 枚\n")

    shooting_path = (BASE_DIR / "../config/shooting.txt").resolve()
    if os.path.exists(shooting_path):
        cfg.write("\n撮影条件:\n")
        with open(shooting_path, "r", encoding="utf-8") as shoot:
            for line in shoot:
                cfg.write(f" - {line.strip()}\n")
    else:
        cfg.write("\n撮影条件: shooting.txt が見つかりません\n")

    # --- 使用モデル ---
    train_config_path = os.path.join(train_exp_dir, latest_train_dir, "train_config.txt")
    if os.path.exists(train_config_path):
        cfg.write("\n使用モデルの詳細:\n")
        with open(train_config_path, "r", encoding="utf-8") as tcf:
            for line in tcf:
                cfg.write(f" - {line.strip()}\n")
    else:
        cfg.write("\n使用モデルの詳細: train_config.txt が見つかりません\n")

# --- CSV（全件）保存 ---
output_all = exp_dir / "result_all.csv"
with open(output_all, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "true", "pred", "correct", "confidence"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# --- CSV（誤分類のみ）保存 ---
output_wrong = exp_dir / "result_wrong.csv"
with open(output_wrong, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "true", "pred", "correct", "confidence"])
    writer.writeheader()
    for r in results:
        if not r["correct"]:
            writer.writerow(r)

print(f"📁 CSV出力完了: {output_all}, {output_wrong}")
