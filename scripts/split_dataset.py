# raw_images/ 以下のクラス別画像を train/val に分割して data/ にコピーします。
# raw_images 側の画像は削除せず、train/val は実行時に初期化されます。

from pathlib import Path
import shutil
import random

# --- スクリプトのある場所を基準にルートを解決 ---
BASE_DIR = Path(__file__).resolve().parent

# --- 設定 ---
original_data_dir = (BASE_DIR / "../raw_images").resolve()  # 元画像（クラス別フォルダ）
output_dir = (BASE_DIR / "../data").resolve()               # 出力先（train/val 分割）
train_ratio = 0.8                                           # 学習用割合

random.seed(42)  # 再現性のある分割のために固定

# --- train / val フォルダを初期化 ---
for phase in ["train", "val"]:
    phase_dir = output_dir / phase
    if phase_dir.exists():
        try:
            shutil.rmtree(phase_dir)
        except Exception as e:
            print(f"❌ {phase_dir} の削除に失敗しました: {e}")
            continue
    phase_dir.mkdir(parents=True, exist_ok=True)

# --- 各クラスフォルダごとに分割 ---
for class_dir in original_data_dir.iterdir():
    if not class_dir.is_dir():
        continue  # ディレクトリ以外はスキップ

    images = sorted([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for phase, image_list in zip(["train", "val"], [train_images, val_images]):
        dest_dir = output_dir / phase / class_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_list:
            dst_path = dest_dir / img_path.name
            shutil.copy2(img_path, dst_path)

print("✅ 画像を train / val に分割しました（raw_images 側はそのまま残ります）")
