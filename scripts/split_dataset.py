# raw_images/ 以下のクラス別画像を train/val に分割して data/ にコピーします。
# raw_images 側の画像は削除せず、train/val は実行時に初期化されます。
# Windows環境の PermissionError に強い rmtree 実装を使用。

import os
import shutil
import random
from pathlib import Path
import stat

# --- スクリプトのあるディレクトリを基準に絶対パスを構築 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 設定（絶対パス） ---
original_data_dir = os.path.join(BASE_DIR, "../raw_images") # 元画像がある場所（クラス別フォルダ）
output_dir = os.path.join(BASE_DIR, "../data")              # 出力先（train/valに分かれる）
train_ratio = 0.8                                           # 学習用の割合（80%）

# ランダムな分割を毎回同じにするためのシード
random.seed(42)

# --- アクセス拒否対応：削除時に読み取り専用属性を解除 ---
def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# --- 既存の data/train, data/val を削除して空にする ---
for phase in ["train", "val"]:
    phase_dir = os.path.join(output_dir, phase)
    if os.path.exists(phase_dir):
        shutil.rmtree(phase_dir, onerror=handle_remove_readonly)
    os.makedirs(phase_dir, exist_ok=True)

# --- raw_images 内の各クラスごとに処理 ---
for class_name in os.listdir(original_data_dir):
    class_path = os.path.join(original_data_dir, class_name)

    # ディレクトリでなければスキップ（ファイル混在対策）
    if not os.path.isdir(class_path):
        continue

    # 画像ファイルを取得してシャッフル
    images = os.listdir(class_path)
    random.shuffle(images)

    # 分割位置を決定し、train/val に分ける
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # --- train・val フォルダに画像をコピー ---
    for phase, image_list in zip(["train", "val"], [train_images, val_images]):
        dest_dir = os.path.join(output_dir, phase, class_name)
        os.makedirs(dest_dir, exist_ok=True)  # クラスフォルダを作成

        for img_name in image_list:
            src_path = os.path.join(class_path, img_name)
            dest_path = os.path.join(dest_dir, img_name)

            shutil.copy2(src_path, dest_path)  # コピー（raw_images 側は削除しない）

print("✅ 画像を train / val に分割しました（raw_images 側はそのまま残ります）")
