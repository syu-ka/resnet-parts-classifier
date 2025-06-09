# このスクリプトは、指定されたフォルダ（例: new_images）以下のクラス別フォルダの中身（画像ファイル）を全て削除します。
# フォルダ構造は残します。スクリプトのある scripts/ を起点とした絶対パスで動作します。

from pathlib import Path
import argparse

# --- スクリプトのあるディレクトリを基準に絶対パスを構築 ---
BASE_DIR = Path(__file__).resolve().parent

# --- 引数処理 ---
parser = argparse.ArgumentParser(description="指定フォルダ内のクラス別画像を削除（フォルダ構造は残す）")
parser.add_argument("target_folder", help="削除対象のルートフォルダ名（例: new_images）")
args = parser.parse_args()

target_root = (BASE_DIR / f"../{args.target_folder}").resolve()

if not target_root.exists() or not target_root.is_dir():
    print(f"❌ フォルダが存在しません: {target_root}")
    exit(1)

# --- 各クラスフォルダの中身を削除 ---
deleted_files = 0
for class_dir in sorted(target_root.iterdir()):
    if not class_dir.is_dir():
        continue
    for file in class_dir.iterdir():
        if file.is_file():
            file.unlink()
            deleted_files += 1

print(f"✅ 削除完了: {deleted_files} 個のファイルを削除しました（{target_root} 以下）")
