# このスクリプトは、result_all.csv などのCSVファイルから混同行列を描画・保存します。
# ファイルのあるフォルダに confusion_matrix.png を保存し、同時に画面表示も行います。

from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 引数処理 ---
parser = argparse.ArgumentParser(description="CSVから混同行列を描画・保存")
parser.add_argument("csv_path", help="入力CSVファイル（例: experiments/20250609_debug/result_all.csv）")
args = parser.parse_args()

csv_path = Path(args.csv_path).resolve()
if not csv_path.exists():
    print(f"❌ CSVが見つかりません: {csv_path}")
    exit(1)

# --- CSV読み込み ---
df = pd.read_csv(csv_path)

if "true" not in df.columns or "pred" not in df.columns:
    print("❌ CSVに 'true' または 'pred' 列がありません")
    exit(1)

# --- 混同行列作成 ---
true_labels = df["true"].tolist()
pred_labels = df["pred"].tolist()
labels = sorted(set(true_labels) | set(pred_labels))  # クラス一覧

cm = confusion_matrix(true_labels, pred_labels, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# --- 描画・保存 ---
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()

output_path = csv_path.parent / "confusion_matrix_from_csv.png"
plt.savefig(output_path)
plt.show()

print(f"✅ 混同行列を保存しました: {output_path}")
