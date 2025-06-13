# confidence score(信頼度スコア) のヒストグラム表示＋保存を行うスクリプト.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import sys

# 引数から実験フォルダを受け取る
if len(sys.argv) < 2:
    print("❌ 実験フォルダのパスを指定してください")
    sys.exit(1)

exp_dir = Path(sys.argv[1])
csv_path = exp_dir / "result_all.csv"

if not csv_path.exists():
    print(f"❌ CSVファイルが見つかりません: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

if "confidence" not in df.columns:
    print("❌ 'confidence' 列が存在しません")
    sys.exit(1)

mpl.rcParams['font.family'] = 'Meiryo'
plt.figure(figsize=(10, 6))
sns.histplot(df["confidence"], bins=20, kde=True)
plt.title("信頼度スコア分布")  # 元タイトル：Confidence Score Distribution
plt.xlabel("信頼度スコア")  # 元ラベル：Confidence Score
plt.ylabel("件\n数", rotation=0, labelpad=20)  # 元ラベル：Count # rotation=0:横書き # labelpad:軸との距離（外側にどれだけ離すか）
plt.grid(True)

output_path = exp_dir / "confidence_histogram.png"
plt.savefig(output_path)
print(f"✅ ヒストグラムを保存しました: {output_path}")
