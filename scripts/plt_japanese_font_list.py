# matplotlib が使える 日本語フォントのうち、一般的によく使われるものだけに絞って表示するスクリプト.

from matplotlib import font_manager

# 一般的な日本語フォント名のリスト（必要に応じて追加）
common_jp_fonts = [
    "Meiryo",                  # メイリオ（Windows）
    "Yu Gothic",               # 游ゴシック（Windows 10以降）
    "MS Gothic",               # MS ゴシック
    "MS Mincho",               # MS 明朝
    "Hiragino Maru Gothic Pro",  # ヒラギノ丸ゴ（macOS）
    "Hiragino Kaku Gothic Pro",  # ヒラギノ角ゴ（macOS）
    "TakaoGothic",             # Takaoゴシック（Linux）
    "Noto Sans CJK JP",        # GoogleのNotoフォント（Linuxなど）
    "IPAexGothic",             # IPAフォント（オープンソース）
]

# 実際に使用可能なフォントと照合
available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
usable_jp_fonts = [font for font in common_jp_fonts if font in available_fonts]

# 結果表示
print("使用可能な日本語フォント:")
for font in usable_jp_fonts:
    print(" -", font)

print("\n※上記のフォント名のいずれかを matplotlib の rcParams['font.family'] に設定すると日本語で表示することが出来るようになります。\n")