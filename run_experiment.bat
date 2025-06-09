@REM split_dataset.py, train.py, predict_analyze.py を順に実行するバッチスクリプト
@REM 実験名と乱数シードを指定可能

@echo off
@REM スクリプトのエンコーディングを UTF-8 に設定
chcp 65001 >nul

REM --- 実験名の指定（空白の場合は省略） ---
set EXP_NAME=imageCount_each_5_light_flash_magnificationRate_3

REM --- 乱数シードの指定（空白の場合は省略） ---
@REM set SEED=42

REM --- 日時を取得（YYYYMMDD_HHMM形式） ---
for /f %%i in ('powershell -command "Get-Date -Format yyyyMMdd_HHmm"') do set TIMESTAMP=%%i

REM --- 1. データを train/val に分割 ---
echo 🔄 [%TIMESTAMP%] split_dataset.py を実行中...
python scripts\split_dataset.py
if errorlevel 1 goto :error

REM --- 2. 学習（--seed および --expname） ---
echo 🧠 [%TIMESTAMP%] train.py を実行中...
if not "%SEED%"=="" (
    python scripts\train.py --seed %SEED% --expname %EXP_NAME%
) else (
    python scripts\train.py --expname %EXP_NAME%
)
if errorlevel 1 goto :error

REM --- 3. 検証（--expname） ---
echo 🔍 [%TIMESTAMP%] predict_analyze.py を実行中...
python scripts\predict_analyze.py ../data/val --expname %EXP_NAME%
if errorlevel 1 goto :error

echo ✅ 実験完了 [%TIMESTAMP%]
goto :eof

:error
echo ❌ エラーが発生しました。スクリプトを中断します。
pause
exit /b 1
