@echo off
chcp 65001 >nul
echo ==== 実験を3回繰り返します ====

for /L %%i in (1,1,3) do (
    echo ==== %%i 回目の実行 ====
    call run_experiment.bat
)

echo ==== 完了しました ====
pause
exit /b 0