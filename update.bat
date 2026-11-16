@echo off
rem Double-click to refresh NBA data, log forecasts and publish to GitHub (Streamlit redeploys by itself).
cd /d "%~dp0"

echo [1/3] Downloading new results and logging forecasts...
".venv\Scripts\python.exe" -m nba_predictor update
if errorlevel 1 (
    echo.
    echo UPDATE FAILED. Nothing was published. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [2/3] Saving changes...
git add -A
git diff --cached --quiet
if not errorlevel 1 (
    echo Nothing new since the last update.
    pause
    exit /b 0
)
git commit -q -m "Update data %date%"

echo.
echo [3/3] Publishing to GitHub...
git push
if errorlevel 1 (
    echo.
    echo PUSH FAILED. The data is updated on this computer but not online. Run "git push" to try again.
    pause
    exit /b 1
)

echo.
echo Done. The website refreshes itself in a minute or two.
pause
