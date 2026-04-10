@echo off
title Painterly — server (close this window to stop)
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
echo Starting Gradio… browser may open automatically.
echo Close this window when you are done to stop the server.
echo.
python -m painterly_app.app
echo.
echo Server stopped.
pause
