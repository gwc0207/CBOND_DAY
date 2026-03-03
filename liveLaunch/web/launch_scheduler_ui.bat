@echo off
cd /d %~dp0\..\..
start "" pythonw -m liveLaunch.web.app
timeout /t 1 /nobreak >nul
start "" http://127.0.0.1:5003

