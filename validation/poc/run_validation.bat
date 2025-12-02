@echo off
cd /d "%~dp0"
echo Running validation at %date% %time% > ..\results\run_log.txt
python run_all_validation.py >> ..\results\run_log.txt 2>&1
echo Exit code: %ERRORLEVEL% >> ..\results\run_log.txt
echo Done. Check results\run_log.txt

