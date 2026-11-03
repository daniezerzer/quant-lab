@echo off
rem Runs the tests and all analysis scripts.
rem Everything printed is also saved to results.txt
cd /d "%~dp0"
call .venv\Scripts\activate.bat
(
  echo === pytest ===
  python -m pytest
  echo.
  echo === compare ===
  python scripts\compare.py
  echo.
  echo === sweep ===
  python scripts\sweep.py
  echo.
  echo === robustness ===
  python scripts\robustness.py
) > results.txt 2>&1
type results.txt
echo.
echo Done. Results are saved in results.txt, charts are in the docs folder.
pause
