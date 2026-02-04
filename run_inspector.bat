@echo off
REM Production Barcode Inspector - Run Script
REM Double-click to start the barcode inspector

REM Change to script directory
cd /d "%~dp0"

echo Starting Barcode Inspector...
echo.
echo Controls:
echo   SPACE = Set Golden Sample (Calibration mode)
echo   c     = Switch to Calibration mode
echo   p     = Switch to Production mode
echo   q     = Quit
echo   s     = Save current frame
echo   r     = Reset Golden Sample
echo.

python production_barcode_inspector.py

if errorlevel 1 (
    echo.
    echo Error: Python or dependencies may not be installed.
    echo Please run: pip install opencv-python numpy pyzbar pyyaml
    pause
)
