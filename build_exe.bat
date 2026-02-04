@echo off
REM Build Standalone Executable - Creates .exe with no Python needed!

REM Change to script directory
cd /d "%~dp0"

echo ========================================
echo   Building Standalone Executable
echo ========================================
echo Working directory: %CD%
echo.
echo [1/2] Installing PyInstaller...
pip install pyinstaller >nul 2>&1
echo       Done.

echo.
echo [2/2] Building executable...
echo       (This takes 1-2 minutes, please wait...)
python -m PyInstaller --onefile --noconsole --collect-all pyzbar production_barcode_inspector.py

if exist "dist\production_barcode_inspector.exe" (
    echo.
    echo ========================================
    echo   Build Complete!
    echo ========================================
    echo.
    echo Your executable is ready at:
    echo   %CD%\dist\production_barcode_inspector.exe
    echo.
    echo Copy this .exe to any computer - no Python required!
) else (
    echo.
    echo [ERROR] Build failed! Check errors above.
)

echo.
pause
