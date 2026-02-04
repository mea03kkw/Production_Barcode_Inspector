# Production Barcode Inspector

**Barcode Verification System for Production Line**

## Overview

This is a production line barcode verification system designed for workers. It allows you to:
1. **Calibrate** with a golden sample (correct) barcode
2. **Verify** production barcodes against the golden sample
3. Get instant **PASS/FAIL** feedback with visual indicators and **BEEP sound**

## Quick Start (For Non-IT Users)

### Option 1: Standalone Executable (Easiest - No Python Needed!)

1. **Double-click `build_exe.bat`** to create the standalone executable
2. Wait ~1-2 minutes for the build to complete
3. Find your executable at: `dist\production_barcode_inspector.exe`
4. **Copy this .exe to any computer** - no Python or installation required!

### Option 2: Run with Python

1. Install dependencies:
   ```bash
   pip install opencv-python numpy pyzbar pyyaml
   ```

2. **Double-click `run_inspector.bat`** to start the program

## How to Use

### Step 1: Calibration Mode (First Time)

1. The program starts in **CALIBRATION MODE** (orange header)
2. Show the **golden sample** (correct) barcode to the camera
3. When the barcode is detected, press **SPACE** to set it as the golden sample
4. The program automatically switches to **PRODUCTION MODE**

### Step 2: Production Mode

1. Show production barcodes to the camera
2. The system automatically verifies each barcode against the golden sample
3. **GREEN** = PASS (barcode matches golden sample) + BEEP
4. **RED** = FAIL (barcode doesn't match or quality is too low) + DOUBLE BEEP

## Controls

| Key | Action |
|-----|--------|
| **SPACE** | Set golden sample (in calibration mode) |
| **c** | Switch to Calibration mode |
| **p** | Switch to Production mode |
| **q** | Quit program |
| **s** | Save current frame to output folder |
| **r** | Reset golden sample (clear calibration) |

## Tips for Best Results

1. **Good Lighting**: Ensure even illumination without shadows
2. **Proper Distance**: Hold 20-40cm from camera
3. **Flat Angle**: Keep barcode flat (not skewed)
4. **Clear Focus**: Ensure barcode is in focus
5. **Clean Barcode**: Ensure barcode is not damaged or dirty

## Troubleshooting

**No barcode detected?**
- Increase lighting
- Ensure barcode is in frame
- Check camera connection
- Hold barcode closer to camera

**False FAIL results?**
- Ensure barcode is clean and undamaged
- Check lighting conditions
- Verify you're using the correct golden sample
- Try scanning at a different angle

**Need to change golden sample?**
- Press `r` to reset the golden sample
- Press `c` to switch to calibration mode
- Scan new golden sample and press `SPACE`

## Supported Barcode Types

**2D Barcodes:**
- QR Code, Data Matrix, Aztec, PDF417, MaxiCode

**1D Barcodes:**
- EAN-13, EAN-8, EAN-5, EAN-2
- UPC-A, UPC-E
- CODE128, CODE39, CODE93
- CODABAR, ITF

## File Descriptions

| File | Description |
|------|-------------|
| `production_barcode_inspector.py` | Main Python script |
| `run_inspector.bat` | Double-click to run with Python |
| `build_exe.bat` | Creates standalone .exe file |
| `dist\production_barcode_inspector.exe` | Standalone executable (after running build) |
| `output\` | Saved screenshots folder |
| `golden_sample.json` | Stored golden sample data |
| `config.yaml` | Optional configuration file |

## Advanced: Command Line Options

```bash
# Use different camera
python production_barcode_inspector.py --camera 1

# Use custom config
python production_barcode_inspector.py --config myconfig.yaml
```

## Example Workflow

```
1. Start program → CALIBRATION MODE
2. Show golden sample barcode
3. Press SPACE → PRODUCTION MODE
4. Scan production barcode #1 → PASS (green)
5. Scan production barcode #2 → FAIL (red)
6. Press 's' to save screenshot
7. Press 'q' to quit
```

## License

This project is for educational purposes.

---

**Happy Production Line Scanning! 🏭**
