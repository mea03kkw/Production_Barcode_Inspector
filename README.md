# Production Barcode Inspector

**Barcode Verification System for Production Line**

## Overview

This is a production line barcode verification system designed for workers. It allows you to:
1. **Calibrate** with a golden sample (correct) barcode
2. **Verify** production barcodes against the golden sample
3. Get instant **PASS/FAIL** feedback with visual indicators and **BEEP sound**

## Single-Scan Workflow

The system uses a single-scan workflow optimized for production line:
- Each barcode is scanned **once** when detected
- **PASS** verification plays a **BEEP sound** (1000Hz, 200ms)
- System waits for barcode to be removed before scanning next one
- No continuous video capture - efficient for production line use

## Accuracy Improvements

The system includes advanced preprocessing for enhanced detection reliability:

1. **Bilateral Filter** - Preserves edges (crucial for 1D barcodes like EAN-13)
2. **Vertical Sobel Enhancement** - Dramatically improves decode rate for 1D barcodes
3. **Relaxed Quality Threshold** - 0.2 for 1D barcodes vs 0.5 for 2D barcodes
4. **Retry Logic** - 3 decode attempts per frame (10-25% improvement for noisy captures)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Program

```bash
python production_barcode_inspector.py
```

With a different camera:
```bash
python production_barcode_inspector.py --camera 1
```

## How to Use

### Step 1: Calibration Mode (First Time)

1. The program starts in **CALIBRATION MODE** (orange header)
2. Show the **golden sample** (correct) barcode to the camera
3. When the barcode is detected, press **SPACE** to set it as the golden sample
4. The program automatically switches to **PRODUCTION MODE**

### Step 2: Production Mode

1. Show production barcodes to the camera
2. The system automatically verifies each barcode against the golden sample
3. **GREEN** = PASS (barcode matches golden sample)
4. **RED** = FAIL (barcode doesn't match or quality is too low)

## Controls

| Key | Action |
|-----|--------|
| **SPACE** | Set golden sample (in calibration mode) |
| **c** | Switch to Calibration mode |
| **p** | Switch to Production mode |
| **q** | Quit program |
| **s** | Save current frame to output folder |
| **r** | Reset golden sample (clear calibration) |

## Screen Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ MODE: CALIBRATION / PRODUCTION                                  │
│ Scan GOLDEN SAMPLE barcode / Golden: [data]...                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    Camera View                                  │
│                                                                 │
│              [PASS/FAIL] - QRCODE                              │
│              [Barcode Data]                                      │
│              [Reason if failed]                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ SPACE = Set Golden Sample  c = Calibration  p = Production      │
│ q = Quit  s = Save Frame  r = Reset Golden Sample              │
└─────────────────────────────────────────────────────────────────┘
```

## Statistics Panel (Right Side)

- **FPS**: Current frame rate
- **Total Scanned**: Number of barcodes scanned
- **PASSED**: Number of barcodes that passed verification
- **FAILED**: Number of barcodes that failed verification
- **Pass Rate**: Percentage of passed barcodes

## What Makes a Barcode PASS or FAIL?

### PASS (Green)
- Barcode type matches golden sample
- Barcode data matches golden sample exactly
- Quality score meets threshold:
  - **1D barcodes**: ≥ 0.2 (relaxed for better detection)
  - **2D barcodes**: ≥ 0.5

### FAIL (Red)
- Type mismatch (e.g., QR Code vs Data Matrix)
- Data mismatch (different barcode content)
- Quality too low (poor scan quality)

**Note:** 1D barcodes (EAN, CODE128, etc.) use a relaxed quality threshold because polygon-based scoring was designed for 2D barcodes, not 1D barcodes.

## Tips for Workers

### Best Practices
1. **Good Lighting**: Ensure even illumination without shadows
2. **Proper Distance**: Hold 20-40cm from camera
3. **Flat Angle**: Keep barcode flat (not skewed)
4. **Clear Focus**: Ensure barcode is in focus
5. **Clean Barcode**: Ensure barcode is not damaged or dirty

### Troubleshooting

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

## Output Files

- **Saved frames**: Stored in `output/` folder
- **Golden sample**: Stored in `golden_sample.json`

## Supported Barcode Types

**2D Barcodes:**
- QR Code (QRCODE)
- Data Matrix (DATAMATRIX)
- Aztec (AZTEC)
- PDF417 (PDF417)
- MaxiCode (MAXICODE)

**1D Barcodes (with enhanced detection):**
- EAN-13, EAN-8, EAN-5, EAN-2
- UPC-A, UPC-E
- CODE128, CODE39, CODE93
- CODABAR, ITF

## Example Workflow

```
1. Start program → CALIBRATION MODE
2. Show golden sample barcode
3. Press SPACE → PRODUCTION MODE
4. Scan production barcode #1 → PASS (green)
5. Scan production barcode #2 → FAIL (red) - wrong data
6. Scan production barcode #3 → PASS (green)
7. Press 's' to save screenshot
8. Press 'q' to quit
```

## Configuration (Optional)

You can create a YAML config file to customize settings:

```yaml
# config.yaml
frame_width: 1280
frame_height: 720
min_barcode_size: 20
max_barcode_size: 1000
quality_threshold: 0.5
output_dir: 'output'
golden_sample_file: 'golden_sample.json'
```

Run with config:
```bash
python production_barcode_inspector.py --config config.yaml
```

## License

This project is for educational purposes.

---

**Happy Production Line Scanning! 🏭**
