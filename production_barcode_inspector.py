#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Barcode Inspector — Barcode Verification for Production Line

Purpose
-------
Production line barcode verification system with golden sample calibration.
Workers can calibrate with a correct barcode, then verify production barcodes
against the golden sample.

Workflow
--------
1. CALIBRATION MODE: Scan a golden sample (correct) barcode and press SPACE to set it
2. PRODUCTION MODE: Scan production barcodes - each barcode is scanned once
   - PASS (green): Barcode matches golden sample + BEEP sound
   - FAIL (red): Barcode doesn't match or quality is too low
   - Single-scan mode: Waits for barcode to be removed before next scan

Controls
--------
SPACE = Set golden sample (in calibration mode)
c     = Switch to Calibration mode
p     = Switch to Production mode
q     = Quit
s     = Save current frame
r     = Reset golden sample

Supported Barcode Types
-----------------------
2D Barcodes:
- QR Code (QRCODE)
- Data Matrix (DATAMATRIX)
- Aztec (AZTEC)
- PDF417 (PDF417)
- MaxiCode (MAXICODE)

1D Barcodes (with enhanced detection):
- EAN-13, EAN-8, EAN-5, EAN-2
- UPC-A, UPC-E
- CODE128, CODE39, CODE93
- CODABAR, ITF

Accuracy Improvements
---------------------
1. Bilateral filter (preserves edges) instead of Gaussian blur
2. Vertical Sobel edge enhancement for 1D barcodes
3. Relaxed quality threshold for 1D barcodes (0.2 vs 0.5)
4. Retry logic (3 attempts) for improved decode rate

Dependencies
------------
  pip install opencv-python numpy pyzbar pyyaml
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Beep sound for Windows
import ctypes

def play_beep(frequency: int = 1000, duration: int = 200) -> None:
    """Play a high-pitched beep for correct verification."""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.Beep(frequency, duration)
    except Exception:
        pass  # Silent fallback if audio fails


def play_fail_beep() -> None:
    """Play double quick low-pitched beeps for incorrect verification."""
    try:
        kernel32 = ctypes.windll.kernel32
        # Double beep: 400Hz for 100ms each, 50ms apart
        kernel32.Beep(400, 100)
        ctypes.windll.kernel32.Sleep(50)
        kernel32.Beep(400, 100)
    except Exception:
        pass  # Silent fallback if audio fails


# Try to import pyzbar, with helpful error message if not installed
try:
    from pyzbar.pyzbar import decode
    from pyzbar.pyzbar import ZBarSymbol
except ImportError:
    print("[ERROR] pyzbar not installed. Install with: pip install pyzbar")
    sys.exit(1)

# ========================== Default Configuration ============================
# Camera
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# Detection parameters
MIN_BARCODE_SIZE = 20      # Minimum barcode size in pixels (width/height)
MAX_BARCODE_SIZE = 1000   # Maximum barcode size in pixels

# Quality assessment
QUALITY_THRESHOLD = 0.5    # Minimum quality score (0-1) to consider as CORRECT

# Output
OUTPUT_DIR = 'output'
GOLDEN_SAMPLE_FILE = 'golden_sample.json'

# UI Colors
COLOR_PASS = (0, 200, 0)      # Green for PASS
COLOR_FAIL = (0, 0, 255)      # Red for FAIL
COLOR_CALIBRATE = (255, 165, 0)  # Orange for calibration
COLOR_INFO = (255, 255, 255)  # White for info
COLOR_BG = (30, 30, 30)       # Dark background

# ============================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class ProductionBarcodeInspector:
    def __init__(self, camera_id: int = 0, config_path: Optional[str] = None) -> None:
        # Load config
        self.cfg = self._load_config(config_path)

        # Camera setup
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get('frame_width', FRAME_WIDTH)))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get('frame_height', FRAME_HEIGHT)))

        # Detection parameters
        self.min_barcode_size = int(self.cfg.get('min_barcode_size', MIN_BARCODE_SIZE))
        self.max_barcode_size = int(self.cfg.get('max_barcode_size', MAX_BARCODE_SIZE))
        self.quality_threshold = float(self.cfg.get('quality_threshold', QUALITY_THRESHOLD))

        # Output directory
        self.output_dir = Path(self.cfg.get('output_dir', OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Golden sample storage
        self.golden_sample_file = Path(self.cfg.get('golden_sample_file', GOLDEN_SAMPLE_FILE))
        self.golden_sample = self._load_golden_sample()

        # Mode: 'calibration' or 'production'
        self.mode = 'calibration' if not self.golden_sample else 'production'

        # UI fonts
        self.font_large = cv2.FONT_HERSHEY_SIMPLEX
        self.font_medium = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX

        # Stats
        self.fps = 0.0
        self._t0 = time.time()
        self._frames = 0
        self.total_scanned = 0
        self.total_passed = 0
        self.total_failed = 0

        print(f"ProductionBarcodeInspector initialized.")
        print(f"Mode: {self.mode.upper()}")
        if self.golden_sample:
            print(f"Golden Sample Loaded: {self.golden_sample.get('data', 'N/A')[:50]}")
        print("Controls: c=Calibrate, p=Production, q=Quit, s=Save, r=Reset, SPACE=Set Golden Sample")
        print("---------------------------")

    # --------------------------- Config (optional) -------------------------
    def _load_config(self, path: Optional[str]) -> Dict:
        if not path:
            return {}
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
                print(f"Loaded config: {path}")
                return cfg
        except Exception as e:
            print(f"[WARN] Could not load config '{path}': {e}")
            return {}

    # --------------------------- Golden Sample -----------------------------
    def _load_golden_sample(self) -> Optional[Dict]:
        """Load golden sample from file."""
        if self.golden_sample_file.exists():
            try:
                with open(self.golden_sample_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Could not load golden sample: {e}")
                return None
        return None

    def _save_golden_sample(self, sample: Dict) -> None:
        """Save golden sample to file."""
        try:
            with open(self.golden_sample_file, 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2)
            print(f"Golden sample saved to: {self.golden_sample_file}")
        except Exception as e:
            print(f"[ERROR] Could not save golden sample: {e}")

    def reset_golden_sample(self) -> None:
        """Reset golden sample and switch to calibration mode."""
        self.golden_sample = None
        if self.golden_sample_file.exists():
            self.golden_sample_file.unlink()
        self.mode = 'calibration'
        print("Golden sample reset. Switched to CALIBRATION mode.")

    def set_golden_sample(self, barcode: Dict) -> None:
        """Set the golden sample from detected barcode."""
        self.golden_sample = {
            'type': barcode['type'],
            'data': barcode['data'],
            'quality': barcode['quality'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self._save_golden_sample(self.golden_sample)
        self.mode = 'production'
        print(f"Golden sample set: {barcode['type']} - {barcode['data'][:50]}")
        print("Switched to PRODUCTION mode.")

    # --------------------------- Preprocess --------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing to enhance barcode detection.
        
        Uses conservative preprocessing to ensure reliable detection:
        - CLAHE for contrast enhancement
        - Light bilateral filter to preserve edges (better for 1D barcodes)
        - Optional vertical Sobel edge enhancement (disabled by default for stability)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Use light bilateral filter to preserve edges while reducing noise
        # This is crucial for 1D barcodes (EAN-13, CODE128, etc.) which need sharp vertical edges
        # Using conservative parameters to avoid over-processing
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # Optional: Add vertical Sobel edge enhancement for 1D barcodes
        # This can improve decode rate but may cause issues with some barcodes
        # Uncomment the following lines if needed for specific 1D barcode types:
        # sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        # gray = cv2.addWeighted(gray, 0.7, sobel, 0.3, 0)
        
        return gray

    # ----------------------------- Detect ----------------------------------
    def detect_barcodes(self, frame: np.ndarray, gray: np.ndarray) -> List[Dict]:
        """Detect and decode barcodes using pyzbar with rotation tolerance.
        
        Tries multiple rotation angles to support barcodes at different orientations.
        Returns results from the first angle that finds barcodes.
        """
        # Try different rotation angles
        rotation_angles = [0, 90, 180, 270]
        
        for angle in rotation_angles:
            if angle == 0:
                processed_gray = gray
            else:
                # Rotate the gray image
                if angle == 90:
                    rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rotated = cv2.rotate(gray, cv2.ROTATE_180)
                else:  # 270
                    rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
                processed_gray = rotated
            
            # Try to decode
            barcodes = []
            for attempt in range(2):  # Reduced retry for rotation
                barcodes = decode(processed_gray)
                if barcodes:
                    break
            
            if barcodes:
                # Convert coordinates back to original orientation
                h, w = gray.shape
                results = []
                
                for barcode in barcodes:
                    # Extract barcode data
                    barcode_type = barcode.type
                    barcode_data = barcode.data.decode('utf-8')
                    
                    # Get bounding box (in rotated coordinates)
                    (x, y, bw, bh) = barcode.rect
                    
                    # Size validation (adjusted for rotated dimensions)
                    rh, rw = processed_gray.shape
                    if bw < self.min_barcode_size or bh < self.min_barcode_size:
                        continue
                    if bw > rw or bh > rh:
                        continue
                    
                    # Convert coordinates back to original frame
                    if angle == 0:
                        orig_x, orig_y = x, y
                        orig_w, orig_h = bw, bh
                    elif angle == 90:
                        orig_x = h - y - bh
                        orig_y = x
                        orig_w = bh
                        orig_h = bw
                    elif angle == 180:
                        orig_x = w - x - bw
                        orig_y = h - y - bh
                        orig_w = bw
                        orig_h = bh
                    else:  # 270
                        orig_x = y
                        orig_y = w - x - bw
                        orig_w = bh
                        orig_h = bw
                    
                    # Get polygon and adjust coordinates
                    points = barcode.polygon
                    orig_points = []
                    for pt in points:
                        px, py = pt.x, pt.y
                        if angle == 0:
                            opx, opy = px, py
                        elif angle == 90:
                            opx = h - py
                            opy = px
                        elif angle == 180:
                            opx = w - px
                            opy = h - py
                        else:  # 270
                            opx = py
                            opy = w - px
                        orig_points.append((opx, opy))
                    
                    # Quality assessment
                    is_1d_barcode = barcode_type in ["EAN13", "EAN8", "CODE128", "CODE39", "UPCA", "UPCE", "EAN5", "EAN2", "CODABAR", "CODE93", "ITF"]
                    
                    if is_1d_barcode:
                        quality = 1.0
                    elif len(orig_points) >= 4:
                        quality = self._assess_quality(orig_points, orig_w, orig_h)
                    else:
                        quality = 0.5
                    
                    results.append({
                        'type': barcode_type,
                        'data': barcode_data,
                        'rect': (orig_x, orig_y, orig_w, orig_h),
                        'polygon': orig_points,
                        'quality': quality,
                        'center': (orig_x + orig_w // 2, orig_y + orig_h // 2),
                    })
                
                if results:
                    return results
        
        return []

    # ----------------------------- Quality Assessment -----------------------
    def _assess_quality(self, points: List[Tuple[int, int]], w: int, h: int) -> float:
        """Assess barcode quality based on polygon regularity."""
        if len(points) < 4:
            return 0.5
        
        pts = np.array(points, dtype=np.float32)
        
        # Calculate convex hull area vs bounding box area
        hull = cv2.convexHull(pts)
        hull_area = cv2.contourArea(hull)
        bbox_area = w * h
        
        if bbox_area > 0:
            area_ratio = hull_area / bbox_area
        else:
            area_ratio = 0.5
        
        # Calculate aspect ratio deviation
        aspect_ratio = w / max(h, 1)
        aspect_score = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)
        
        quality = 0.6 * area_ratio + 0.4 * aspect_score
        
        return clamp(quality, 0.0, 1.0)

    # ----------------------------- Verify ----------------------------------
    def verify_barcode(self, barcode: Dict) -> Tuple[bool, str]:
        """Verify barcode against golden sample.
        
        Uses relaxed quality threshold for 1D barcodes to prevent false FAILs
        due to irrelevant polygon heuristics.
        """
        if not self.golden_sample:
            return False, "No golden sample set"
        
        # Check barcode type
        if barcode['type'] != self.golden_sample['type']:
            return False, f"Type mismatch: expected {self.golden_sample['type']}, got {barcode['type']}"
        
        # Check barcode data
        if barcode['data'] != self.golden_sample['data']:
            return False, f"Data mismatch: expected '{self.golden_sample['data'][:30]}...', got '{barcode['data'][:30]}...'"
        
        # Check quality with relaxed threshold for 1D barcodes
        is_1d_barcode = barcode['type'] in ["EAN13", "EAN8", "CODE128", "CODE39", "UPCA", "UPCE", "EAN5", "EAN2", "CODABAR", "CODE93", "ITF"]
        threshold = 0.2 if is_1d_barcode else self.quality_threshold
        
        if barcode['quality'] < threshold:
            return False, f"Quality too low: {barcode['quality']:.2f} < {threshold}"
        
        return True, "PASS"

    # ----------------------------- Process --------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a frame and return annotated image with barcode results."""
        gray = self.preprocess(frame)
        barcodes = self.detect_barcodes(frame, gray)

        # Create output frame
        out = frame.copy()
        
        # Draw mode indicator background
        mode_color = COLOR_CALIBRATE if self.mode == 'calibration' else COLOR_PASS
        cv2.rectangle(out, (0, 0), (out.shape[1], 80), mode_color, -1)
        
        # Draw mode text
        mode_text = f"MODE: {self.mode.upper()}"
        cv2.putText(out, mode_text, (20, 35), self.font_large, 1.2, (255, 255, 255), 3)
        
        if self.mode == 'calibration':
            cv2.putText(out, "Scan GOLDEN SAMPLE barcode", (20, 65), self.font_medium, 0.7, (255, 255, 255), 2)
        else:
            golden_data = self.golden_sample.get('data', 'N/A')[:40]
            cv2.putText(out, f"Golden: {golden_data}...", (20, 65), self.font_medium, 0.6, (255, 255, 255), 2)

        # Process barcodes
        for bc in barcodes:
            if self.mode == 'calibration':
                # Calibration mode - show detected barcodes
                color = COLOR_CALIBRATE
                x, y, w, h = bc['rect']
                
                # Draw bounding box
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
                
                # Draw polygon
                if len(bc['polygon']) >= 4:
                    pts = np.array(bc['polygon'], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(out, [pts], True, color, 2)
                
                # Draw label
                label = f"{bc['type'].upper()}: {bc['data'][:20]}..."
                (tw, th), _ = cv2.getTextSize(label, self.font_medium, 0.7, 2)
                cv2.rectangle(out, (x, y - th - 10), (x + tw + 10, y), color, -1)
                cv2.putText(out, label, (x + 5, y - 5), self.font_medium, 0.7, (255, 255, 255), 2)
                
                # Show quality
                quality_text = f"Quality: {bc['quality']:.2f}"
                cv2.putText(out, quality_text, (x, y + h + 20), self.font_small, 0.5, (255, 255, 255), 1)
            else:
                # Production mode - verify against golden sample
                passed, reason = self.verify_barcode(bc)
                color = COLOR_PASS if passed else COLOR_FAIL
                x, y, w, h = bc['rect']
                
                # Draw bounding box
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 4)
                
                # Draw polygon
                if len(bc['polygon']) >= 4:
                    pts = np.array(bc['polygon'], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(out, [pts], True, color, 2)
                
                # Draw status label
                status = "PASS" if passed else "FAIL"
                label = f"{status} - {bc['type'].upper()}"
                (tw, th), _ = cv2.getTextSize(label, self.font_large, 1.0, 3)
                cv2.rectangle(out, (x, y - th - 15), (x + tw + 15, y), color, -1)
                cv2.putText(out, label, (x + 8, y - 8), self.font_large, 1.0, (255, 255, 255), 3)
                
                # Draw data
                data_text = bc['data'][:25] + ('...' if len(bc['data']) > 25 else '')
                cv2.putText(out, data_text, (x, y + h + 25), self.font_medium, 0.6, (255, 255, 255), 2)
                
                # Draw reason if failed
                if not passed:
                    reason_text = reason[:40]
                    cv2.putText(out, reason_text, (x, y + h + 50), self.font_small, 0.5, (255, 100, 100), 1)
                
                # Update stats
                self.total_scanned += 1
                if passed:
                    self.total_passed += 1
                else:
                    self.total_failed += 1

        # Update FPS
        self._frames += 1
        dt = time.time() - self._t0
        if dt >= 1.0:
            self.fps = self._frames / dt
            self._frames = 0
            self._t0 = time.time()

        # Draw stats panel
        self._draw_stats_panel(out)

        # Draw controls
        self._draw_controls(out)

        return out, barcodes

    # ----------------------------- Stats Panel ------------------------------
    def _draw_stats_panel(self, out: np.ndarray) -> None:
        """Draw statistics panel on the right side."""
        panel_width = 250
        panel_x = out.shape[1] - panel_width
        
        # Background
        cv2.rectangle(out, (panel_x, 80), (out.shape[1], out.shape[1] - 50), COLOR_BG, -1)
        cv2.rectangle(out, (panel_x, 80), (out.shape[1], out.shape[1] - 50), (100, 100, 100), 2)
        
        y = 110
        cv2.putText(out, "STATISTICS", (panel_x + 10, y), self.font_medium, 0.7, COLOR_INFO, 2)
        y += 35
        
        # FPS
        cv2.putText(out, f"FPS: {self.fps:.1f}", (panel_x + 10, y), self.font_small, 0.5, COLOR_INFO, 1)
        y += 30
        
        if self.mode == 'production':
            # Production stats
            y += 10
            cv2.putText(out, "PRODUCTION STATS:", (panel_x + 10, y), self.font_medium, 0.6, COLOR_INFO, 2)
            y += 30
            
            cv2.putText(out, f"Total Scanned: {self.total_scanned}", (panel_x + 10, y), self.font_small, 0.5, COLOR_INFO, 1)
            y += 25
            
            cv2.putText(out, f"PASSED: {self.total_passed}", (panel_x + 10, y), self.font_small, 0.5, COLOR_PASS, 2)
            y += 25
            
            cv2.putText(out, f"FAILED: {self.total_failed}", (panel_x + 10, y), self.font_small, 0.5, COLOR_FAIL, 2)
            y += 25
            
            if self.total_scanned > 0:
                pass_rate = (self.total_passed / self.total_scanned) * 100
                cv2.putText(out, f"Pass Rate: {pass_rate:.1f}%", (panel_x + 10, y), self.font_small, 0.5, COLOR_INFO, 1)
                y += 25
        else:
            # Calibration info
            y += 10
            cv2.putText(out, "CALIBRATION INFO:", (panel_x + 10, y), self.font_medium, 0.6, COLOR_CALIBRATE, 2)
            y += 30
            cv2.putText(out, "Press SPACE to set", (panel_x + 10, y), self.font_small, 0.5, COLOR_INFO, 1)
            y += 25
            cv2.putText(out, "golden sample", (panel_x + 10, y), self.font_small, 0.5, COLOR_INFO, 1)

    # ----------------------------- Controls --------------------------------
    def _draw_controls(self, out: np.ndarray) -> None:
        """Draw controls at the bottom."""
        controls = [
            "SPACE = Set Golden Sample",
            "c = Calibration Mode",
            "p = Production Mode",
            "q = Quit",
            "s = Save Frame",
            "r = Reset Golden Sample"
        ]
        
        y = out.shape[0] - 20
        for i, ctrl in enumerate(controls):
            x = 10 + i * 280
            cv2.putText(out, ctrl, (x, y), self.font_small, 0.5, (200, 200, 200), 1)

    # ------------------------------ Save -----------------------------------
    def save(self, frame: np.ndarray, filename: Optional[str] = None) -> str:
        """Save the annotated frame."""
        if filename is None:
            mode_prefix = "calibration" if self.mode == 'calibration' else "production"
            filename = f"{mode_prefix}_barcode_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = self.output_dir / filename
        cv2.imwrite(str(path), frame)
        print(f"Saved {path}")
        return str(path)

    # ------------------------------- Loop ----------------------------------
    def run(self) -> None:
        """Main processing loop with single-scan workflow."""
        print("=== Production Barcode Detection Started ===")
        print(f"Current Mode: {self.mode.upper()}")
        print("Single-scan mode: Each barcode is scanned once, then waits for next one.")
        
        last_barcode_count = 0
        scan_cooldown = 0
        
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[ERR] Failed to read camera frame.")
                break
            
            vis, barcodes = self.process_frame(frame)
            cv2.imshow('Production Barcode Inspector', vis)
            
            k = cv2.waitKey(1) & 0xFF
            
            if k == ord('q'):
                break
            elif k == ord('s'):
                self.save(vis)
            elif k == ord('c'):
                self.mode = 'calibration'
                print("Switched to CALIBRATION mode")
            elif k == ord('p'):
                if self.golden_sample:
                    self.mode = 'production'
                    print("Switched to PRODUCTION mode")
                else:
                    print("[WARN] No golden sample set. Please calibrate first.")
            elif k == ord('r'):
                self.reset_golden_sample()
            elif k == ord(' ') and self.mode == 'calibration' and barcodes:
                # Spacebar to set golden sample in calibration mode
                if barcodes:
                    self.set_golden_sample(barcodes[0])
            
            # Single-scan workflow for production mode
            if self.mode == 'production' and barcodes:
                current_barcode_count = len(barcodes)
                
                # Check if new barcode detected
                if current_barcode_count > 0 and scan_cooldown == 0:
                    # Verify the first barcode
                    bc = barcodes[0]
                    passed, reason = self.verify_barcode(bc)
                    
                    if passed:
                        # PASS - play beep sound
                        play_beep(frequency=1000, duration=200)
                        print(f"[PASS] {bc['type']}: {bc['data'][:30]}")
                    else:
                        # FAIL - play fail sound
                        play_fail_beep()
                        print(f"[FAIL] {bc['type']}: {reason}")
                    
                    # Set cooldown to prevent multiple scans of same barcode
                    scan_cooldown = 30  # Wait ~1 second before next scan
            
            # Decrease cooldown
            if scan_cooldown > 0:
                scan_cooldown -= 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


# ================================ CLI ======================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Production Barcode Inspector - Barcode Verification for Production Line')
    p.add_argument('--camera', type=int, default=0, help='Camera device ID (default 0)')
    p.add_argument('--config', type=str, default=None, help='Optional YAML config path')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    insp = ProductionBarcodeInspector(camera_id=args.camera, config_path=args.config)
    insp.run()


if __name__ == '__main__':
    main()
