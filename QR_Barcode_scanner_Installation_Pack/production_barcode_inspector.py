#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Barcode Inspector — Barcode Verification for Production Line
Purpose: Golden sample calibration + production barcode verification.
Controls: SPACE=Set Golden, c=Calibrate, p=Production, q=Quit, s=Save, r=Reset
Supports: QR, Data Matrix, EAN-13/8, UPC-A/E, CODE128, CODE39, and more.
Dependencies: pip install opencv-python numpy pyzbar pyyaml
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import cv2, numpy as np

# Windows beep
try:
    import ctypes
    def _beep(freq: int, dur: int) -> None: ctypes.windll.kernel32.Beep(freq, dur)
    def play_beep() -> None: _beep(1000, 200)
    def play_fail_beep() -> None: (_beep(400, 100), ctypes.windll.kernel32.Sleep(50), _beep(400, 100))
except Exception:
    def play_beep() -> None: pass
    def play_fail_beep() -> None: pass

# pyzbar import
try:
    from pyzbar.pyzbar import decode
except ImportError:
    print("[ERROR] pyzbar not installed. Run: pip install pyzbar")
    sys.exit(1)

# ============================ Constants ====================================
# Camera
W, H = 1280, 720
# Detection
MIN_SIZE, MAX_SIZE = 20, 1000
QUALITY_THRESHOLD = 0.5
# Output
OUTPUT_DIR = 'output'
GOLDEN_FILE = 'golden_sample.json'
# Colors
COLORS = {
    'pass': (0, 200, 0), 'fail': (0, 0, 255), 'calibrate': (255, 165, 0),
    'info': (255, 255, 255), 'bg': (30, 30, 30), 'text': (200, 200, 200)
}
# 1D barcode types (relaxed quality threshold)
_1D_BARCODES: Set[str] = {
    "EAN13", "EAN8", "CODE128", "CODE39", "UPCA", "UPCE",
    "EAN5", "EAN2", "CODABAR", "CODE93", "ITF"
}
# Rotation mappings
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# ============================================================================

def clamp(v: float, lo: float, hi: float) -> float: return max(lo, min(hi, v))

class Inspector:
    def __init__(self, camera_id: int = 0, config_path: Optional[str] = None) -> None:
        self.cfg = self._load_config(config_path)
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get('frame_width', W)))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get('frame_height', H)))
        self.min_size = int(self.cfg.get('min_barcode_size', MIN_SIZE))
        self.max_size = int(self.cfg.get('max_barcode_size', MAX_SIZE))
        self.quality_thresh = float(self.cfg.get('quality_threshold', QUALITY_THRESHOLD))
        self.output_dir = Path(self.cfg.get('output_dir', OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.golden_file = Path(self.cfg.get('golden_file', GOLDEN_FILE))
        self.golden = self._load_golden()
        self.mode = 'calibration' if not self.golden else 'production'
        self.fps = 0.0
        self._t0 = time.time()
        self._frames = 0
        self.stats = {'scanned': 0, 'passed': 0, 'failed': 0}
        print(f"Inspector initialized. Mode: {self.mode.upper()}")

    # --------------------------- Config & Golden --------------------------
    def _load_config(self, path: Optional[str]) -> Dict:
        if not path: return {}
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Config load failed: {e}")
            return {}

    def _load_golden(self) -> Optional[Dict]:
        if self.golden_file.exists():
            try:
                with open(self.golden_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Golden load failed: {e}")
        return None

    def _save_golden(self, sample: Dict) -> None:
        try:
            with open(self.golden_file, 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Golden save failed: {e}")

    def set_golden(self, barcode: Dict) -> None:
        self.golden = {
            'type': barcode['type'], 'data': barcode['data'],
            'quality': barcode['quality'], 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self._save_golden(self.golden)
        self.mode = 'production'
        print(f"Golden set: {barcode['type']} - {barcode['data'][:30]}")

    def reset_golden(self) -> None:
        self.golden = None
        if self.golden_file.exists():
            self.golden_file.unlink()
        self.mode = 'calibration'
        print("Golden reset. Mode: CALIBRATION")

    # --------------------------- Preprocess -------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        return gray

    # --------------------------- Detect -----------------------------------
    def detect_barcodes(self, frame: np.ndarray, gray: np.ndarray) -> List[Dict]:
        for angle in [0, 90, 180, 270]:
            processed = gray if angle == 0 else cv2.rotate(gray, ROTATIONS[angle])
            barcodes = decode(processed) or []
            if not barcodes:
                continue

            h, w = gray.shape
            results = []
            for bc in barcodes:
                x, y, bw, bh = bc.rect
                rh, rw = processed.shape
                if bw < self.min_size or bh < self.min_size or bw > rw or bh > rh:
                    continue

                # Coordinate conversion
                if angle == 0:
                    ox, oy, ow, oh = x, y, bw, bh
                elif angle == 90:
                    ox, oy, ow, oh = h - y - bh, x, bh, bw
                elif angle == 180:
                    ox, oy, ow, oh = w - x - bw, h - y - bh, bw, bh
                else:
                    ox, oy, ow, oh = y, w - x - bw, bh, bw

                # Polygon conversion
                orig_pts = []
                for px, py in [(p.x, p.y) for p in bc.polygon]:
                    if angle == 0: opx, opy = px, py
                    elif angle == 90: opx, opy = h - py, px
                    elif angle == 180: opx, opy = w - px, h - py
                    else: opx, opy = py, w - px
                    orig_pts.append((opx, opy))

                # Quality assessment
                is_1d = bc.type in _1D_BARCODES
                quality = 1.0 if is_1d else self._assess_quality(orig_pts, ow, oh)

                results.append({
                    'type': bc.type, 'data': bc.data.decode('utf-8'),
                    'rect': (ox, oy, ow, oh), 'polygon': orig_pts,
                    'quality': quality, 'center': (ox + ow // 2, oy + oh // 2)
                })
            if results:
                return results
        return []

    def _assess_quality(self, pts: List[Tuple], w: int, h: int) -> float:
        if len(pts) < 4:
            return 0.5
        arr = np.array(pts, dtype=np.float32)
        area_ratio = cv2.contourArea(cv2.convexHull(arr)) / max(w * h, 1)
        aspect_score = 1.0 - min(abs(w / max(h, 1) - 1.0), 1.0)
        return clamp(0.6 * area_ratio + 0.4 * aspect_score, 0.0, 1.0)

    # --------------------------- Verify -----------------------------------
    def verify(self, bc: Dict) -> Tuple[bool, str]:
        if not self.golden:
            return False, "No golden sample"
        if bc['type'] != self.golden['type']:
            return False, f"Type mismatch"
        if bc['data'] != self.golden['data']:
            return False, f"Data mismatch"
        thresh = 0.2 if bc['type'] in _1D_BARCODES else self.quality_thresh
        if bc['quality'] < thresh:
            return False, f"Quality low: {bc['quality']:.2f}"
        return True, "PASS"

    # --------------------------- Draw -------------------------------------
    def _draw_barcode(self, out: np.ndarray, bc: Dict, color: Tuple, label: str) -> None:
        x, y, w, h = bc['rect']
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 4)
        if len(bc['polygon']) >= 4:
            pts = np.array(bc['polygon'], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts], True, color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.rectangle(out, (x, y - th - 15), (x + tw + 15, y), color, -1)
        cv2.putText(out, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    def _draw_panel(self, out: np.ndarray) -> None:
        pw, px = 250, out.shape[1] - 250
        cv2.rectangle(out, (px, 80), (out.shape[1], out.shape[0] - 50), COLORS['bg'], -1)
        cv2.rectangle(out, (px, 80), (out.shape[1], out.shape[0] - 50), (100, 100, 100), 2)
        y = 110
        cv2.putText(out, "STATISTICS", (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['info'], 2)
        y += 35
        cv2.putText(out, f"FPS: {self.fps:.1f}", (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['info'], 1)
        y += 30
        if self.mode == 'production':
            s = self.stats
            y += 10
            cv2.putText(out, "PRODUCTION STATS:", (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['info'], 2)
            y += 30
            cv2.putText(out, f"Scanned: {s['scanned']}", (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['info'], 1)
            y += 25
            cv2.putText(out, f"PASSED: {s['passed']}", (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['pass'], 2)
            y += 25
            cv2.putText(out, f"FAILED: {s['failed']}", (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['fail'], 2)
            y += 25
            if s['scanned'] > 0:
                cv2.putText(out, f"Rate: {s['passed']/s['scanned']*100:.1f}%", (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['info'], 1)
        else:
            y += 10
            cv2.putText(out, "CALIBRATION:", (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['calibrate'], 2)
            y += 30
            cv2.putText(out, "SPACE = Set Golden", (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['info'], 1)

    # --------------------------- Process ---------------------------------
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        gray = self.preprocess(frame)
        barcodes = self.detect_barcodes(frame, gray)
        out = frame.copy()

        # Mode header
        mode_color = COLORS['calibrate'] if self.mode == 'calibration' else COLORS['pass']
        cv2.rectangle(out, (0, 0), (out.shape[1], 80), mode_color, -1)
        cv2.putText(out, f"MODE: {self.mode.upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        hint = "Scan GOLDEN SAMPLE" if self.mode == 'calibration' else f"Golden: {self.golden.get('data','N/A')[:35]}"
        cv2.putText(out, hint, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for bc in barcodes:
            if self.mode == 'calibration':
                self._draw_barcode(out, bc, COLORS['calibrate'], f"{bc['type'].upper()}: {bc['data'][:20]}")
                cv2.putText(out, f"Quality: {bc['quality']:.2f}", (bc['rect'][0], bc['rect'][1] + bc['rect'][3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['info'], 1)
            else:
                passed, reason = self.verify(bc)
                color = COLORS['pass'] if passed else COLORS['fail']
                self._draw_barcode(out, bc, color, f"{'PASS' if passed else 'FAIL'} - {bc['type'].upper()}")
                cv2.putText(out, bc['data'][:25] + ('...' if len(bc['data']) > 25 else ''), (bc['rect'][0], bc['rect'][1] + bc['rect'][3] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['info'], 2)
                if not passed:
                    cv2.putText(out, reason[:40], (bc['rect'][0], bc['rect'][1] + bc['rect'][3] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                self.stats['scanned'] += 1
                if passed:
                    self.stats['passed'] += 1
                else:
                    self.stats['failed'] += 1

        # FPS update
        self._frames += 1
        if (dt := time.time() - self._t0) >= 1.0:
            self.fps = self._frames / dt
            self._frames = 0
            self._t0 = time.time()

        self._draw_panel(out)
        # Controls
        for i, ctrl in enumerate(["SPACE=Set", "c=Cal", "p=Prod", "q=Quit", "s=Save", "r=Reset"]):
            cv2.putText(out, ctrl, (10 + i * 200, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        return out, barcodes

    # --------------------------- Save & Run -------------------------------
    def save(self, frame: np.ndarray) -> str:
        prefix = "calibration" if self.mode == 'calibration' else "production"
        path = self.output_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), frame)
        print(f"Saved {path}")
        return str(path)

    def run(self) -> None:
        print("=== Started ===")
        cooldown = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[ERR] Camera read failed")
                break
            vis, barcodes = self.process(frame)
            cv2.imshow('Barcode Inspector', vis)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'): break
            if k == ord('s'): self.save(vis)
            if k == ord('c'): self.mode = 'calibration'
            if k == ord('p') and self.golden: self.mode = 'production'
            if k == ord('r'): self.reset_golden()
            if k == ord(' ') and self.mode == 'calibration' and barcodes:
                self.set_golden(barcodes[0])

            # Single-scan workflow
            if self.mode == 'production' and barcodes and cooldown == 0:
                passed, reason = self.verify(barcodes[0])
                (play_beep if passed else play_fail_beep)()
                print(f"[{'PASS' if passed else 'FAIL'}] {barcodes[0]['type']}: {reason}")
                cooldown = 30
            cooldown = max(0, cooldown - 1)

        self.cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")

# ================================ CLI ======================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Production Barcode Inspector')
    p.add_argument('--camera', type=int, default=0, help='Camera device ID (default 0)')
    p.add_argument('--config', type=str, default=None, help='YAML config file path')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    Inspector(camera_id=args.camera, config_path=args.config).run()

if __name__ == '__main__':
    main()
