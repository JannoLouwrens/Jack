"""
Screen Capture - Take screenshots for vision analysis

Cross-platform screenshot capture without heavy dependencies.
"""

import os
import io
import base64
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime


@dataclass
class Screenshot:
    """Captured screenshot data"""
    width: int
    height: int
    timestamp: datetime
    image_bytes: bytes  # PNG format
    region: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h if partial

    def to_base64(self) -> str:
        """Convert to base64 for API calls"""
        return base64.b64encode(self.image_bytes).decode('utf-8')

    def save(self, path: str) -> None:
        """Save screenshot to file"""
        Path(path).write_bytes(self.image_bytes)

    @property
    def size_kb(self) -> float:
        """Size in KB"""
        return len(self.image_bytes) / 1024


class ScreenCapture:
    """
    Cross-platform screen capture.

    Tries multiple backends:
    1. mss (fastest, cross-platform)
    2. PIL/Pillow ImageGrab
    3. pyautogui
    4. OS-specific fallback (screenshot commands)
    """

    def __init__(self):
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect available screenshot backend"""
        # Try mss first (fastest)
        try:
            import mss
            return "mss"
        except ImportError:
            pass

        # Try PIL
        try:
            from PIL import ImageGrab
            return "pil"
        except ImportError:
            pass

        # Try pyautogui
        try:
            import pyautogui
            return "pyautogui"
        except ImportError:
            pass

        # Fallback to OS commands
        return "os"

    def capture(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Screenshot:
        """
        Capture screenshot.

        Args:
            region: Optional (x, y, width, height) to capture specific area

        Returns:
            Screenshot object with image data
        """
        if self._backend == "mss":
            return self._capture_mss(region)
        elif self._backend == "pil":
            return self._capture_pil(region)
        elif self._backend == "pyautogui":
            return self._capture_pyautogui(region)
        else:
            return self._capture_os(region)

    def _capture_mss(self, region: Optional[Tuple[int, int, int, int]]) -> Screenshot:
        """Capture using mss"""
        import mss
        from PIL import Image

        with mss.mss() as sct:
            if region:
                x, y, w, h = region
                monitor = {"left": x, "top": y, "width": w, "height": h}
            else:
                monitor = sct.monitors[0]  # Full screen

            img = sct.grab(monitor)

            # Convert to PNG bytes
            pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

            return Screenshot(
                width=img.width,
                height=img.height,
                timestamp=datetime.now(),
                image_bytes=png_bytes,
                region=region,
            )

    def _capture_pil(self, region: Optional[Tuple[int, int, int, int]]) -> Screenshot:
        """Capture using PIL ImageGrab"""
        from PIL import ImageGrab

        if region:
            x, y, w, h = region
            bbox = (x, y, x + w, y + h)
            img = ImageGrab.grab(bbox=bbox)
        else:
            img = ImageGrab.grab()

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        return Screenshot(
            width=img.width,
            height=img.height,
            timestamp=datetime.now(),
            image_bytes=png_bytes,
            region=region,
        )

    def _capture_pyautogui(self, region: Optional[Tuple[int, int, int, int]]) -> Screenshot:
        """Capture using pyautogui"""
        import pyautogui

        if region:
            img = pyautogui.screenshot(region=region)
        else:
            img = pyautogui.screenshot()

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        return Screenshot(
            width=img.width,
            height=img.height,
            timestamp=datetime.now(),
            image_bytes=png_bytes,
            region=region,
        )

    def _capture_os(self, region: Optional[Tuple[int, int, int, int]]) -> Screenshot:
        """Capture using OS commands (fallback)"""
        import subprocess
        import platform

        # Create temp file
        temp_path = Path(tempfile.mktemp(suffix=".png"))

        try:
            system = platform.system().lower()

            if system == "darwin":  # macOS
                cmd = ["screencapture", "-x", str(temp_path)]
                if region:
                    x, y, w, h = region
                    cmd = ["screencapture", "-x", "-R", f"{x},{y},{w},{h}", str(temp_path)]

            elif system == "linux":
                # Try gnome-screenshot, scrot, or import (ImageMagick)
                if region:
                    x, y, w, h = region
                    cmd = ["import", "-window", "root", "-crop", f"{w}x{h}+{x}+{y}", str(temp_path)]
                else:
                    cmd = ["import", "-window", "root", str(temp_path)]

            elif system == "windows":
                # Use PowerShell
                ps_script = f"""
                Add-Type -AssemblyName System.Windows.Forms
                [System.Windows.Forms.Screen]::PrimaryScreen | ForEach-Object {{
                    $bitmap = New-Object System.Drawing.Bitmap($_.Bounds.Width, $_.Bounds.Height)
                    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                    $graphics.CopyFromScreen($_.Bounds.Location, [System.Drawing.Point]::Empty, $_.Bounds.Size)
                    $bitmap.Save('{temp_path}')
                }}
                """
                cmd = ["powershell", "-Command", ps_script]

            else:
                raise RuntimeError(f"Unsupported OS: {system}")

            subprocess.run(cmd, check=True, capture_output=True)

            # Read the file
            png_bytes = temp_path.read_bytes()

            # Get dimensions from PNG header (simple parsing)
            # PNG IHDR chunk starts at byte 16, width at 16-20, height at 20-24
            width = int.from_bytes(png_bytes[16:20], 'big')
            height = int.from_bytes(png_bytes[20:24], 'big')

            return Screenshot(
                width=width,
                height=height,
                timestamp=datetime.now(),
                image_bytes=png_bytes,
                region=region,
            )

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        if self._backend == "mss":
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                return monitor["width"], monitor["height"]

        elif self._backend == "pyautogui":
            import pyautogui
            return pyautogui.size()

        else:
            # Fallback: capture and check size
            screenshot = self.capture()
            return screenshot.width, screenshot.height
