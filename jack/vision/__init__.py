"""
Jack Vision - Screen Understanding and GUI Automation

Enables Jack to:
1. See what's on screen (screenshots)
2. Understand UI elements (buttons, text, images)
3. Control mouse/keyboard
4. Automate GUI tasks

Uses vision-language models (Claude 3, GPT-4V) or local models.
"""

from .screen import ScreenCapture, Screenshot
from .understanding import VisionModel, UIElement, ScreenAnalysis
from .control import MouseKeyboard, GUIAction

__all__ = [
    "ScreenCapture",
    "Screenshot",
    "VisionModel",
    "UIElement",
    "ScreenAnalysis",
    "MouseKeyboard",
    "GUIAction",
]
