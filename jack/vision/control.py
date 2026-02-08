"""
Mouse and Keyboard Control - Execute GUI actions

Cross-platform mouse/keyboard automation.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class Key(Enum):
    """Special keys"""
    ENTER = "enter"
    TAB = "tab"
    ESCAPE = "escape"
    BACKSPACE = "backspace"
    DELETE = "delete"
    SPACE = "space"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    HOME = "home"
    END = "end"
    PAGEUP = "pageup"
    PAGEDOWN = "pagedown"
    CTRL = "ctrl"
    ALT = "alt"
    SHIFT = "shift"
    CMD = "cmd"  # macOS command key
    WIN = "win"  # Windows key


@dataclass
class GUIAction:
    """Record of a GUI action taken"""
    action_type: str  # "click", "type", "scroll", "key"
    timestamp: float
    position: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    key: Optional[str] = None
    success: bool = True


class MouseKeyboard:
    """
    Cross-platform mouse and keyboard control.

    Uses pyautogui if available, falls back to OS-specific methods.
    """

    def __init__(self, failsafe: bool = True, pause: float = 0.1):
        """
        Args:
            failsafe: If True, moving mouse to corner aborts
            pause: Seconds to pause between actions
        """
        self.failsafe = failsafe
        self.pause = pause
        self._backend = self._detect_backend()
        self.action_history: List[GUIAction] = []

        # Configure pyautogui if available
        if self._backend == "pyautogui":
            import pyautogui
            pyautogui.FAILSAFE = failsafe
            pyautogui.PAUSE = pause

    def _detect_backend(self) -> str:
        """Detect available backend"""
        try:
            import pyautogui
            return "pyautogui"
        except ImportError:
            pass

        try:
            import pynput
            return "pynput"
        except ImportError:
            pass

        return "none"

    def click(
        self,
        x: int,
        y: int,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
    ) -> bool:
        """
        Click at position.

        Args:
            x, y: Screen coordinates
            button: Which mouse button
            clicks: Number of clicks (2 for double-click)

        Returns:
            True if successful
        """
        action = GUIAction(
            action_type="click",
            timestamp=time.time(),
            position=(x, y),
        )

        try:
            if self._backend == "pyautogui":
                import pyautogui
                pyautogui.click(x, y, clicks=clicks, button=button.value)

            elif self._backend == "pynput":
                from pynput.mouse import Controller, Button
                mouse = Controller()
                mouse.position = (x, y)
                btn = Button.left if button == MouseButton.LEFT else Button.right
                mouse.click(btn, clicks)

            else:
                raise RuntimeError("No mouse control backend available")

            action.success = True
            self.action_history.append(action)
            return True

        except Exception as e:
            action.success = False
            self.action_history.append(action)
            return False

    def double_click(self, x: int, y: int) -> bool:
        """Double click at position"""
        return self.click(x, y, clicks=2)

    def right_click(self, x: int, y: int) -> bool:
        """Right click at position"""
        return self.click(x, y, button=MouseButton.RIGHT)

    def move_to(self, x: int, y: int, duration: float = 0.25) -> bool:
        """Move mouse to position"""
        try:
            if self._backend == "pyautogui":
                import pyautogui
                pyautogui.moveTo(x, y, duration=duration)

            elif self._backend == "pynput":
                from pynput.mouse import Controller
                mouse = Controller()
                mouse.position = (x, y)

            return True
        except Exception:
            return False

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
    ) -> bool:
        """Drag from start to end position"""
        try:
            if self._backend == "pyautogui":
                import pyautogui
                pyautogui.moveTo(start_x, start_y)
                pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)

            elif self._backend == "pynput":
                from pynput.mouse import Controller, Button
                mouse = Controller()
                mouse.position = (start_x, start_y)
                mouse.press(Button.left)
                mouse.position = (end_x, end_y)
                mouse.release(Button.left)

            return True
        except Exception:
            return False

    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """
        Scroll wheel.

        Args:
            clicks: Positive = up, negative = down
            x, y: Position to scroll at (current if None)
        """
        action = GUIAction(
            action_type="scroll",
            timestamp=time.time(),
            position=(x, y) if x and y else None,
        )

        try:
            if self._backend == "pyautogui":
                import pyautogui
                if x is not None and y is not None:
                    pyautogui.scroll(clicks, x, y)
                else:
                    pyautogui.scroll(clicks)

            elif self._backend == "pynput":
                from pynput.mouse import Controller
                mouse = Controller()
                if x is not None and y is not None:
                    mouse.position = (x, y)
                mouse.scroll(0, clicks)

            action.success = True
            self.action_history.append(action)
            return True

        except Exception:
            action.success = False
            self.action_history.append(action)
            return False

    def type_text(self, text: str, interval: float = 0.02) -> bool:
        """
        Type text string.

        Args:
            text: Text to type
            interval: Seconds between keystrokes
        """
        action = GUIAction(
            action_type="type",
            timestamp=time.time(),
            text=text,
        )

        try:
            if self._backend == "pyautogui":
                import pyautogui
                pyautogui.typewrite(text, interval=interval)

            elif self._backend == "pynput":
                from pynput.keyboard import Controller
                keyboard = Controller()
                for char in text:
                    keyboard.type(char)
                    time.sleep(interval)

            action.success = True
            self.action_history.append(action)
            return True

        except Exception:
            action.success = False
            self.action_history.append(action)
            return False

    def press_key(self, key: Key) -> bool:
        """Press a special key"""
        action = GUIAction(
            action_type="key",
            timestamp=time.time(),
            key=key.value,
        )

        try:
            if self._backend == "pyautogui":
                import pyautogui
                pyautogui.press(key.value)

            elif self._backend == "pynput":
                from pynput.keyboard import Controller, Key as PynputKey
                keyboard = Controller()
                key_map = {
                    Key.ENTER: PynputKey.enter,
                    Key.TAB: PynputKey.tab,
                    Key.ESCAPE: PynputKey.esc,
                    Key.BACKSPACE: PynputKey.backspace,
                    Key.DELETE: PynputKey.delete,
                    Key.SPACE: PynputKey.space,
                    Key.UP: PynputKey.up,
                    Key.DOWN: PynputKey.down,
                    Key.LEFT: PynputKey.left,
                    Key.RIGHT: PynputKey.right,
                }
                if key in key_map:
                    keyboard.press(key_map[key])
                    keyboard.release(key_map[key])

            action.success = True
            self.action_history.append(action)
            return True

        except Exception:
            action.success = False
            self.action_history.append(action)
            return False

    def hotkey(self, *keys: str) -> bool:
        """
        Press key combination (e.g., ctrl+c).

        Args:
            keys: Key names like 'ctrl', 'c', 'alt', 'tab'
        """
        action = GUIAction(
            action_type="hotkey",
            timestamp=time.time(),
            key="+".join(keys),
        )

        try:
            if self._backend == "pyautogui":
                import pyautogui
                pyautogui.hotkey(*keys)

            elif self._backend == "pynput":
                from pynput.keyboard import Controller, Key as PynputKey
                keyboard = Controller()
                # This is simplified - would need proper key mapping
                for k in keys:
                    keyboard.press(k)
                for k in reversed(keys):
                    keyboard.release(k)

            action.success = True
            self.action_history.append(action)
            return True

        except Exception:
            action.success = False
            self.action_history.append(action)
            return False

    def get_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        if self._backend == "pyautogui":
            import pyautogui
            return pyautogui.position()

        elif self._backend == "pynput":
            from pynput.mouse import Controller
            return Controller().position

        return (0, 0)

    def copy(self) -> bool:
        """Ctrl+C / Cmd+C"""
        import platform
        if platform.system() == "Darwin":
            return self.hotkey("command", "c")
        return self.hotkey("ctrl", "c")

    def paste(self) -> bool:
        """Ctrl+V / Cmd+V"""
        import platform
        if platform.system() == "Darwin":
            return self.hotkey("command", "v")
        return self.hotkey("ctrl", "v")

    def select_all(self) -> bool:
        """Ctrl+A / Cmd+A"""
        import platform
        if platform.system() == "Darwin":
            return self.hotkey("command", "a")
        return self.hotkey("ctrl", "a")
