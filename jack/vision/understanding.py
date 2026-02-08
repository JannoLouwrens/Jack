"""
Vision Understanding - Analyze screenshots with AI

Uses vision-language models to:
1. Identify UI elements (buttons, text fields, menus)
2. Read text on screen
3. Understand layout and context
4. Answer questions about what's visible

Supports:
- Claude 3 Vision (Anthropic API)
- GPT-4V (OpenAI API)
- Local models (future: LLaVA, etc.)
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .screen import Screenshot


class UIElementType(Enum):
    """Types of UI elements"""
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    LINK = "link"
    IMAGE = "image"
    ICON = "icon"
    MENU = "menu"
    TAB = "tab"
    TEXT = "text"
    WINDOW = "window"
    UNKNOWN = "unknown"


@dataclass
class UIElement:
    """A detected UI element"""
    type: UIElementType
    text: str  # Visible text/label
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float = 1.0
    clickable: bool = True
    state: str = ""  # "enabled", "disabled", "checked", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point for clicking"""
        x, y, w, h = self.bounds
        return (x + w // 2, y + h // 2)

    def contains_point(self, px: int, py: int) -> bool:
        """Check if point is within bounds"""
        x, y, w, h = self.bounds
        return x <= px <= x + w and y <= py <= y + h


@dataclass
class ScreenAnalysis:
    """Result of analyzing a screenshot"""
    screenshot: Screenshot
    elements: List[UIElement] = field(default_factory=list)
    text_content: str = ""  # All readable text
    description: str = ""  # Natural language description
    app_name: str = ""  # Detected application
    window_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def find_element_by_text(self, text: str, partial: bool = True) -> Optional[UIElement]:
        """Find UI element by its text"""
        text_lower = text.lower()
        for element in self.elements:
            if partial:
                if text_lower in element.text.lower():
                    return element
            else:
                if text_lower == element.text.lower():
                    return element
        return None

    def find_elements_by_type(self, element_type: UIElementType) -> List[UIElement]:
        """Find all elements of a specific type"""
        return [e for e in self.elements if e.type == element_type]

    def find_clickable_at(self, x: int, y: int) -> Optional[UIElement]:
        """Find clickable element at coordinates"""
        for element in self.elements:
            if element.clickable and element.contains_point(x, y):
                return element
        return None


class VisionModel:
    """
    Vision-language model for screen understanding.

    The LOOP for GUI automation:
    1. Capture screenshot
    2. Send to vision model with task
    3. Get back: element locations, actions to take
    4. Execute action (click, type)
    5. Capture new screenshot
    6. Repeat until task complete
    """

    def __init__(
        self,
        provider: str = "anthropic",  # "anthropic", "openai", "local"
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model()
        self._client = None

    def _default_model(self) -> str:
        """Get default model for provider"""
        if self.provider == "anthropic":
            return "claude-sonnet-4-20250514"
        elif self.provider == "openai":
            return "gpt-4o"
        return "local"

    def analyze(
        self,
        screenshot: Screenshot,
        task: Optional[str] = None,
        find_elements: bool = True,
    ) -> ScreenAnalysis:
        """
        Analyze screenshot with vision model.

        Args:
            screenshot: Screenshot to analyze
            task: Optional task context (e.g., "find the login button")
            find_elements: Whether to detect UI elements

        Returns:
            ScreenAnalysis with detected elements and description
        """
        if self.provider == "anthropic":
            return self._analyze_anthropic(screenshot, task, find_elements)
        elif self.provider == "openai":
            return self._analyze_openai(screenshot, task, find_elements)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _analyze_anthropic(
        self,
        screenshot: Screenshot,
        task: Optional[str],
        find_elements: bool,
    ) -> ScreenAnalysis:
        """Analyze using Claude Vision"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        if self._client is None:
            api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key)

        # Build prompt
        prompt = self._build_analysis_prompt(task, find_elements)

        # Send to Claude with image
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot.to_base64(),
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        # Parse response
        response_text = response.content[0].text
        return self._parse_analysis_response(screenshot, response_text)

    def _analyze_openai(
        self,
        screenshot: Screenshot,
        task: Optional[str],
        find_elements: bool,
    ) -> ScreenAnalysis:
        """Analyze using GPT-4 Vision"""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        if self._client is None:
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=api_key)

        prompt = self._build_analysis_prompt(task, find_elements)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot.to_base64()}",
                            },
                        },
                    ],
                }
            ],
        )

        response_text = response.choices[0].message.content
        return self._parse_analysis_response(screenshot, response_text)

    def _build_analysis_prompt(self, task: Optional[str], find_elements: bool) -> str:
        """Build prompt for vision model"""
        prompt = """Analyze this screenshot.

"""
        if task:
            prompt += f"TASK: {task}\n\n"

        if find_elements:
            prompt += """Identify all interactive UI elements. For each element, provide:
- type: button, text_field, checkbox, dropdown, link, menu, tab, text
- text: visible label or text
- bounds: approximate [x, y, width, height] as percentages of screen (0-100)
- clickable: true/false

"""

        prompt += """Respond in this JSON format:
{
    "description": "Brief description of what's on screen",
    "app_name": "Name of application if identifiable",
    "window_title": "Window title if visible",
    "text_content": "All readable text on screen",
    "elements": [
        {
            "type": "button",
            "text": "Submit",
            "bounds": [45, 80, 10, 5],
            "clickable": true
        }
    ],
    "suggested_action": "What to click/do next for the task (if task provided)"
}

IMPORTANT: bounds are percentages [x%, y%, width%, height%] from top-left."""

        return prompt

    def _parse_analysis_response(self, screenshot: Screenshot, response: str) -> ScreenAnalysis:
        """Parse vision model response into ScreenAnalysis"""
        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}

        # Parse elements
        elements = []
        for elem_data in data.get("elements", []):
            try:
                # Convert percentage bounds to pixels
                bounds_pct = elem_data.get("bounds", [0, 0, 10, 5])
                bounds = (
                    int(bounds_pct[0] * screenshot.width / 100),
                    int(bounds_pct[1] * screenshot.height / 100),
                    int(bounds_pct[2] * screenshot.width / 100),
                    int(bounds_pct[3] * screenshot.height / 100),
                )

                element = UIElement(
                    type=UIElementType(elem_data.get("type", "unknown")),
                    text=elem_data.get("text", ""),
                    bounds=bounds,
                    clickable=elem_data.get("clickable", True),
                )
                elements.append(element)
            except Exception:
                continue

        return ScreenAnalysis(
            screenshot=screenshot,
            elements=elements,
            text_content=data.get("text_content", ""),
            description=data.get("description", ""),
            app_name=data.get("app_name", ""),
            window_title=data.get("window_title", ""),
            metadata={
                "suggested_action": data.get("suggested_action", ""),
                "raw_response": response,
            },
        )

    def ask(
        self,
        screenshot: Screenshot,
        question: str,
    ) -> str:
        """
        Ask a question about the screenshot.

        Args:
            screenshot: Screenshot to analyze
            question: Natural language question

        Returns:
            Answer as string
        """
        if self.provider == "anthropic":
            return self._ask_anthropic(screenshot, question)
        elif self.provider == "openai":
            return self._ask_openai(screenshot, question)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _ask_anthropic(self, screenshot: Screenshot, question: str) -> str:
        """Ask question using Claude"""
        import anthropic

        if self._client is None:
            api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot.to_base64(),
                            },
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                }
            ],
        )

        return response.content[0].text

    def _ask_openai(self, screenshot: Screenshot, question: str) -> str:
        """Ask question using GPT-4V"""
        import openai

        if self._client is None:
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=api_key)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot.to_base64()}",
                            },
                        },
                    ],
                }
            ],
        )

        return response.choices[0].message.content


class GUITaskLoop:
    """
    The vision-action loop for GUI automation.

    1. Capture screenshot
    2. Analyze with vision model
    3. Determine next action
    4. Execute action
    5. Repeat until done
    """

    def __init__(
        self,
        vision_model: VisionModel,
        screen_capture: 'ScreenCapture' = None,
        mouse_keyboard: 'MouseKeyboard' = None,
        max_steps: int = 20,
        verbose: bool = True,
    ):
        from .screen import ScreenCapture
        from .control import MouseKeyboard

        self.vision = vision_model
        self.screen = screen_capture or ScreenCapture()
        self.control = mouse_keyboard or MouseKeyboard()
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, task: str) -> bool:
        """
        Run GUI automation task.

        Args:
            task: Natural language task description

        Returns:
            True if task completed successfully
        """
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"GUI Task: {task}")
            print('='*50)

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n--- Step {step} ---")

            # 1. Capture screenshot
            screenshot = self.screen.capture()
            if self.verbose:
                print(f"Captured {screenshot.width}x{screenshot.height} screenshot")

            # 2. Analyze with task context
            analysis = self.vision.analyze(screenshot, task=task)
            if self.verbose:
                print(f"App: {analysis.app_name}")
                print(f"Found {len(analysis.elements)} UI elements")
                print(f"Description: {analysis.description[:100]}...")

            # 3. Get suggested action
            suggested = analysis.metadata.get("suggested_action", "")
            if self.verbose:
                print(f"Suggested: {suggested}")

            # 4. Check if done
            if self._is_task_complete(analysis, task):
                if self.verbose:
                    print("\nTask completed!")
                return True

            # 5. Execute action
            if suggested:
                success = self._execute_suggested_action(suggested, analysis)
                if not success:
                    if self.verbose:
                        print("Failed to execute action")
                    continue

            # Small delay between actions
            import time
            time.sleep(0.5)

        if self.verbose:
            print(f"\nMax steps ({self.max_steps}) reached without completion")
        return False

    def _is_task_complete(self, analysis: ScreenAnalysis, task: str) -> bool:
        """Check if task appears complete"""
        # Ask the vision model
        question = f"Is this task complete: '{task}'? Answer only YES or NO."
        answer = self.vision.ask(analysis.screenshot, question)
        return "YES" in answer.upper()

    def _execute_suggested_action(self, action: str, analysis: ScreenAnalysis) -> bool:
        """Execute a suggested action"""
        action_lower = action.lower()

        # Parse action type
        if "click" in action_lower:
            # Find what to click
            for element in analysis.elements:
                if element.text.lower() in action_lower and element.clickable:
                    x, y = element.center
                    self.control.click(x, y)
                    return True

        elif "type" in action_lower or "enter" in action_lower:
            # Extract text to type
            import re
            match = re.search(r'["\'](.+?)["\']', action)
            if match:
                text = match.group(1)
                self.control.type_text(text)
                return True

        elif "scroll" in action_lower:
            if "down" in action_lower:
                self.control.scroll(-3)
            else:
                self.control.scroll(3)
            return True

        return False
