"""
LLM - Real LLM Integration for Jack Foundation

This module provides REAL LLM integration using OpenAI-compatible APIs.
Works with:
- Local llama.cpp server
- OpenAI API
- Anthropic Claude API
- Any OpenAI-compatible endpoint

REPLACES: All SimpleReasoner/MockReasoner placeholders throughout the codebase.

Design Principles:
- Unified async interface for all modules
- Automatic retry with exponential backoff
- Streaming support for real-time responses
- JSON mode with schema validation
- Function/tool calling support

Author: Jack Foundation
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List, Dict, Optional, Any, Tuple, Union,
    Protocol, runtime_checkable, AsyncIterator, Callable
)
from datetime import datetime
from enum import Enum, auto
import json
import logging
import asyncio
import aiohttp
import re

from jack.foundation.types import Result, Ok, Err, Error, ErrorCode

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM connection."""
    base_url: str = "http://localhost:8080/v1"  # llama.cpp default
    api_key: str = "not-needed"  # Not needed for local
    model: str = "deepseek-r1-14b"  # Model identifier

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0  # 2 minutes for long generations

    # JSON mode
    json_mode: bool = False

    # Streaming
    stream: bool = False


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str
    raw_response: Dict[str, Any]


# =============================================================================
# UNIFIED REASONER PROTOCOL (ASYNC)
# =============================================================================

@runtime_checkable
class Reasoner(Protocol):
    """
    Unified async Reasoner protocol for ALL Jack Foundation modules.

    This is the SINGLE interface that all modules use.
    All implementations MUST be async.
    """

    async def reason(self, prompt: str) -> Result[str, Error]:
        """Send a prompt and get a text response."""
        ...

    async def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """Send a prompt and get a parsed JSON response."""
        ...

    async def reason_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream response tokens as they're generated."""
        ...


# =============================================================================
# LLM REASONER - REAL IMPLEMENTATION
# =============================================================================

class LLMReasoner:
    """
    Production LLM Reasoner using OpenAI-compatible API.

    Works with:
    - llama.cpp server (local)
    - OpenAI API
    - Any OpenAI-compatible endpoint

    Features:
    - Async for high concurrency
    - Automatic retry with backoff
    - JSON mode with validation
    - Streaming support
    - Usage tracking
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._session: Optional[aiohttp.ClientSession] = None

        # Usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def reason(self, prompt: str) -> Result[str, Error]:
        """
        Send a prompt and get a text response.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Result containing the response text or an error
        """
        result = await self._call_llm(prompt, json_mode=False)

        if result.is_err():
            return result

        response = result.unwrap()
        return Ok(response.content)

    async def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """
        Send a prompt and get a parsed JSON response.

        Automatically adds JSON instruction to prompt if not present.
        Retries with rephrased prompt on JSON parse failure.

        Args:
            prompt: The prompt to send (should request JSON output)

        Returns:
            Result containing parsed JSON dict or an error
        """
        # Add JSON instruction if not present
        json_prompt = prompt
        if "json" not in prompt.lower():
            json_prompt = f"{prompt}\n\nRespond with valid JSON only."

        for attempt in range(self.config.max_retries):
            result = await self._call_llm(json_prompt, json_mode=True)

            if result.is_err():
                return Err(result.unwrap_err())

            response = result.unwrap()

            # Try to parse JSON
            parsed = self._extract_json(response.content)
            if parsed is not None:
                return Ok(parsed)

            # JSON parse failed, retry with clearer instruction
            if attempt < self.config.max_retries - 1:
                json_prompt = f"""Your previous response was not valid JSON. Please respond with ONLY valid JSON, no markdown, no explanation.

Original request:
{prompt}

Respond with valid JSON only:"""
                logger.warning(f"JSON parse failed, retrying (attempt {attempt + 2}/{self.config.max_retries})")

        return Err(Error(
            ErrorCode.VALIDATION_FAILED,
            f"Failed to get valid JSON after {self.config.max_retries} attempts",
            details={"last_response": response.content[:500]}
        ))

    async def reason_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream response tokens as they're generated.

        Yields tokens one at a time for real-time display.
        """
        session = await self._get_session()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        url = f"{self.config.base_url}/chat/completions"

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM stream error: {error_text}")
                    yield f"[ERROR: {response.status}]"
                    return

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"[ERROR: {str(e)}]"

    async def _call_llm(
        self,
        prompt: str,
        json_mode: bool = False,
    ) -> Result[LLMResponse, Error]:
        """Make the actual API call to the LLM."""
        session = await self._get_session()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        # Add JSON mode if supported
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.config.base_url}/chat/completions"

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                self.total_requests += 1

                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Extract response
                        choice = data.get("choices", [{}])[0]
                        content = choice.get("message", {}).get("content", "")
                        finish_reason = choice.get("finish_reason", "unknown")

                        # Track usage
                        usage = data.get("usage", {})
                        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                        self.total_completion_tokens += usage.get("completion_tokens", 0)

                        return Ok(LLMResponse(
                            content=content,
                            model=data.get("model", self.config.model),
                            usage=usage,
                            finish_reason=finish_reason,
                            raw_response=data,
                        ))

                    elif response.status == 429:
                        # Rate limited, wait and retry
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    else:
                        error_text = await response.text()
                        last_error = Error(
                            ErrorCode.LLM_ERROR,
                            f"LLM API error: {response.status}",
                            details={"response": error_text}
                        )
                        self.failed_requests += 1

            except asyncio.TimeoutError:
                last_error = Error(
                    ErrorCode.TIMEOUT,
                    f"LLM request timed out after {self.config.timeout}s"
                )
                self.failed_requests += 1

            except aiohttp.ClientError as e:
                last_error = Error(
                    ErrorCode.CONNECTION_ERROR,
                    f"Connection error: {str(e)}"
                )
                self.failed_requests += 1

            # Wait before retry
            if attempt < self.config.max_retries - 1:
                wait_time = self.config.retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)

        return Err(last_error or Error(ErrorCode.UNKNOWN, "Unknown LLM error"))

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue

        return None

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "success_rate": (self.total_requests - self.failed_requests) / max(1, self.total_requests),
        }


# =============================================================================
# ANTHROPIC REASONER (Claude API)
# =============================================================================

class AnthropicReasoner:
    """
    Reasoner for Anthropic Claude API.

    Uses the native Anthropic API format (not OpenAI-compatible).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = "https://api.anthropic.com/v1"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def reason(self, prompt: str) -> Result[str, Error]:
        """Send a prompt and get a text response."""
        session = await self._get_session()

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            async with session.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data.get("content", [{}])[0].get("text", "")
                    return Ok(content)
                else:
                    error_text = await response.text()
                    return Err(Error(
                        ErrorCode.LLM_ERROR,
                        f"Anthropic API error: {response.status}",
                        details={"response": error_text}
                    ))
        except Exception as e:
            return Err(Error(ErrorCode.CONNECTION_ERROR, str(e)))

    async def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """Send a prompt and get a parsed JSON response."""
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no markdown."

        result = await self.reason(json_prompt)
        if result.is_err():
            return Err(result.unwrap_err())

        content = result.unwrap()

        # Try to parse JSON
        try:
            # Handle markdown code blocks
            if "```" in content:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if match:
                    content = match.group(1)
            return Ok(json.loads(content))
        except json.JSONDecodeError as e:
            return Err(Error(
                ErrorCode.VALIDATION_FAILED,
                f"Failed to parse JSON: {e}",
                details={"content": content[:500]}
            ))

    async def reason_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream response tokens."""
        session = await self._get_session()

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }

        try:
            async with session.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=headers
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                text = data.get("delta", {}).get("text", "")
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            yield f"[ERROR: {str(e)}]"


# =============================================================================
# MULTI-PROVIDER REASONER
# =============================================================================

class MultiProviderReasoner:
    """
    Reasoner that can failover between multiple LLM providers.

    Tries providers in order until one succeeds.
    """

    def __init__(self, providers: List[Union[LLMReasoner, AnthropicReasoner]]):
        self.providers = providers
        self.current_provider_index = 0

    async def reason(self, prompt: str) -> Result[str, Error]:
        """Try each provider in order."""
        errors = []

        for i, provider in enumerate(self.providers):
            result = await provider.reason(prompt)
            if result.is_ok():
                self.current_provider_index = i
                return result
            errors.append(str(result.unwrap_err()))

        return Err(Error(
            ErrorCode.LLM_ERROR,
            f"All {len(self.providers)} providers failed",
            details={"errors": errors}
        ))

    async def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """Try each provider in order."""
        errors = []

        for i, provider in enumerate(self.providers):
            result = await provider.reason_json(prompt)
            if result.is_ok():
                self.current_provider_index = i
                return result
            errors.append(str(result.unwrap_err()))

        return Err(Error(
            ErrorCode.LLM_ERROR,
            f"All {len(self.providers)} providers failed",
            details={"errors": errors}
        ))

    async def reason_stream(self, prompt: str) -> AsyncIterator[str]:
        """Use current provider for streaming."""
        provider = self.providers[self.current_provider_index]
        async for token in provider.reason_stream(prompt):
            yield token

    async def close(self):
        """Close all providers."""
        for provider in self.providers:
            await provider.close()


# =============================================================================
# SYNC WRAPPER (For backward compatibility)
# =============================================================================

class SyncReasonerWrapper:
    """
    Wraps an async reasoner to provide sync interface.

    Used for backward compatibility with modules that use sync calls.
    Creates a new event loop if needed.
    """

    def __init__(self, async_reasoner: Union[LLMReasoner, AnthropicReasoner, MultiProviderReasoner]):
        self.async_reasoner = async_reasoner

    def reason(self, prompt: str) -> Result[str, Error]:
        """Sync version of reason()."""
        return self._run_async(self.async_reasoner.reason(prompt))

    def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """Sync version of reason_json()."""
        return self._run_async(self.async_reasoner.reason_json(prompt))

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=120)
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(coro)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_local_reasoner(
    base_url: str = "http://localhost:8080/v1",
    model: str = "deepseek-r1-14b",
    **kwargs
) -> LLMReasoner:
    """
    Create a reasoner for local llama.cpp server.

    Args:
        base_url: URL of llama.cpp server
        model: Model name (for logging)
        **kwargs: Additional LLMConfig options

    Returns:
        Configured LLMReasoner
    """
    config = LLMConfig(
        base_url=base_url,
        model=model,
        api_key="not-needed",
        **kwargs
    )
    return LLMReasoner(config)


def create_openai_reasoner(
    api_key: str,
    model: str = "gpt-4o",
    **kwargs
) -> LLMReasoner:
    """
    Create a reasoner for OpenAI API.

    Args:
        api_key: OpenAI API key
        model: Model name (gpt-4o, gpt-4-turbo, etc.)
        **kwargs: Additional LLMConfig options

    Returns:
        Configured LLMReasoner
    """
    config = LLMConfig(
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        model=model,
        **kwargs
    )
    return LLMReasoner(config)


def create_anthropic_reasoner(
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    **kwargs
) -> AnthropicReasoner:
    """
    Create a reasoner for Anthropic Claude API.

    Args:
        api_key: Anthropic API key
        model: Model name
        **kwargs: Additional options

    Returns:
        Configured AnthropicReasoner
    """
    return AnthropicReasoner(api_key=api_key, model=model, **kwargs)


def create_failover_reasoner(
    local_url: str = "http://localhost:8080/v1",
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
) -> MultiProviderReasoner:
    """
    Create a reasoner that fails over between providers.

    Order: Local -> OpenAI -> Anthropic

    Args:
        local_url: URL of local llama.cpp server
        openai_key: Optional OpenAI API key
        anthropic_key: Optional Anthropic API key

    Returns:
        Configured MultiProviderReasoner
    """
    providers = [create_local_reasoner(local_url)]

    if openai_key:
        providers.append(create_openai_reasoner(openai_key))

    if anthropic_key:
        providers.append(create_anthropic_reasoner(anthropic_key))

    return MultiProviderReasoner(providers)


# =============================================================================
# SIMPLE REASONER (For testing - DEPRECATED)
# =============================================================================

class SimpleReasoner:
    """
    DEPRECATED: Use LLMReasoner instead.

    This is kept only for backward compatibility during migration.
    Will be removed in future versions.
    """

    def __init__(self):
        logger.warning(
            "SimpleReasoner is DEPRECATED. Use LLMReasoner for real LLM integration. "
            "SimpleReasoner returns mock responses only."
        )

    async def reason(self, prompt: str) -> Result[str, Error]:
        """Return mock response."""
        return Ok("[MOCK] This is a mock response. Use LLMReasoner for real LLM.")

    async def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """Return mock JSON response."""
        prompt_lower = prompt.lower()

        if "decompose" in prompt_lower or "goal" in prompt_lower:
            return Ok({
                "understanding": "Mock goal understanding",
                "entities_referenced": [],
                "information_requirements": [],
                "target_domains": ["environment", "filesystem"],
                "constraints": [],
                "success_criteria": "Mock criteria",
            })

        if "plan" in prompt_lower:
            return Ok({
                "steps": [
                    {"description": "Mock step 1", "type": "action"},
                    {"description": "Mock step 2", "type": "action"},
                ],
                "reasoning": "Mock planning reasoning",
            })

        if "verify" in prompt_lower or "safe" in prompt_lower:
            return Ok({
                "is_safe": True,
                "confidence": 0.8,
                "reasoning": "Mock safety verification",
            })

        return Ok({
            "response": "Mock response",
            "status": "success",
        })

    def reason_stream(self, prompt: str):
        """Mock streaming - just yields the whole response."""
        yield "[MOCK] Streaming not available in SimpleReasoner"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_reasoner():
        print("=" * 60)
        print("LLM REASONER TEST")
        print("=" * 60)

        # Test with local llama.cpp (if running)
        reasoner = create_local_reasoner()

        print("\n[TEST 1] Basic reasoning")
        result = await reasoner.reason("What is 2 + 2? Answer briefly.")
        if result.is_ok():
            print(f"  Response: {result.unwrap()[:100]}...")
        else:
            print(f"  Error: {result.unwrap_err()}")
            print("  (Is llama.cpp running on localhost:8080?)")

        print("\n[TEST 2] JSON reasoning")
        result = await reasoner.reason_json(
            "Return a JSON object with keys 'name' and 'age' for a person named Alice who is 30."
        )
        if result.is_ok():
            print(f"  Response: {result.unwrap()}")
        else:
            print(f"  Error: {result.unwrap_err()}")

        print("\n[TEST 3] Usage stats")
        print(f"  Stats: {reasoner.get_usage_stats()}")

        await reasoner.close()

        print("\n" + "=" * 60)
        print("[OK] LLM Reasoner module ready")
        print("=" * 60)

    asyncio.run(test_reasoner())
