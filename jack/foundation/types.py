"""
CORE TYPE SYSTEM - Rust-Inspired Result and Option Types

This module provides explicit error handling without exceptions.
All operations that can fail return Result[T, E] instead of raising.

Design Philosophy:
- Explicit is better than implicit
- Errors are values, not control flow
- Force handling of all cases
- Type-safe and composable

Usage:
    def divide(a: int, b: int) -> Result[float, str]:
        if b == 0:
            return Err("Division by zero")
        return Ok(a / b)

    result = divide(10, 2)
    match result:
        case Ok(value):
            print(f"Result: {value}")
        case Err(error):
            print(f"Error: {error}")

References:
- https://doc.rust-lang.org/std/result/
- https://github.com/rustedpy/result
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Callable, Union, Iterator,
    Optional, Any, Tuple, List, overload
)
from functools import wraps
import traceback

# Type variables
T = TypeVar('T')  # Success type
E = TypeVar('E')  # Error type
U = TypeVar('U')  # Mapped type
F = TypeVar('F')  # Mapped error type


# =============================================================================
# RESULT TYPE
# =============================================================================

class ResultBase(ABC, Generic[T, E]):
    """
    Base class for Result type.

    A Result represents either success (Ok) or failure (Err).
    This forces explicit handling of both cases.
    """

    @abstractmethod
    def is_ok(self) -> bool:
        """Returns True if this is Ok."""
        ...

    @abstractmethod
    def is_err(self) -> bool:
        """Returns True if this is Err."""
        ...

    @abstractmethod
    def ok(self) -> Option[T]:
        """Converts to Option, discarding error."""
        ...

    @abstractmethod
    def err(self) -> Option[E]:
        """Converts to Option of error, discarding success."""
        ...

    @abstractmethod
    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        """Maps the success value, leaving error unchanged."""
        ...

    @abstractmethod
    def map_err(self, fn: Callable[[E], F]) -> Result[T, F]:
        """Maps the error value, leaving success unchanged."""
        ...

    @abstractmethod
    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chains operations that return Result (flatMap)."""
        ...

    @abstractmethod
    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Provides fallback for errors."""
        ...

    @abstractmethod
    def unwrap(self) -> T:
        """Returns value or raises. Use sparingly."""
        ...

    @abstractmethod
    def unwrap_or(self, default: T) -> T:
        """Returns value or default."""
        ...

    @abstractmethod
    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        """Returns value or computes from error."""
        ...

    @abstractmethod
    def expect(self, msg: str) -> T:
        """Returns value or raises with message."""
        ...


@dataclass(frozen=True, slots=True)
class Ok(ResultBase[T, E]):
    """
    Success variant of Result.

    Immutable and hashable.
    """
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def ok(self) -> Option[T]:
        return Some(self.value)

    def err(self) -> Option[E]:
        return NONE

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], F]) -> Result[T, F]:
        return Ok(self.value)

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return fn(self.value)

    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        return Ok(self.value)

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        return self.value

    def expect(self, msg: str) -> T:
        return self.value

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class Err(ResultBase[T, E]):
    """
    Error variant of Result.

    Immutable and hashable (if error is hashable).
    """
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def ok(self) -> Option[T]:
        return NONE

    def err(self) -> Option[E]:
        return Some(self.error)

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        return Err(self.error)

    def map_err(self, fn: Callable[[E], F]) -> Result[T, F]:
        return Err(fn(self.error))

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return Err(self.error)

    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        return fn(self.error)

    def unwrap(self) -> T:
        raise UnwrapError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        return fn(self.error)

    def unwrap_err(self) -> E:
        """Returns error value. Only valid on Err."""
        return self.error

    def expect(self, msg: str) -> T:
        raise UnwrapError(f"{msg}: {self.error}")

    def __repr__(self) -> str:
        return f"Err({self.error!r})"

    def __bool__(self) -> bool:
        return False


# Type alias for Result
Result = Union[Ok[T, E], Err[T, E]]


class UnwrapError(Exception):
    """Raised when unwrap is called on Err or NONE."""
    pass


# =============================================================================
# OPTION TYPE
# =============================================================================

class OptionBase(ABC, Generic[T]):
    """
    Base class for Option type.

    An Option represents a value that may or may not exist.
    This is explicit handling of null/None.
    """

    @abstractmethod
    def is_some(self) -> bool:
        """Returns True if this contains a value."""
        ...

    @abstractmethod
    def is_none(self) -> bool:
        """Returns True if this is empty."""
        ...

    @abstractmethod
    def map(self, fn: Callable[[T], U]) -> Option[U]:
        """Maps the value if present."""
        ...

    @abstractmethod
    def and_then(self, fn: Callable[[T], Option[U]]) -> Option[U]:
        """Chains operations that return Option."""
        ...

    @abstractmethod
    def or_else(self, fn: Callable[[], Option[T]]) -> Option[T]:
        """Provides fallback if empty."""
        ...

    @abstractmethod
    def unwrap(self) -> T:
        """Returns value or raises."""
        ...

    @abstractmethod
    def unwrap_or(self, default: T) -> T:
        """Returns value or default."""
        ...

    @abstractmethod
    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        """Returns value or computes default."""
        ...

    @abstractmethod
    def ok_or(self, error: E) -> Result[T, E]:
        """Converts to Result with given error."""
        ...


@dataclass(frozen=True, slots=True)
class Some(OptionBase[T]):
    """
    Some variant of Option - contains a value.
    """
    value: T

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def map(self, fn: Callable[[T], U]) -> Option[U]:
        return Some(fn(self.value))

    def and_then(self, fn: Callable[[T], Option[U]]) -> Option[U]:
        return fn(self.value)

    def or_else(self, fn: Callable[[], Option[T]]) -> Option[T]:
        return self

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return self.value

    def ok_or(self, error: E) -> Result[T, E]:
        return Ok(self.value)

    def __repr__(self) -> str:
        return f"Some({self.value!r})"

    def __bool__(self) -> bool:
        return True

    def __iter__(self) -> Iterator[T]:
        yield self.value


class _None(OptionBase[Any]):
    """
    None variant of Option - empty.

    Singleton instance.
    """
    _instance: Optional[_None] = None

    def __new__(cls) -> _None:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_some(self) -> bool:
        return False

    def is_none(self) -> bool:
        return True

    def map(self, fn: Callable[[Any], U]) -> Option[U]:
        return self

    def and_then(self, fn: Callable[[Any], Option[U]]) -> Option[U]:
        return self

    def or_else(self, fn: Callable[[], Option[T]]) -> Option[T]:
        return fn()

    def unwrap(self) -> Any:
        raise UnwrapError("Called unwrap on NONE")

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return fn()

    def ok_or(self, error: E) -> Result[Any, E]:
        return Err(error)

    def __repr__(self) -> str:
        return "NONE"

    def __bool__(self) -> bool:
        return False

    def __iter__(self) -> Iterator[Any]:
        return iter([])


# Singleton NONE instance
NONE: _None = _None()

# Type alias for Option
Option = Union[Some[T], _None]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def catch(
    fn: Callable[..., T],
    *args,
    error_type: type = Exception,
    **kwargs
) -> Result[T, str]:
    """
    Catch exceptions and convert to Result.

    Usage:
        result = catch(lambda: 1/0)
        # Returns Err("division by zero")
    """
    try:
        return Ok(fn(*args, **kwargs))
    except error_type as e:
        return Err(str(e))


def catch_detailed(
    fn: Callable[..., T],
    *args,
    error_type: type = Exception,
    **kwargs
) -> Result[T, Tuple[str, str]]:
    """
    Catch exceptions with traceback.

    Returns Err((error_message, traceback_string))
    """
    try:
        return Ok(fn(*args, **kwargs))
    except error_type as e:
        tb = traceback.format_exc()
        return Err((str(e), tb))


def from_optional(value: Optional[T]) -> Option[T]:
    """Convert Python Optional to Option type."""
    if value is None:
        return NONE
    return Some(value)


def to_optional(option: Option[T]) -> Optional[T]:
    """Convert Option type to Python Optional."""
    if option.is_some():
        return option.unwrap()
    return None


def collect_results(results: List[Result[T, E]]) -> Result[List[T], E]:
    """
    Collect a list of Results into a Result of list.

    Returns first error encountered, or Ok with all values.
    """
    values = []
    for result in results:
        if result.is_err():
            return result
        values.append(result.unwrap())
    return Ok(values)


def collect_options(options: List[Option[T]]) -> Option[List[T]]:
    """
    Collect a list of Options into an Option of list.

    Returns NONE if any option is NONE, otherwise Some with all values.
    """
    values = []
    for option in options:
        if option.is_none():
            return NONE
        values.append(option.unwrap())
    return Some(values)


# =============================================================================
# DECORATORS
# =============================================================================

def result_from_exception(
    error_type: type = Exception
) -> Callable[[Callable[..., T]], Callable[..., Result[T, str]]]:
    """
    Decorator to convert function that raises to function that returns Result.

    Usage:
        @result_from_exception()
        def might_fail(x: int) -> int:
            if x < 0:
                raise ValueError("Negative!")
            return x * 2

        result = might_fail(-1)  # Returns Err("Negative!")
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., Result[T, str]]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Result[T, str]:
            try:
                return Ok(fn(*args, **kwargs))
            except error_type as e:
                return Err(str(e))
        return wrapper
    return decorator


# =============================================================================
# ERROR TYPES
# =============================================================================

@dataclass(frozen=True)
class Error:
    """
    Structured error type for detailed error information.
    """
    code: str
    message: str
    details: Optional[dict] = None
    cause: Optional[Error] = None

    def chain(self, code: str, message: str) -> Error:
        """Create a new error with this one as the cause."""
        return Error(code=code, message=message, cause=self)

    def __str__(self) -> str:
        if self.cause:
            return f"[{self.code}] {self.message} (caused by: {self.cause})"
        return f"[{self.code}] {self.message}"


# Common error codes
class ErrorCode:
    # General
    UNKNOWN = "UNKNOWN"
    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TIMEOUT = "TIMEOUT"

    # Action-specific
    EXECUTION_FAILED = "EXECUTION_FAILED"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    POSTCONDITION_FAILED = "POSTCONDITION_FAILED"

    # Plan-specific
    PLANNING_FAILED = "PLANNING_FAILED"
    STEP_FAILED = "STEP_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"

    # Memory-specific
    STORAGE_ERROR = "STORAGE_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CORE TYPE SYSTEM - Result and Option")
    print("=" * 60)

    # Test Result
    print("\n[TEST] Result type")

    def safe_divide(a: int, b: int) -> Result[float, str]:
        if b == 0:
            return Err("Division by zero")
        return Ok(a / b)

    r1 = safe_divide(10, 2)
    r2 = safe_divide(10, 0)

    print(f"  10 / 2 = {r1}")
    print(f"  10 / 0 = {r2}")

    # Test chaining
    result = (
        safe_divide(10, 2)
        .and_then(lambda x: safe_divide(x, 2))
        .map(lambda x: x * 100)
    )
    print(f"  Chained: {result}")

    # Test Option
    print("\n[TEST] Option type")

    def find_in_list(lst: list, pred) -> Option[Any]:
        for item in lst:
            if pred(item):
                return Some(item)
        return NONE

    o1 = find_in_list([1, 2, 3], lambda x: x > 2)
    o2 = find_in_list([1, 2, 3], lambda x: x > 10)

    print(f"  Find > 2: {o1}")
    print(f"  Find > 10: {o2}")

    # Test Error type
    print("\n[TEST] Error type")

    err = Error(
        code=ErrorCode.NOT_FOUND,
        message="File not found",
        details={"path": "/tmp/missing.txt"}
    )
    chained = err.chain(ErrorCode.EXECUTION_FAILED, "Could not read config")
    print(f"  Error: {err}")
    print(f"  Chained: {chained}")

    # Test decorator
    print("\n[TEST] Decorator")

    @result_from_exception()
    def risky_operation(x: int) -> int:
        if x < 0:
            raise ValueError("Negative value!")
        return x * 2

    print(f"  risky(5) = {risky_operation(5)}")
    print(f"  risky(-1) = {risky_operation(-1)}")

    print("\n" + "=" * 60)
    print("[OK] Core type system working")
    print("=" * 60)
