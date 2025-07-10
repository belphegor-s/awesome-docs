# Python Cheat Sheet - Complete Documentation

## Table of Contents

1. [Basic Syntax & Variables](#basic-syntax--variables)
2. [Data Types & Structures](#data-types--structures)
3. [Control Flow](#control-flow)
4. [Functions](#functions)
5. [Object-Oriented Programming](#object-oriented-programming)
6. [Error Handling](#error-handling)
7. [File I/O & Path Operations](#file-io--path-operations)
8. [Iterators & Generators](#iterators--generators)
9. [Functional Programming](#functional-programming)
10. [Concurrency & Parallelism](#concurrency--parallelism)
11. [Advanced Features](#advanced-features)
12. [Testing & Debugging](#testing--debugging)
13. [Performance & Optimization](#performance--optimization)
14. [Logging](#logging)
15. [Regular Expressions](#regular-expressions)
16. [Professional Best Practices](#professional-best-practices)

---

## Basic Syntax & Variables

### Variable Naming Conventions (PEP 8)

- **Variables & Functions**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Classes**: `PascalCase`
- **Private attributes**: `_single_underscore`
- **Name mangling**: `__double_underscore`

### Type Hints

```python
# Basic types
name: str = "Python"
age: int = 25
price: float = 19.99
is_active: bool = True

# Collections
numbers: List[int] = [1, 2, 3]
person: Dict[str, Any] = {"name": "Alice", "age": 30}
coordinates: Tuple[int, int] = (10, 20)

# Optional and Union types
optional_name: Optional[str] = None
id_or_name: Union[int, str] = "user_123"
```

### Multiple Assignment & Unpacking

```python
# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0

# Unpacking
first, *middle, last = [1, 2, 3, 4, 5]
head, *tail = ["a", "b", "c", "d"]
```

---

## Data Types & Structures

### Strings

```python
# F-strings (Python 3.6+) - Preferred method
name = "Alice"
age = 30
message = f"Hello, {name}! You are {age} years old."

# String methods
text = "  Hello, World!  "
cleaned = text.strip().lower().replace("world", "python")
words = text.split()
joined = "-".join(words)
```

### Lists

```python
# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
matrix = [[i*j for j in range(3)] for i in range(3)]

# List methods
numbers = [1, 2, 3]
numbers.append(4)           # Add to end
numbers.insert(0, 0)        # Insert at index
numbers.extend([5, 6])      # Add multiple items
numbers.remove(2)           # Remove first occurrence
popped = numbers.pop()      # Remove and return last item
```

### Dictionaries

```python
# Dictionary comprehensions
squared_dict = {x: x**2 for x in range(5)}
filtered_dict = {k: v for k, v in data.items() if condition}

# Dictionary methods
person = {"name": "Alice", "age": 30}
name = person.get("name", "Unknown")    # Safe access
keys = person.keys()
values = person.values()
items = person.items()
```

### Sets

```python
# Set operations
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}

union = set_a | set_b           # {1, 2, 3, 4, 5, 6}
intersection = set_a & set_b    # {3, 4}
difference = set_a - set_b      # {1, 2}
symmetric_diff = set_a ^ set_b  # {1, 2, 5, 6}
```

---

## Control Flow

### Conditional Statements

```python
# If-elif-else
def categorize_age(age: int) -> str:
    if age < 13:
        return "child"
    elif age < 20:
        return "teenager"
    elif age < 65:
        return "adult"
    else:
        return "senior"

# Ternary operator
status = "active" if user.is_logged_in else "inactive"

# Match statement (Python 3.10+)
def handle_status(status: str) -> str:
    match status:
        case "pending":
            return "Processing..."
        case "approved":
            return "Access granted"
        case "rejected":
            return "Access denied"
        case _:
            return "Unknown status"
```

### Loops

```python
# For loops with enumerate
for index, value in enumerate(items):
    print(f"{index}: {value}")

# For loops with zip
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Dictionary iteration
for key, value in dictionary.items():
    print(f"{key}: {value}")

# While loops with else
counter = 0
while counter < 5:
    print(counter)
    counter += 1
else:
    print("Loop completed normally")
```

---

## Functions

### Function Definitions

```python
# Basic function with type hints
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

# Function with default parameters
def greet(name: str, greeting: str = "Hello") -> str:
    """Greet a person with a custom greeting."""
    return f"{greeting}, {name}!"

# Function with variable arguments
def sum_numbers(*args: int) -> int:
    """Sum any number of integers."""
    return sum(args)

def create_profile(**kwargs: Any) -> Dict[str, Any]:
    """Create a user profile from keyword arguments."""
    return kwargs
```

### Advanced Function Features

```python
# Lambda functions
square = lambda x: x**2
is_even = lambda x: x % 2 == 0

# Higher-order functions
def apply_function(func: Callable, data: List[Any]) -> List[Any]:
    """Apply a function to each item in a list."""
    return [func(item) for item in data]

# Closures
def make_multiplier(factor: int) -> Callable[[int], int]:
    """Create a multiplier function."""
    def multiplier(x: int) -> int:
        return x * factor
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)
```

### Decorators

```python
# Function decorator
def log_calls(func):
    """Decorator to log function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

# Decorator with parameters
def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# Class decorator
def singleton(cls):
    """Decorator to make a class a singleton."""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance
```

---

## Object-Oriented Programming

### Basic Classes

```python
class BankAccount:
    """A simple bank account class."""

    # Class variable
    bank_name = "MyBank"

    def __init__(self, account_number: str, initial_balance: float = 0.0):
        self.account_number = account_number
        self._balance = initial_balance  # Protected attribute
        self.__pin = None  # Private attribute

    def deposit(self, amount: float) -> None:
        """Deposit money into the account."""
        if amount > 0:
            self._balance += amount
        else:
            raise ValueError("Deposit amount must be positive")

    def withdraw(self, amount: float) -> None:
        """Withdraw money from the account."""
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

    @property
    def balance(self) -> float:
        """Get the current balance."""
        return self._balance

    def __str__(self) -> str:
        return f"Account {self.account_number}: ${self._balance:.2f}"

    def __repr__(self) -> str:
        return f"BankAccount('{self.account_number}', {self._balance})"
```

### Inheritance

```python
class SavingsAccount(BankAccount):
    """A savings account with interest."""

    def __init__(self, account_number: str, initial_balance: float = 0.0,
                 interest_rate: float = 0.01):
        super().__init__(account_number, initial_balance)
        self.interest_rate = interest_rate

    def apply_interest(self) -> None:
        """Apply interest to the account."""
        interest = self._balance * self.interest_rate
        self.deposit(interest)

    def __str__(self) -> str:
        return f"Savings {super().__str__()} @ {self.interest_rate*100:.1f}%"
```

### Abstract Base Classes

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    """Abstract base class for vehicles."""

    def __init__(self, brand: str, model: str):
        self.brand = brand
        self.model = model

    @abstractmethod
    def start_engine(self) -> None:
        """Start the vehicle's engine."""
        pass

    @abstractmethod
    def stop_engine(self) -> None:
        """Stop the vehicle's engine."""
        pass

    def honk(self) -> str:
        """Make a honking sound."""
        return "Beep beep!"

class Car(Vehicle):
    """Car implementation of Vehicle."""

    def start_engine(self) -> None:
        print(f"Starting {self.brand} {self.model} engine...")

    def stop_engine(self) -> None:
        print(f"Stopping {self.brand} {self.model} engine...")
```

### Dataclasses

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Product:
    """Product dataclass with automatic methods."""
    name: str
    price: float
    quantity: int = 0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Called after __init__."""
        if self.price < 0:
            raise ValueError("Price cannot be negative")

    @property
    def total_value(self) -> float:
        """Calculate total value of inventory."""
        return self.price * self.quantity

@dataclass(frozen=True)  # Immutable dataclass
class Point:
    """Immutable point in 2D space."""
    x: float
    y: float

    def distance_from_origin(self) -> float:
        """Calculate distance from origin."""
        return (self.x**2 + self.y**2)**0.5
```

### Properties and Descriptors

```python
class Temperature:
    """Temperature class with validation."""

    def __init__(self, celsius: float = 0):
        self._celsius = celsius

    @property
    def celsius(self) -> float:
        """Get temperature in Celsius."""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        """Set temperature in Celsius."""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """Get temperature in Fahrenheit."""
        return (self._celsius * 9/5) + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        """Set temperature in Fahrenheit."""
        self.celsius = (value - 32) * 5/9

    @property
    def kelvin(self) -> float:
        """Get temperature in Kelvin."""
        return self._celsius + 273.15
```

---

## Error Handling

### Exception Types

```python
# Built-in exceptions
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Division by zero: {e}")
except ValueError as e:
    print(f"Invalid value: {e}")
except TypeError as e:
    print(f"Type error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
else:
    print("No exception occurred")
finally:
    print("This always executes")
```

### Custom Exceptions

```python
class ApplicationError(Exception):
    """Base exception for application errors."""
    pass

class ValidationError(ApplicationError):
    """Exception for validation failures."""

    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)

class DatabaseError(ApplicationError):
    """Exception for database-related errors."""

    def __init__(self, message: str, query: str = None):
        self.message = message
        self.query = query
        super().__init__(self.message)

# Usage
def validate_email(email: str) -> None:
    """Validate email format."""
    if "@" not in email:
        raise ValidationError("Invalid email format", "email")
```

### Context Managers for Error Handling

```python
from contextlib import contextmanager

@contextmanager
def handle_errors():
    """Context manager for error handling."""
    try:
        yield
    except ValidationError as e:
        print(f"Validation error: {e.message}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

# Usage
with handle_errors():
    validate_email("invalid-email")
```

---

## File I/O & Path Operations

### Modern File Operations with Pathlib

```python
from pathlib import Path

# Path creation and manipulation
current_dir = Path.cwd()
home_dir = Path.home()
data_dir = Path("data")
file_path = data_dir / "users.txt"

# File operations
def read_file(file_path: Path) -> str:
    """Read entire file content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(file_path: Path, content: str) -> None:
    """Write content to file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def append_to_file(file_path: Path, content: str) -> None:
    """Append content to file."""
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(content)

# File information
def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get file information."""
    if not file_path.exists():
        return {"exists": False}

    stat = file_path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "extension": file_path.suffix,
        "stem": file_path.stem
    }
```

### Working with Different File Formats

```python
import json
import csv
from typing import Any, Dict, List

def read_json(file_path: Path) -> Any:
    """Read JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(file_path: Path, data: Any) -> None:
    """Write data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def read_csv(file_path: Path) -> List[Dict[str, Any]]:
    """Read CSV file as list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return list(csv.DictReader(file))

def write_csv(file_path: Path, data: List[Dict[str, Any]]) -> None:
    """Write data to CSV file."""
    if not data:
        return

    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
```

---

## Iterators & Generators

### Generators

```python
def fibonacci(n: int) -> Generator[int, None, None]:
    """Generate Fibonacci sequence."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def read_large_file(file_path: Path) -> Generator[str, None, None]:
    """Read large file line by line."""
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Generator expressions
squares = (x**2 for x in range(1000000))  # Memory efficient
filtered_data = (item for item in data if condition(item))
```

### Custom Iterators

```python
class Range:
    """Custom range iterator."""

    def __init__(self, start: int, end: int, step: int = 1):
        self.start = start
        self.end = end
        self.step = step

    def __iter__(self):
        return RangeIterator(self.start, self.end, self.step)

class RangeIterator:
    """Iterator for Range class."""

    def __init__(self, start: int, end: int, step: int):
        self.current = start
        self.end = end
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += self.step
        return value
```

---

## Functional Programming

### Higher-Order Functions

```python
from functools import reduce, partial

# Map, filter, reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
sum_all = reduce(lambda x, y: x + y, numbers)

# Partial application
def power(base: int, exponent: int) -> int:
    """Calculate base raised to exponent."""
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

# Function composition
def compose(*functions):
    """Compose multiple functions."""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# Usage
add_one = lambda x: x + 1
multiply_by_two = lambda x: x * 2
add_one_then_double = compose(multiply_by_two, add_one)
```

### Functional Programming Patterns

```python
from typing import Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')

def map_function(func: Callable[[T], U], items: List[T]) -> List[U]:
    """Functional map implementation."""
    return [func(item) for item in items]

def filter_function(predicate: Callable[[T], bool], items: List[T]) -> List[T]:
    """Functional filter implementation."""
    return [item for item in items if predicate(item)]

def reduce_function(func: Callable[[T, T], T], items: List[T], initial: T = None) -> T:
    """Functional reduce implementation."""
    iterator = iter(items)
    if initial is None:
        result = next(iterator)
    else:
        result = initial

    for item in iterator:
        result = func(result, item)
    return result
```

---

## Concurrency & Parallelism

### Threading

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def worker_function(name: str, duration: int) -> str:
    """Worker function for threading example."""
    print(f"Worker {name} starting")
    time.sleep(duration)
    print(f"Worker {name} finished")
    return f"Result from {name}"

# Basic threading
def basic_threading():
    """Demonstrate basic threading."""
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker_function, args=(f"Thread-{i}", i+1))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

# ThreadPoolExecutor
def thread_pool_example():
    """Demonstrate ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker_function, f"Worker-{i}", i+1)
                  for i in range(5)]

        for future in as_completed(futures):
            result = future.result()
            print(f"Got result: {result}")
```

### Async/Await

```python
import asyncio
from typing import List

async def fetch_data(url: str, delay: float = 1.0) -> str:
    """Simulate fetching data from URL."""
    await asyncio.sleep(delay)
    return f"Data from {url}"

async def fetch_multiple_urls(urls: List[str]) -> List[str]:
    """Fetch data from multiple URLs concurrently."""
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Context manager for async operations
class AsyncContextManager:
    """Async context manager example."""

    async def __aenter__(self):
        print("Entering async context")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Exiting async context")
        await asyncio.sleep(0.1)

# Usage
async def main():
    """Main async function."""
    urls = ["http://example.com", "http://google.com", "http://github.com"]

    async with AsyncContextManager():
        results = await fetch_multiple_urls(urls)
        for result in results:
            print(result)

# Run async code
# asyncio.run(main())
```

---

## Advanced Features

### Metaclasses

```python
class SingletonMeta(type):
    """Metaclass for singleton pattern."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    """Singleton database connection."""

    def __init__(self):
        self.connection = "database_connection"
        print("Database connection created")

# Usage
db1 = Database()
db2 = Database()
print(db1 is db2)  # True
```

### Descriptors

```python
class ValidatedAttribute:
    """Descriptor for validated attributes."""

    def __init__(self, validator: Callable[[Any], bool], name: str = None):
        self.validator = validator
        self.name = name

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}: {value}")
        instance.__dict__[self.name] = value

# Validators
def positive_number(value):
    """Validate positive number."""
    return isinstance(value, (int, float)) and value > 0

def non_empty_string(value):
    """Validate non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0

# Usage
class Product:
    """Product with validated attributes."""
    name = ValidatedAttribute(non_empty_string)
    price = ValidatedAttribute(positive_number)

    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price
```

### Protocol (Structural Typing)

```python
from typing import Protocol

class Drawable(Protocol):
    """Protocol for drawable objects."""

    def draw(self) -> None:
        """Draw the object."""
        ...

class Circle:
    """Circle that implements Drawable protocol."""

    def __init__(self, radius: float):
        self.radius = radius

    def draw(self) -> None:
        print(f"Drawing circle with radius {self.radius}")

class Square:
    """Square that implements Drawable protocol."""

    def __init__(self, side: float):
        self.side = side

    def draw(self) -> None:
        print(f"Drawing square with side {self.side}")

def draw_shape(shape: Drawable) -> None:
    """Draw any shape that implements Drawable protocol."""
    shape.draw()
```

---

## Testing & Debugging

### Unit Testing

```python
import unittest
from unittest.mock import Mock, patch

class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def divide(self, a: int, b: int) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestCalculator(unittest.TestCase):
    """Test cases for Calculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = Calculator()

    def test_add(self):
        """Test addition."""
        result = self.calculator.add(2, 3)
        self.assertEqual(result, 5)

    def test_divide(self):
        """Test division."""
        result = self.calculator.divide(10, 2)
        self.assertEqual(result, 5.0)

    def test_divide_by_zero(self):
        """Test division by zero."""
        with self.assertRaises(ValueError):
            self.calculator.divide(10, 0)

    @patch('builtins.print')
    def test_with_mock(self, mock_print):
        """Test with mock."""
        mock_print.return_value = None
        print("Hello, World!")
        mock_print.assert_called_once_with("Hello, World!")
```

### Debugging Tools

```python
import pdb
import traceback

def debug_function(x: int) -> int:
    """Function with debugging."""
    # Set breakpoint
    # pdb.set_trace()

    result = x * 2
    return result

def handle_exceptions():
    """Exception handling with traceback."""
    try:
        result = 10 / 0
    except Exception as e:
        print(f"Error: {e}")
        print("Traceback:")
        traceback.print_exc()

# Assertions for debugging
def validate_input(value: int) -> None:
    """Validate input with assertions."""
    assert isinstance(value, int), f"Expected int, got {type(value)}"
    assert value > 0, f"Expected positive value, got {value}"
```

---

## Performance & Optimization

### Caching

```python
from functools import lru_cache
import time

@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    """Fibonacci with caching."""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Manual caching
cache = {}

def fibonacci_manual(n: int) -> int:
    """Fibonacci with manual caching."""
    if n in cache:
        return cache[n]

    if n < 2:
        result = n
    else:
        result = fibonacci_manual(n-1) + fibonacci_manual(n-2)

    cache[n] = result
    return result
```

### Timing and Profiling

```python
import time
import cProfile
import timeit

def time_function(func, *args, **kwargs):
    """Time function execution."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{func.__name__} took {end - start:.4f} seconds")
    return result

# Context manager for timing
@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:.4f} seconds")

# Usage
with timer():
    time.sleep(1)

# Using timeit
def test_function():
    """Function to test."""
    return sum(range(1000))

execution_time = timeit.timeit(test_function, number=10000)
print(f"Average execution time: {execution_time/10000:.6f} seconds")
```

### Memory Optimization

```python
import sys
from typing import Iterator

def memory_efficient_processing(data: Iterator[str]) -> Iterator[str]:
    """Process data in memory-efficient way."""
    for item in data:
        # Process item
        processed = item.upper().strip()
        yield processed

# Using slots for memory efficiency
class Point:
    """Memory-efficient Point class."""
    __slots__ = ['x', 'y']

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# Generator expressions vs list comprehensions
# Memory efficient
data_gen = (x**2 for x in range(1000000))

# Memory intensive
# data_list = [x**2 for x in range(1000000)]
```

---

## Logging

### Logging Setup

```python
import logging
from logging.handlers import RotatingFileHandler

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Custom logger
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: str = "app.log"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper())

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

# Usage
def process_data(data: List[Any]) -> None:
    """Process data with logging."""
    logger.info(f"Processing {len(data)} items")

    try:
        for item in data:
            logger.debug(f"Processing item: {item}")
            # Process item
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
```

---

## Regular Expressions

### Common Patterns

```python
import re
from typing import List, Optional

# Compiled patterns for efficiency
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})
PHONE_PATTERN = re.compile(r'^\+?1?\d{9,15})
URL_PATTERN = re.compile(r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?')
IP_PATTERN = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))

def validate_email(email: str) -> bool:
    """Validate email format."""
    return bool(EMAIL_PATTERN.match(email))

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    return URL_PATTERN.findall(text)

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def extract_numbers(text: str) -> List[float]:
    """Extract numbers from text."""
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]

# Advanced regex operations
def replace_with_function(text: str, pattern: str, replacement_func: callable) -> str:
    """Replace matches using a function."""
    return re.sub(pattern, replacement_func, text)

# Example: Convert to title case
def to_title_case(match):
    """Convert match to title case."""
    return match.group(0).title()

# Usage
text = "hello world python programming"
result = replace_with_function(text, r'\b\w+\b', to_title_case)
```

---

## Professional Best Practices

### Code Organization

```python
# Project structure
"""
project/
├── src/
│   ├── __init__.py
│   ├
```
