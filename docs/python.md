# Complete Python Cheat Sheet & Professional Practices

## Table of Contents

1. [Python Basics](#python-basics)
2. [Data Types & Variables](#data-types--variables)
3. [Control Flow](#control-flow)
4. [Functions](#functions)
5. [Data Structures](#data-structures)
6. [Object-Oriented Programming](#object-oriented-programming)
7. [Error Handling](#error-handling)
8. [File I/O](#file-io)
9. [Modules & Packages](#modules--packages)
10. [Built-in Functions](#built-in-functions)
11. [List/Dict/Set Comprehensions](#listdictset-comprehensions)
12. [Generators & Iterators](#generators--iterators)
13. [Decorators](#decorators)
14. [Context Managers](#context-managers)
15. [Regular Expressions](#regular-expressions)
16. [Testing](#testing)
17. [Performance & Optimization](#performance--optimization)
18. [Professional Best Practices](#professional-best-practices)
19. [Code Style & Standards](#code-style--standards)
20. [Advanced Topics](#advanced-topics)

---

## Python Basics

### Comments

```python
# Single line comment
"""
Multi-line comment
or docstring
"""
```

### Print & Input

```python
print("Hello, World!")
print("Name:", name, "Age:", age)
print(f"Name: {name}, Age: {age}")  # f-strings (preferred)
name = input("Enter your name: ")
age = int(input("Enter your age: "))
```

### Variables & Assignment

```python
# Variable assignment
x = 5
y, z = 10, 20  # Multiple assignment
x, y = y, x    # Swap values

# Constants (by convention, use UPPERCASE)
PI = 3.14159
MAX_SIZE = 100
```

---

## Data Types & Variables

### Basic Data Types

```python
# Integers
num = 42
binary = 0b1010  # 10 in binary
octal = 0o12     # 10 in octal
hex_num = 0xA    # 10 in hexadecimal

# Floats
pi = 3.14159
scientific = 1.5e-4  # 0.00015

# Strings
name = "Alice"
message = 'Hello'
multiline = """This is a
multiline string"""

# Booleans
is_valid = True
is_empty = False

# None
result = None
```

### String Operations

```python
# String methods
text = "Hello, World!"
print(text.upper())      # HELLO, WORLD!
print(text.lower())      # hello, world!
print(text.title())      # Hello, World!
print(text.strip())      # Remove whitespace
print(text.replace("Hello", "Hi"))  # Hi, World!
print(text.split(","))   # ['Hello', ' World!']

# String formatting
name = "Alice"
age = 30
print(f"Name: {name}, Age: {age}")  # f-strings (Python 3.6+)
print("Name: {}, Age: {}".format(name, age))  # .format()
print("Name: %s, Age: %d" % (name, age))  # % formatting (old style)

# String slicing
text = "Python"
print(text[0])      # P
print(text[-1])     # n
print(text[1:4])    # yth
print(text[:3])     # Pyt
print(text[3:])     # hon
print(text[::-1])   # nohtyP (reverse)
```

### Type Conversion

```python
# Convert between types
str_num = "42"
int_num = int(str_num)
float_num = float(str_num)
str_again = str(int_num)

# Check types
print(type(42))         # <class 'int'>
print(isinstance(42, int))  # True
```

---

## Control Flow

### Conditional Statements

```python
# if-elif-else
age = 18
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# Ternary operator
status = "Adult" if age >= 18 else "Minor"

# Multiple conditions
if age >= 18 and age < 65:
    print("Working age")

if name == "Alice" or name == "Bob":
    print("Known person")
```

### Loops

```python
# for loop
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

for i in range(1, 6):
    print(i)  # 1, 2, 3, 4, 5

for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

# Iterating over sequences
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)

# Enumerate for index and value
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# while loop
count = 0
while count < 5:
    print(count)
    count += 1

# Loop control
for i in range(10):
    if i == 3:
        continue  # Skip this iteration
    if i == 7:
        break     # Exit loop
    print(i)

# else clause with loops
for i in range(5):
    print(i)
else:
    print("Loop completed normally")  # Only if no break
```

---

## Functions

### Function Definition

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Function with default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Function with multiple parameters
def add(a, b):
    return a + b

# Function with *args and **kwargs
def flexible_func(*args, **kwargs):
    print("Args:", args)
    print("Kwargs:", kwargs)

flexible_func(1, 2, 3, name="Alice", age=30)

# Type hints (Python 3.5+)
def add_numbers(a: int, b: int) -> int:
    return a + b

# Function with optional parameters
def create_user(name: str, email: str = None, age: int = None) -> dict:
    user = {"name": name}
    if email:
        user["email"] = email
    if age:
        user["age"] = age
    return user
```

### Lambda Functions

```python
# Lambda (anonymous) functions
square = lambda x: x ** 2
add = lambda x, y: x + y

# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
even = list(filter(lambda x: x % 2 == 0, numbers))
```

### Higher-Order Functions

```python
# Functions as arguments
def apply_operation(func, x, y):
    return func(x, y)

result = apply_operation(lambda a, b: a + b, 5, 3)  # 8

# Functions returning functions
def create_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

double = create_multiplier(2)
print(double(5))  # 10
```

---

## Data Structures

### Lists

```python
# List creation
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# List operations
fruits.append("grape")          # Add to end
fruits.insert(1, "kiwi")        # Insert at index
fruits.remove("banana")         # Remove by value
popped = fruits.pop()           # Remove and return last
fruits.extend(["mango", "pear"]) # Add multiple items

# List slicing
print(fruits[1:3])    # Elements from index 1 to 2
print(fruits[:2])     # First 2 elements
print(fruits[2:])     # From index 2 to end
print(fruits[-2:])    # Last 2 elements

# List methods
fruits.sort()                   # Sort in place
sorted_fruits = sorted(fruits)  # Return new sorted list
fruits.reverse()                # Reverse in place
count = fruits.count("apple")   # Count occurrences
index = fruits.index("apple")   # Find index

# List comprehensions
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

### Tuples

```python
# Tuple creation
point = (3, 4)
colors = ("red", "green", "blue")
single_item = (42,)  # Note the comma

# Tuple unpacking
x, y = point
first, *rest = colors  # first="red", rest=["green", "blue"]

# Named tuples
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
print(p.x, p.y)  # 3 4
```

### Dictionaries

```python
# Dictionary creation
person = {"name": "Alice", "age": 30, "city": "New York"}
empty_dict = {}
dict_from_keys = dict.fromkeys(["a", "b", "c"], 0)

# Dictionary operations
person["email"] = "alice@example.com"  # Add/update
age = person.get("age", 0)             # Get with default
person.setdefault("country", "USA")    # Set if not exists
person.update({"phone": "123-456-7890", "age": 31})

# Dictionary methods
keys = person.keys()
values = person.values()
items = person.items()

# Dictionary comprehensions
squares = {x: x**2 for x in range(5)}
filtered = {k: v for k, v in person.items() if isinstance(v, str)}
```

### Sets

```python
# Set creation
fruits = {"apple", "banana", "orange"}
numbers = set([1, 2, 3, 4, 5])
empty_set = set()  # Note: {} creates an empty dict

# Set operations
fruits.add("grape")
fruits.remove("banana")  # KeyError if not found
fruits.discard("kiwi")   # No error if not found

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
union = set1 | set2           # {1, 2, 3, 4, 5, 6}
intersection = set1 & set2    # {3, 4}
difference = set1 - set2      # {1, 2}
symmetric_diff = set1 ^ set2  # {1, 2, 5, 6}

# Set comprehensions
even_squares = {x**2 for x in range(10) if x % 2 == 0}
```

---

## Object-Oriented Programming

### Classes and Objects

```python
class Person:
    # Class variable
    species = "Homo sapiens"

    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age

    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."

    def have_birthday(self):
        self.age += 1

    # String representation
    def __str__(self):
        return f"Person(name='{self.name}', age={self.age})"

    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

# Creating objects
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

print(person1.introduce())
person1.have_birthday()
print(person1)
```

### Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

    def info(self):
        return f"This is {self.name}"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Call parent constructor
        self.breed = breed

    def speak(self):
        return "Woof!"

    def info(self):
        return f"{super().info()}, a {self.breed}"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Usage
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers")

print(dog.speak())  # Woof!
print(cat.speak())  # Meow!
print(dog.info())   # This is Buddy, a Golden Retriever
```

### Advanced OOP Concepts

```python
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self.account_number = account_number
        self._balance = initial_balance  # Protected attribute
        self.__pin = None               # Private attribute

    @property
    def balance(self):
        """Getter for balance"""
        return self._balance

    @balance.setter
    def balance(self, amount):
        """Setter for balance with validation"""
        if amount < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = amount

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount

    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount
            return True
        return False

    @staticmethod
    def validate_account_number(account_number):
        """Static method - doesn't need instance"""
        return len(account_number) == 10

    @classmethod
    def create_savings_account(cls, account_number):
        """Class method - alternative constructor"""
        return cls(account_number, 100)  # Savings starts with $100

# Usage
account = BankAccount("1234567890", 1000)
print(account.balance)  # 1000
account.deposit(500)
print(account.balance)  # 1500

savings = BankAccount.create_savings_account("0987654321")
print(BankAccount.validate_account_number("1234567890"))  # True
```

### Abstract Classes and Interfaces

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Usage
rect = Rectangle(5, 3)
circle = Circle(4)

print(f"Rectangle area: {rect.area()}")
print(f"Circle area: {circle.area()}")
```

---

## Error Handling

### Try-Except Blocks

```python
# Basic exception handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid input! Please enter a number.")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Catching multiple exceptions
try:
    # Some risky operation
    pass
except (ValueError, TypeError) as e:
    print(f"Error occurred: {e}")

# Catching all exceptions
try:
    # Some risky operation
    pass
except Exception as e:
    print(f"Unexpected error: {e}")

# Try-except-else-finally
try:
    file = open("data.txt", "r")
except FileNotFoundError:
    print("File not found!")
else:
    # Executed if no exception occurred
    content = file.read()
    print("File read successfully!")
finally:
    # Always executed
    if 'file' in locals():
        file.close()
```

### Custom Exceptions

```python
class CustomError(Exception):
    """Custom exception class"""
    pass

class ValidationError(Exception):
    """Exception for validation errors"""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

def validate_age(age):
    if age < 0:
        raise ValidationError("Age cannot be negative", code="NEGATIVE_AGE")
    if age > 150:
        raise ValidationError("Age seems unrealistic", code="UNREALISTIC_AGE")

# Usage
try:
    validate_age(-5)
except ValidationError as e:
    print(f"Validation error: {e}")
    print(f"Error code: {e.code}")
```

---

## File I/O

### File Operations

```python
# Reading files
with open("file.txt", "r") as file:
    content = file.read()          # Read entire file

with open("file.txt", "r") as file:
    lines = file.readlines()       # Read all lines as list

with open("file.txt", "r") as file:
    for line in file:              # Read line by line
        print(line.strip())

# Writing files
with open("output.txt", "w") as file:
    file.write("Hello, World!")

with open("output.txt", "a") as file:  # Append mode
    file.write("\nNew line")

# Writing multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("output.txt", "w") as file:
    file.writelines(lines)

# Binary file operations
with open("image.jpg", "rb") as file:
    binary_data = file.read()

with open("copy.jpg", "wb") as file:
    file.write(binary_data)
```

### Working with CSV

```python
import csv

# Reading CSV
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Reading CSV with headers
with open("data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row["name"], row["age"])

# Writing CSV
data = [
    ["Name", "Age", "City"],
    ["Alice", 30, "New York"],
    ["Bob", 25, "Los Angeles"]
]

with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Writing CSV with DictWriter
fieldnames = ["name", "age", "city"]
with open("output.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"name": "Alice", "age": 30, "city": "New York"})
```

### JSON Operations

```python
import json

# Reading JSON
with open("data.json", "r") as file:
    data = json.load(file)

# Writing JSON
data = {"name": "Alice", "age": 30, "city": "New York"}
with open("output.json", "w") as file:
    json.dump(data, file, indent=2)

# JSON string operations
json_string = '{"name": "Alice", "age": 30}'
data = json.loads(json_string)
json_string = json.dumps(data, indent=2)
```

---

## Modules & Packages

### Importing Modules

```python
# Different ways to import
import math
import math as m
from math import sqrt, pi
from math import *  # Not recommended

# Using imported modules
print(math.sqrt(16))  # 4.0
print(m.pi)          # 3.141592653589793
print(sqrt(25))      # 5.0

# Importing from packages
from collections import defaultdict, Counter
from datetime import datetime, timedelta
```

### Creating Modules

```python
# mymodule.py
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

PI = 3.14159

if __name__ == "__main__":
    # Code that runs only when module is executed directly
    print("Module is being run directly")
```

### Package Structure

```
mypackage/
    __init__.py
    module1.py
    module2.py
    subpackage/
        __init__.py
        submodule.py
```

```python
# __init__.py
from .module1 import function1
from .module2 import function2

__all__ = ['function1', 'function2']
```

---

## Built-in Functions

### Essential Built-ins

```python
# Type and conversion functions
print(type(42))           # <class 'int'>
print(isinstance(42, int)) # True
print(int("42"))          # 42
print(float(42))          # 42.0
print(str(42))            # "42"
print(bool(0))            # False

# Sequence functions
numbers = [1, 2, 3, 4, 5]
print(len(numbers))       # 5
print(max(numbers))       # 5
print(min(numbers))       # 1
print(sum(numbers))       # 15
print(sorted(numbers, reverse=True))  # [5, 4, 3, 2, 1]

# Range and enumerate
for i in range(5):
    print(i)

for index, value in enumerate(["a", "b", "c"]):
    print(index, value)

# Zip
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Map, filter, reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
even = list(filter(lambda x: x % 2 == 0, numbers))

from functools import reduce
product = reduce(lambda x, y: x * y, numbers)  # 120

# Any and all
print(any([True, False, False]))   # True
print(all([True, True, False]))    # False

# Abs, round, pow
print(abs(-5))        # 5
print(round(3.14159, 2))  # 3.14
print(pow(2, 3))      # 8
```

---

## List/Dict/Set Comprehensions

### List Comprehensions

```python
# Basic list comprehension
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# Nested comprehensions
matrix = [[i*j for j in range(3)] for i in range(3)]
# [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

# Flattening a matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# String manipulation
words = ["hello", "world", "python"]
upper_words = [word.upper() for word in words]
# ['HELLO', 'WORLD', 'PYTHON']
```

### Dictionary Comprehensions

```python
# Basic dictionary comprehension
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# With condition
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
# {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# From two lists
keys = ["a", "b", "c"]
values = [1, 2, 3]
dict_from_lists = {k: v for k, v in zip(keys, values)}
# {'a': 1, 'b': 2, 'c': 3}

# Transforming existing dictionary
original = {"a": 1, "b": 2, "c": 3}
doubled = {k: v*2 for k, v in original.items()}
# {'a': 2, 'b': 4, 'c': 6}
```

### Set Comprehensions

```python
# Basic set comprehension
unique_squares = {x**2 for x in range(-5, 6)}
# {0, 1, 4, 9, 16, 25}

# With condition
even_nums = {x for x in range(20) if x % 2 == 0}
# {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
```

---

## Generators & Iterators

### Generators

```python
# Generator function
def count_up_to(max_count):
    count = 1
    while count <= max_count:
        yield count
        count += 1

# Using generator
counter = count_up_to(3)
for num in counter:
    print(num)  # 1, 2, 3

# Generator expression
squares = (x**2 for x in range(10))
print(next(squares))  # 0
print(next(squares))  # 1

# Infinite generator
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
for _ in range(10):
    print(next(fib))  # First 10 Fibonacci numbers
```

### Iterators

```python
# Custom iterator
class CountDown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# Usage
countdown = CountDown(3)
for num in countdown:
    print(num)  # 3, 2, 1

# Using itertools
import itertools

# Infinite iterators
counter = itertools.count(1, 2)  # 1, 3, 5, 7, ...
repeater = itertools.repeat("hello", 3)  # "hello", "hello", "hello"

# Finite iterators
numbers = [1, 2, 3, 4, 5]
accumulated = list(itertools.accumulate(numbers))  # [1, 3, 6, 10, 15]
combinations = list(itertools.combinations(numbers, 2))  # [(1, 2), (1, 3), ...]
```

---

## Decorators

### Basic Decorators

```python
# Simple decorator
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# Equivalent to: say_hello = my_decorator(say_hello)
say_hello()

# Decorator with arguments
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished calling {func.__name__}")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

result = add(5, 3)
```

### Advanced Decorators

```python
import functools
import time

# Decorator with parameters
def repeat(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

# Timing decorator
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
    return "Done"

# Class-based decorator
class CallCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)

@CallCounter
def say_hello():
    print("Hello!")

# Property decorator
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self):
        return 3.14159 * self._radius ** 2
```

---

## Context Managers

### Using Context Managers

```python
# File handling with context manager
with open("file.txt", "r") as file:
    content = file.read()
# File is automatically closed

# Multiple context managers
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    content = infile.read()
    outfile.write(content.upper())
```

### Creating Context Managers

```python
# Using contextlib
from contextlib import contextmanager

@contextmanager
def my_context():
    print("Entering context")
    try:
        yield "Hello from context"
    finally:
        print("Exiting context")

with my_context() as value:
    print(value)

# Class-based context manager
class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None

    def __enter__(self):
        print(f"Connecting to {self.host}:{self.port}")
        self.connection = f"Connected to {self.host}:{self.port}"
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        self.connection = None

# Usage
with DatabaseConnection("localhost", 5432) as conn:
    print(f"Using connection: {conn}")

# Context manager for timing
@contextmanager
def timer():
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"Execution took {end - start:.4f} seconds")

with timer():
    # Some time-consuming operation
    sum(range(1000000))
```

---

## Regular Expressions

### Basic Regex Operations

```python
import re

# Basic matching
text = "Hello, World! My phone number is 123-456-7890."
pattern = r"\d{3}-\d{3}-\d{4}"
match = re.search(pattern, text)
if match:
    print(f"Found: {match.group()}")  # Found: 123-456-7890

# Finding all matches
text = "Email me at john@example.com or jane@test.org"
pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
emails = re.findall(pattern, text)
print(emails)  # ['john@example.com', 'jane@test.org']

# Substitution
text = "The price is $100 and the tax is $10"
pattern = r"\$(\d+)"
new_text = re.sub(pattern, r"£\1", text)
print(new_text)  # The price is £100 and the tax is £10

# Splitting
text = "apple,banana;orange:grape"
fruits = re.split(r"[,;:]", text)
print(fruits)  # ['apple', 'banana', 'orange', 'grape']
```

### Advanced Regex

```python
# Compiled patterns (more efficient for repeated use)
pattern = re.compile(r"\d+")
matches = pattern.findall("I have 5 apples and 3 oranges")
print(matches)  # ['5', '3']

# Groups and capturing
text = "John Smith (age 30) and Jane Doe (age 25)"
pattern = r"(\w+) (\w+) \(age (\d+)\)"
matches = re.findall(pattern, text)
for match in matches:
    first, last, age = match
    print(f"{first} {last} is {age} years old")

# Named groups
pattern = r"(?P<first>\w+) (?P<last>\w+) \(age (?P<age>\d+)\)"
for match in re.finditer(pattern, text):
    print(f"{match.group('first')} {match.group('last')} is {match.group('age')}")

# Lookahead and lookbehind
text = "password123, password456, pass123"
# Positive lookahead: match "password" only if followed by digits
pattern = r"password(?=\d+)"
matches = re.findall(pattern, text)
print(matches)  # ['password', 'password']

# Common regex patterns
patterns = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}-\d{3}-\d{4}\b",
    "url": r"https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?",
    "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
}
```

---

## Testing

### Unit Testing with unittest

```python
import unittest

def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class TestMathFunctions(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)

    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)
        self.assertEqual(divide(9, 3), 3)

        # Test exception
        with self.assertRaises(ValueError):
            divide(10, 0)

    def setUp(self):
        """Run before each test method"""
        self.test_data = [1, 2, 3, 4, 5]

    def tearDown(self):
        """Run after each test method"""
        pass

    def test_list_operations(self):
        self.assertIn(3, self.test_data)
        self.assertNotIn(10, self.test_data)
        self.assertEqual(len(self.test_data), 5)

if __name__ == "__main__":
    unittest.main()
```

### Testing with pytest

```python
import pytest

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_divide():
    assert divide(10, 2) == 5
    assert divide(9, 3) == 3

    with pytest.raises(ValueError):
        divide(10, 0)

# Fixtures
@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_with_fixture(sample_data):
    assert len(sample_data) == 5
    assert 3 in sample_data

# Parametrized tests
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected
```

### Mocking

```python
from unittest.mock import Mock, patch

# Mock object
mock_obj = Mock()
mock_obj.method.return_value = "mocked result"
result = mock_obj.method()
print(result)  # mocked result

# Patching
import requests

def get_user_data(user_id):
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()

# Test with mock
@patch('requests.get')
def test_get_user_data(mock_get):
    mock_response = Mock()
    mock_response.json.return_value = {"name": "John", "id": 123}
    mock_get.return_value = mock_response

    result = get_user_data(123)
    assert result["name"] == "John"
    mock_get.assert_called_once_with("https://api.example.com/users/123")
```

---

## Performance & Optimization

### Measuring Performance

```python
import time
import timeit
from functools import wraps

# Simple timing
start = time.time()
# Some operation
sum(range(1000000))
end = time.time()
print(f"Execution time: {end - start:.4f} seconds")

# Using timeit
execution_time = timeit.timeit(lambda: sum(range(1000)), number=10000)
print(f"Average execution time: {execution_time/10000:.6f} seconds")

# Profiling decorator
def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@profile
def slow_function():
    return sum(range(1000000))
```

### Memory Optimization

```python
import sys
from collections import deque

# Check memory usage
numbers = [1, 2, 3, 4, 5]
print(f"Size of list: {sys.getsizeof(numbers)} bytes")

# Use generators for large datasets
def large_dataset():
    for i in range(1000000):
        yield i * 2

# Memory-efficient operations
# Use deque for frequent insertions/deletions at both ends
d = deque([1, 2, 3, 4, 5])
d.appendleft(0)  # O(1) operation
d.pop()          # O(1) operation

# Use __slots__ to reduce memory usage in classes
class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

# String concatenation optimization
# Bad: O(n²) complexity
result = ""
for i in range(1000):
    result += str(i)

# Good: O(n) complexity
result = "".join(str(i) for i in range(1000))
```

### Algorithm Optimization

```python
import bisect
from collections import Counter, defaultdict

# Use set for membership testing (O(1) vs O(n))
large_list = list(range(10000))
large_set = set(large_list)

# Slow: O(n)
if 5000 in large_list:
    pass

# Fast: O(1)
if 5000 in large_set:
    pass

# Use bisect for sorted lists
sorted_list = [1, 3, 5, 7, 9, 11, 13, 15]
index = bisect.bisect_left(sorted_list, 7)  # O(log n)

# Use Counter for counting
items = ["apple", "banana", "apple", "orange", "banana", "apple"]
count = Counter(items)
print(count.most_common(2))  # [('apple', 3), ('banana', 2)]

# Use defaultdict to avoid key checks
dd = defaultdict(list)
dd['key'].append('value')  # No need to check if key exists
```

---

## Professional Best Practices

### Code Organization

```python
# Module structure
"""
Module docstring explaining the purpose of the module.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Standard library imports
import os
import sys
from datetime import datetime

# Third-party imports
import requests
import numpy as np

# Local imports
from .utils import helper_function
from .models import User

# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# Module-level variables
logger = logging.getLogger(__name__)

class MyClass:
    """Class docstring."""

    def __init__(self):
        """Initialize the class."""
        pass

    def public_method(self):
        """Public method docstring."""
        return self._private_method()

    def _private_method(self):
        """Private method (convention)."""
        return "private"

def main():
    """Main function."""
    pass

if __name__ == "__main__":
    main()
```

### Error Handling Best Practices

```python
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_function(data):
    """
    A function that handles errors gracefully.

    Args:
        data: Input data to process

    Returns:
        Processed data

    Raises:
        ValueError: If data is invalid
        TypeError: If data is wrong type
    """
    if not isinstance(data, (list, tuple)):
        raise TypeError("Data must be a list or tuple")

    if not data:
        raise ValueError("Data cannot be empty")

    try:
        result = process_data(data)
        logger.info(f"Successfully processed {len(data)} items")
        return result
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise
    finally:
        # Cleanup code here
        pass

def process_data(data):
    """Process the data."""
    return [item * 2 for item in data]

# Configuration management
class Config:
    """Configuration class."""

    def __init__(self):
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///default.db")
        self.api_key = os.getenv("API_KEY")

        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")

config = Config()
```

### Type Hints and Documentation

```python
from typing import List, Dict, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum

class Status(Enum):
    """Status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class User:
    """User data class."""
    id: int
    name: str
    email: str
    age: Optional[int] = None
    is_active: bool = True

def process_users(
    users: List[User],
    filter_func: Optional[Callable[[User], bool]] = None,
    sort_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a list of users.

    Args:
        users: List of User objects to process
        filter_func: Optional function to filter users
        sort_key: Optional attribute name to sort by

    Returns:
        Dictionary containing processing results

    Example:
        >>> users = [User(1, "Alice", "alice@example.com")]
        >>> result = process_users(users, sort_key="name")
        >>> print(result["count"])
        1
    """
    if filter_func:
        users = [user for user in users if filter_func(user)]

    if sort_key:
        users.sort(key=lambda u: getattr(u, sort_key))

    return {
        "count": len(users),
        "users": users,
        "processed_at": datetime.now().isoformat()
    }

# Generic type hints
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    """Generic stack implementation."""

    def __init__(self) -> None:
        self._items: List[T] = []

    def push(self, item: T) -> None:
        """Push an item onto the stack."""
        self._items.append(item)

    def pop(self) -> T:
        """Pop an item from the stack."""
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()

    def peek(self) -> T:
        """Peek at the top item without removing it."""
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items[-1]
```

---

## Code Style & Standards

### PEP 8 Guidelines

```python
# Good naming conventions
class UserManager:  # PascalCase for classes
    def __init__(self):
        self.user_count = 0  # snake_case for variables
        self._private_attr = None  # Leading underscore for private
        self.__very_private = None  # Double underscore for name mangling

    def get_user_by_id(self, user_id):  # snake_case for functions
        """Get user by ID."""
        pass

    def _helper_method(self):  # Private method
        """Helper method."""
        pass

# Constants
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30

# Good spacing and formatting
def calculate_total(items, tax_rate=0.1, discount=0.0):
    """Calculate total with tax and discount."""
    subtotal = sum(item.price for item in items)
    discount_amount = subtotal * discount
    taxable_amount = subtotal - discount_amount
    tax_amount = taxable_amount * tax_rate
    total = taxable_amount + tax_amount

    return {
        'subtotal': subtotal,
        'discount': discount_amount,
        'tax': tax_amount,
        'total': total
    }

# Line length and formatting
very_long_variable_name = some_function_with_a_very_long_name(
    first_argument,
    second_argument,
    third_argument,
    fourth_argument
)

# List formatting
items = [
    'first_item',
    'second_item',
    'third_item',
    'fourth_item',
]

# Dictionary formatting
config = {
    'database_url': 'postgresql://localhost/mydb',
    'redis_url': 'redis://localhost:6379',
    'debug': True,
    'max_connections': 100,
}
```

### Code Quality Tools

```python
# Using black for formatting
# pip install black
# black my_file.py

# Using flake8 for linting
# pip install flake8
# flake8 my_file.py

# Using mypy for type checking
# pip install mypy
# mypy my_file.py

# Using isort for import sorting
# pip install isort
# isort my_file.py

# Configuration in pyproject.toml
"""
[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
"""
```

---

## Advanced Topics

### Metaclasses

```python
class SingletonMeta(type):
    """Metaclass that creates a Singleton base class."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    """Singleton database connection."""

    def __init__(self):
        self.connection = "Connected to database"

# Usage
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # True
```

### Descriptors

```python
class ValidatedAttribute:
    """Descriptor that validates attribute values."""

    def __init__(self, validator):
        self.validator = validator
        self.name = None

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

class Person:
    name = ValidatedAttribute(lambda x: isinstance(x, str) and len(x) > 0)
    age = ValidatedAttribute(lambda x: isinstance(x, int) and 0 <= x <= 150)

    def __init__(self, name, age):
        self.name = name
        self.age = age

# Usage
person = Person("Alice", 30)
# person.age = -5  # Raises ValueError
```

### Async Programming

```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    """Fetch a single URL."""
    async with session.get(url) as response:
        return await response.text()

async def fetch_multiple_urls(urls):
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Generator-based coroutines
async def async_generator():
    """Async generator example."""
    for i in range(5):
        await asyncio.sleep(0.1)
        yield i

async def main():
    """Main async function."""
    # Using async generator
    async for value in async_generator():
        print(value)

    # Concurrent execution
    urls = ["http://example.com", "http://google.com"]
    results = await fetch_multiple_urls(urls)
    print(f"Fetched {len(results)} URLs")

# Run async code
# asyncio.run(main())
```

### Data Classes and Enums

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List
import json

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class Status(Enum):
    TODO = auto()
    IN_PROGRESS = auto()
    DONE = auto()

@dataclass
class Task:
    title: str
    description: str = ""
    priority: Priority = Priority.MEDIUM
    status: Status = Status.TODO
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Called after initialization."""
        if not self.title:
            raise ValueError("Title cannot be empty")

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'title': self.title,
            'description': self.description,
            'priority': self.priority.value,
            'status': self.status.name,
            'tags': self.tags
        }

# Usage
task = Task(
    title="Learn Python",
    description="Complete Python tutorial",
    priority=Priority.HIGH,
    tags=["learning", "python"]
)

print(task.to_dict())
```

### Working with APIs

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class APIClient:
    """Robust API client with retry logic."""

    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.session = requests.Session()

        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'MyApp/1.0'
        })

    def get(self, endpoint, params=None):
        """Make GET request."""
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def post(self, endpoint, data=None):
        """Make POST request."""
        url = f"{self.base_url}/{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

# Usage
client = APIClient("https://api.example.com", "your-api-key")
users = client.get("users", params={"limit": 10})
```

### Database Operations

```python
import sqlite3
from contextlib import contextmanager

class DatabaseManager:
    """Database manager with context manager support."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def create_user(self, name, email):
        """Create a new user."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                'INSERT INTO users (name, email) VALUES (?, ?)',
                (name, email)
            )
            return cursor.lastrowid

    def get_user(self, user_id):
        """Get user by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM users WHERE id = ?',
                (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

# Usage
db = DatabaseManager("users.db")
user_id = db.create_user("Alice", "alice@example.com")
user = db.get_user(user_id)
```

---

## Quick Reference

### Common Patterns

```python
# Singleton pattern
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Factory pattern
class ShapeFactory:
    @staticmethod
    def create_shape(shape_type, **kwargs):
        if shape_type == "circle":
            return Circle(kwargs["radius"])
        elif shape_type == "rectangle":
            return Rectangle(kwargs["width"], kwargs["height"])
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

# Observer pattern
class Observable:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, event):
        for observer in self._observers:
            observer.update(event)

# Chain of responsibility
class Handler:
    def __init__(self, successor=None):
        self.successor = successor

    def handle(self, request):
        if self.can_handle(request):
            return self.process(request)
        elif self.successor:
            return self.successor.handle(request)
        else:
            raise ValueError("No handler available")

    def can_handle(self, request):
        raise NotImplementedError

    def process(self, request):
        raise NotImplementedError
```

### Performance Tips

```python
# Use local variables in loops
def slow_function():
    for i in range(1000):
        math.sqrt(i)  # Global lookup

def fast_function():
    sqrt = math.sqrt  # Local variable
    for i in range(1000):
        sqrt(i)

# Use __slots__ for memory efficiency
class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

# Use collections.deque for queues
from collections import deque
queue = deque()
queue.append(1)  # O(1)
queue.popleft()  # O(1)

# Use dict.get() instead of try/except for optional keys
# Slow
try:
    value = dictionary['key']
except KeyError:
    value = default_value

# Fast
value = dictionary.get('key', default_value)

# Use list comprehensions instead of loops
# Slow
result = []
for i in range(10):
    result.append(i * 2)

# Fast
result = [i * 2 for i in range(10)]
```

---

## Conclusion

This comprehensive Python cheat sheet covers fundamental concepts, advanced features, and professional best practices. Remember these key principles:

1. **Write readable code** - Code is read more often than it's written
2. **Follow PEP 8** - Consistent style improves maintainability
3. **Use type hints** - They improve code documentation and catch errors
4. **Handle errors gracefully** - Anticipate and handle edge cases
5. **Test your code** - Automated tests prevent regressions
6. **Profile before optimizing** - Measure performance bottlenecks
7. **Document your code** - Clear docstrings and comments help others
8. **Use virtual environments** - Isolate project dependencies
9. **Keep learning** - Python ecosystem is constantly evolving
10. **Practice regularly** - Consistent coding improves skills

Remember: The best code is not just functional, but also readable, maintainable, and efficient. Happy coding!
