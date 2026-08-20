# Python Decorators — Quick Reference

## Basic decorator

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@decorator
def greet(name):
    return f"hello {name}"
```

## Decorator with arguments

```python
# decorator with arguments — factory returning the actual decorator
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = None
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hi():
    print("hi")

# Generic pattern
def decorator_with_args(arg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # use arg here
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

## Preserving metadata

```python
from functools import wraps

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper docstring"""
        return func(*args, **kwargs)
    return wrapper
```

## Common patterns

```python
# Class decorator and stacking
def bold(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper
```
