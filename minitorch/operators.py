"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

def id(x: float) -> float:
    """Identity function."""
    return x

def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y

def neg(x: float) -> float:
    """Negate a number."""
    return -x

def lt(x: float, y: float) -> bool:
    """Check if x is less than y."""
    return x < y

def eq(x: float, y: float) -> bool:
    """Check if x is equal to y."""
    return x == y

def max(x: float, y: float) -> float:
    """Return the maximum of x and y."""
    return x if x > y else y

def abs(x: float) -> float:
    """Return the absolute value of x."""
    return -x if x < 0 else x

def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close to each other."""
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    """Compute the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
    
def relu(x: float) -> float:
    """Compute the ReLU function."""
    return max(0.0, x)

def log(x: float) -> float:
    """Compute the natural logarithm of x."""
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive values.")
    return math.log(x)

def exp(x: float) -> float:
    """Compute the exponential of x."""
    return math.exp(x)

def log_back(x: float, y: float) -> float:
    """Compute the derivative of the logarithm function."""
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive values.")
    return y / x

def inv(x: float) -> float:
    """Compute the inverse of x."""
    if x == 0:
        raise ValueError("Division by zero is undefined.")
    return 1.0 / x

def inv_back(x: float, y: float) -> float:
    """Compute the derivative of the inverse function."""
    if x == 0:
        raise ValueError("Division by zero is undefined.")
    return -y / (x * x)

def relu_back(x: float, y: float) -> float:
    """Compute the derivative of the ReLU function."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(func: Callable[[float], float], iterable: Iterable[float]) -> list[float]:
    """Apply a function to each element in an iterable."""
    return [func(x) for x in iterable]

def zipWith(func: Callable[[float, float], float], iterable1: Iterable[float], iterable2: Iterable[float]) -> list[float]:
    """Apply a function to pairs of elements from two iterables."""
    return [func(x, y) for x, y in zip(iterable1, iterable2)]

def reduce(func: Callable[[float, float], float], iterable: Iterable[float]) -> float:
    """Reduce an iterable to a single value using a binary function."""
    if not iterable:
        return 0
    it = iter(iterable)
    result = next(it)
    for x in it:
        result = func(result, x)
    return result

def negList(ls: list[float]) -> list[float]:
    """Negate each element in a list."""
    return map(neg, ls)

def addLists(ls1: list[float], ls2: list[float]) -> list[float]:
    """Add two lists element-wise."""
    return zipWith(add, ls1, ls2)

def sum(ls: list[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, ls)

def prod(ls: list[float]) -> float:
    """Compute the product of all elements in a list."""
    return reduce(mul, ls)
