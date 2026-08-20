# Python List Comprehension — Quick Reference

## Basic syntax

```python
# list comprehension — concise list creation
[x for x in iterable if cond]
```

## Examples

```python
# Simple mapping
squares = [x*x for x in range(10)]

# Filtering with if cond
evens = [x for x in range(20) if x % 2 == 0]

# Combined mapping + filtering — list comprehension power
names = ["alice", "bob", "charlie"]
upper_long = [n.upper() for n in names if len(n) > 3]

# Equivalent for-loop for reference
result = []
for x in iterable:
    if cond:
        result.append(x)
```

## Variations

```python
# Conditional expression inside list comprehension
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]

# Nested loops
pairs = [(x, y) for x in range(3) for y in range(2)]

# Dict and set comprehensions (related)
squares_dict = {x: x*x for x in range(5)}
unique_lengths = {len(w) for w in ["hi", "hello", "hi"]}
```

## Common patterns

```python
# Flatten a matrix with list comprehension
matrix = [[1, 2], [3, 4]]
flat = [x for row in matrix for x in row]

# Strip and filter strings
clean = [s.strip() for s in raw if s.strip()]
```
