# Rust Traits — Quick Reference

## Basic trait implementation

```rust
trait Summary {
    fn summarize(&self) -> String;
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.title, self.author, self.date)
    }
}
```

## Common trait implementations

### impl Display (user-facing formatting)

```rust
use std::fmt;

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
```

### impl From (type conversion)

```rust
impl From<i32> for MyNumber {
    fn from(val: i32) -> Self {
        MyNumber(val)
    }
}

// Usage: let x: MyNumber = 42.into();
```

### impl Default (default values)

```rust
impl Default for Config {
    fn default() -> Self {
        Config {
            width: 800,
            height: 600,
            fullscreen: false,
        }
    }
}
```

### impl Clone and Copy (for small, stack-only types)

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}
```

### impl Iterator (lazy sequences)

```rust
struct Counter { count: u32, max: u32 }

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}
```

## Trait bounds in functions

```rust
// Basic bound
fn largest<T: PartialOrd>(list: &[T]) -> &T { ... }

// Multiple bounds
fn process<T: Display + Clone>(item: &T) { ... }

// Where clause (cleaner for complex bounds)
fn process<T>(item: &T) -> String
where
    T: Display + Clone + Debug,
{
    format!("{:?}", item)
}
```

## Trait objects (dynamic dispatch)

```rust
// Box<dyn Trait> for heap-allocated trait objects
fn make_animal() -> Box<dyn Speak> {
    Box::new(Dog { name: "Rex".into() })
}

// &dyn Trait for borrowed trait objects
fn print_all(animals: &[&dyn Speak]) {
    for a in animals {
        println!("{}", a.speak());
    }
}
```

## Supertraits (trait inheritance)

```rust
trait Animal: fmt::Display {
    fn name(&self) -> &str;
}

impl fmt::Display for Dog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dog({})", self.name)
    }
}

impl Animal for Dog {
    fn name(&self) -> &str { &self.name }
}
```

## Blanket implementations

```rust
// Implement a trait for all types that implement another trait
impl<T: fmt::Display> ToString for T {
    fn to_string(&self) -> String {
        // provided by std, but this shows the pattern
        ...
    }
}
```

## derive macros (auto-implement common traits)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
struct Config {
    name: String,
    value: i32,
}
```
