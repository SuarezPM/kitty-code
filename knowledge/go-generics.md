# Go Generics — Quick Reference

## Generic functions

```go
// Basic generic function
func Map[T any, U any](s []T, f func(T) U) []U {
    result := make([]U, len(s))
    for i, v := range s {
        result[i] = f(v)
    }
    return result
}

// Usage
nums := []int{1, 2, 3}
strs := Map(nums, func(n int) string { return strconv.Itoa(n) })
```

## Generic types

### Result type (error handling)

```go
type Result[T any] struct {
    Value T
    Err   error
}

func Ok[T any](v T) Result[T] {
    return Result[T]{Value: v}
}

func Err[T any](err error) Result[T] {
    return Result[T]{Err: err}
}

func (r Result[T]) Unwrap() T {
    if r.Err != nil {
        panic(r.Err)
    }
    return r.Value
}

func (r Result[T]) IsOk() bool {
    return r.Err == nil
}
```

### Option type (nullable values)

```go
type Option[T any] struct {
    value *T
}

func Some[T any](v T) Option[T] {
    return Option[T]{value: &v}
}

func None[T any]() Option[T] {
    return Option[T]{}
}

func (o Option[T]) IsSome() bool {
    return o.value != nil
}

func (o Option[T]) Unwrap() T {
    if o.value == nil {
        panic("unwrap on None")
    }
    return *o.value
}

func (o Option[T]) Or(default_ T) T {
    if o.value != nil {
        return *o.value
    }
    return default_
}
```

## Type constraints

```go
// Built-in constraints
// any        - any type (equivalent to interface{})
// comparable - types that support == and !=

// Custom constraints
type Number interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
    ~float32 | ~float64
}

func Sum[T Number](nums []T) T {
    var total T
    for _, n := range nums {
        total += n
    }
    return total
}

// Ordered constraint (types that support < >)
type Ordered interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
    ~float32 | ~float64 | ~string
}

func Min[T Ordered](a, b T) T {
    if a < b {
        return a
    }
    return b
}
```

## Generic structs

```go
type Stack[T any] struct {
    items []T
}

func NewStack[T any]() *Stack[T] {
    return &Stack[T]{items: make([]T, 0)}
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}
```

## Interface constraints with methods

```go
type Stringer interface {
    String() string
}

// Type must implement Stringer AND be a number
type PrintableNumber interface {
    Number
    Stringer
}

func Print[T PrintableNumber](v T) {
    fmt.Println(v.String())
}
```

## Common patterns

### Filter

```go
func Filter[T any](s []T, f func(T) bool) []T {
    var result []T
    for _, v := range s {
        if f(v) {
            result = append(result, v)
        }
    }
    return result
}
```

### Reduce

```go
func Reduce[T any, U any](s []T, init U, f func(U, T) U) U {
    acc := init
    for _, v := range s {
        acc = f(acc, v)
    }
    return acc
}
```

### Keys of a map

```go
func Keys[K comparable, V any](m map[K]V) []K {
    keys := make([]K, 0, len(m))
    for k := range m {
        keys = append(keys, k)
    }
    return keys
}
```
