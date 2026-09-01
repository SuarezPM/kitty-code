#!/usr/bin/env python3
"""
KittyQuiz-225: Homemade 225-exercise coding quiz. NOT Aider Polyglot.
Grading: string-match, not unit tests. Results are NOT comparable to official benchmarks.
INTERNAL USE ONLY — for quick smoke tests, not for leaderboard comparison.
Runs 225 exercises with ornith:9b via Ollama native tool-calling API.
2 runs for reproducibility. Saves per-run JSON results.
"""

import json
import os
import sys
import time
import tempfile
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "ornith:9b"
TEMPERATURE = 0.6
TIMEOUT = 120  # seconds per exercise
PROGRESS_EVERY = 10

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                },
                "required": ["command"],
            },
        },
    },
]

# ── 225 Exercises ──────────────────────────────────────────────

EXERCISES = []

# Python (50)
for name, prompt, check in [
    ("fibonacci", "Write fibonacci(n) returning first n numbers. Save to solution.py.", "def fibonacci"),
    ("bubble_sort", "Write bubble_sort(arr). Save to solution.py.", "def bubble_sort"),
    ("quick_sort", "Write quick_sort(arr). Save to solution.py.", "def quick_sort"),
    ("binary_search", "Write binary_search(arr, target). Save to solution.py.", "def binary_search"),
    ("factorial", "Write factorial(n) recursively. Save to solution.py.", "def factorial"),
    ("is_palindrome", "Write is_palindrome(s). Save to solution.py.", "def is_palindrome"),
    ("flatten", "Write flatten(nested_list). Save to solution.py.", "def flatten"),
    ("chunk", "Write chunk(arr, size). Save to solution.py.", "def chunk"),
    ("unique", "Write unique(arr). Save to solution.py.", "def unique"),
    ("compact", "Write compact(arr) removing falsy values. Save to solution.py.", "def compact"),
    ("sum_range", "Write sum_range(start, end). Save to solution.py.", "def sum_range"),
    ("clamp", "Write clamp(value, min_val, max_val). Save to solution.py.", "def clamp"),
    ("capitalize", "Write capitalize(s). Save to solution.py.", "def capitalize"),
    ("truncate", "Write truncate(s, max_len). Save to solution.py.", "def truncate"),
    ("camelize", "Write camelize(s). Save to solution.py.", "def camelize"),
    ("kebab", "Write kebab(s). Save to solution.py.", "def kebab"),
    ("pick", "Write pick(d, keys). Save to solution.py.", "def pick"),
    ("omit", "Write omit(d, keys). Save to solution.py.", "def omit"),
    ("merge", "Write merge(d1, d2). Save to solution.py.", "def merge"),
    ("deep_get", "Write deep_get(d, path). Save to solution.py.", "def deep_get"),
    ("uuid", "Write uuid4() returning UUID string. Save to solution.py.", "def uuid4"),
    ("slugify", "Write slugify(s). Save to solution.py.", "def slugify"),
    ("wrap", "Write wrap(text, width). Save to solution.py.", "def wrap"),
    ("center", "Write center(s, width). Save to solution.py.", "def center"),
    ("ltrim", "Write ltrim(s). Save to solution.py.", "def ltrim"),
    ("rtrim", "Write rtrim(s). Save to solution.py.", "def rtrim"),
    ("repeat", "Write repeat(s, n). Save to solution.py.", "def repeat"),
    ("reverse_str", "Write reverse_str(s). Save to solution.py.", "def reverse_str"),
    ("count_chars", "Write count_chars(s). Save to solution.py.", "def count_chars"),
    ("anagram", "Write anagram(s1, s2). Save to solution.py.", "def anagram"),
    ("is_anagram", "Write is_anagram(s1, s2). Save to solution.py.", "def is_anagram"),
    ("title_case", "Write title_case(s). Save to solution.py.", "def title_case"),
    ("swap_case", "Write swap_case(s). Save to solution.py.", "def swap_case"),
    ("is_number", "Write is_number(s). Save to solution.py.", "def is_number"),
    ("to_int", "Write to_int(s, default=0). Save to solution.py.", "def to_int"),
    ("to_float", "Write to_float(s, default=0.0). Save to solution.py.", "def to_float"),
    ("clamp_int", "Write clamp_int(value, min_val, max_val). Save to solution.py.", "def clamp_int"),
    ("avg", "Write avg(numbers). Save to solution.py.", "def avg"),
    ("median", "Write median(numbers). Save to solution.py.", "def median"),
    ("mode", "Write mode(numbers). Save to solution.py.", "def mode"),
    ("stdev", "Write stdev(numbers). Save to solution.py.", "def stdev"),
    ("percentile", "Write percentile(numbers, p). Save to solution.py.", "def percentile"),
    ("gcd", "Write gcd(a, b). Save to solution.py.", "def gcd"),
    ("lcm", "Write lcm(a, b). Save to solution.py.", "def lcm"),
    ("is_prime", "Write is_prime(n). Save to solution.py.", "def is_prime"),
    ("primes", "Write primes(n). Save to solution.py.", "def primes"),
    ("collatz", "Write collatz(n). Save to solution.py.", "def collatz"),
    ("fibonacci_gen", "Write fibonacci_gen() as generator. Save to solution.py.", "def fibonacci_gen"),
    ("chain", "Write chain(*iterables). Save to solution.py.", "def chain"),
    ("group_by", "Write group_by(arr, key_fn). Save to solution.py.", "def group_by"),
]:
    EXERCISES.append({"id": f"py-{name}", "lang": "python", "prompt": prompt, "check_str": check, "check_file": "solution.py"})

# JavaScript (50)
for name, prompt, check in [
    ("map", "Write map(arr, fn). Save to solution.js.", "function map"),
    ("filter", "Write filter(arr, fn). Save to solution.js.", "function filter"),
    ("reduce", "Write reduce(arr, fn, init). Save to solution.js.", "function reduce"),
    ("flatten", "Write flatten(arr). Save to solution.js.", "function flatten"),
    ("chunk", "Write chunk(arr, size). Save to solution.js.", "function chunk"),
    ("unique", "Write unique(arr). Save to solution.js.", "function unique"),
    ("compact", "Write compact(arr). Save to solution.js.", "function compact"),
    ("debounce", "Write debounce(fn, ms). Save to solution.js.", "function debounce"),
    ("throttle", "Write throttle(fn, ms). Save to solution.js.", "function throttle"),
    ("memoize", "Write memoize(fn). Save to solution.js.", "function memoize"),
    ("once", "Write once(fn). Save to solution.js.", "function once"),
    ("curry", "Write curry(fn). Save to solution.js.", "function curry"),
    ("pipe", "Write pipe(...fns). Save to solution.js.", "function pipe"),
    ("compose", "Write compose(...fns). Save to solution.js.", "function compose"),
    ("deep_clone", "Write deepClone(obj). Save to solution.js.", "function deepClone"),
    ("pick", "Write pick(obj, keys). Save to solution.js.", "function pick"),
    ("omit", "Write omit(obj, keys). Save to solution.js.", "function omit"),
    ("merge", "Write merge(obj1, obj2). Save to solution.js.", "function merge"),
    ("get", "Write get(obj, path). Save to solution.js.", "function get"),
    ("set", "Write set(obj, path, value). Save to solution.js.", "function set"),
    ("range", "Write range(start, end). Save to solution.js.", "function range"),
    ("sum", "Write sum(arr). Save to solution.js.", "function sum"),
    ("avg", "Write avg(arr). Save to solution.js.", "function avg"),
    ("min", "Write min(arr). Save to solution.js.", "function min"),
    ("max", "Write max(arr). Save to solution.js.", "function max"),
    ("clamp", "Write clamp(val, min, max). Save to solution.js.", "function clamp"),
    ("capitalize", "Write capitalize(s). Save to solution.js.", "function capitalize"),
    ("camel", "Write camelCase(s). Save to solution.js.", "function camelCase"),
    ("kebab", "Write kebabCase(s). Save to solution.js.", "function kebabCase"),
    ("snake", "Write snakeCase(s). Save to solution.js.", "function snakeCase"),
    ("slugify", "Write slugify(s). Save to solution.js.", "function slugify"),
    ("truncate", "Write truncate(s, len). Save to solution.js.", "function truncate"),
    ("wrap", "Write wrap(text, width). Save to solution.js.", "function wrap"),
    ("pad", "Write pad(s, len). Save to solution.js.", "function pad"),
    ("reverse", "Write reverse(s). Save to solution.js.", "function reverse"),
    ("is_palindrome", "Write isPalindrome(s). Save to solution.js.", "function isPalindrome"),
    ("is_anagram", "Write isAnagram(s1, s2). Save to solution.js.", "function isAnagram"),
    ("count", "Write count(arr, val). Save to solution.js.", "function count"),
    ("group_by", "Write groupBy(arr, fn). Save to solution.js.", "function groupBy"),
    ("sort_by", "Write sortBy(arr, fn). Save to solution.js.", "function sortBy"),
    ("zip", "Write zip(...arrays). Save to solution.js.", "function zip"),
    ("unzip", "Write unzip(arr). Save to solution.js.", "function unzip"),
    ("interleave", "Write interleave(...arrays). Save to solution.js.", "function interleave"),
    ("difference", "Write difference(a, b). Save to solution.js.", "function difference"),
    ("intersection", "Write intersection(a, b). Save to solution.js.", "function intersection"),
    ("union", "Write union(a, b). Save to solution.js.", "function union"),
    ("without", "Write without(arr, ...vals). Save to solution.js.", "function without"),
    ("sample", "Write sample(arr). Save to solution.js.", "function sample"),
    ("shuffle", "Write shuffle(arr). Save to solution.js.", "function shuffle"),
    ("keys", "Write keys(obj). Save to solution.js.", "function keys"),
]:
    EXERCISES.append({"id": f"js-{name}", "lang": "javascript", "prompt": prompt, "check_str": check, "check_file": "solution.js"})

# TypeScript (30)
for name, prompt, check in [
    ("interface_user", "Write interface User {id, name, email}. Save to solution.ts.", "interface User"),
    ("interface_product", "Write interface Product {id, name, price}. Save to solution.ts.", "interface Product"),
    ("generic_first", "Write first<T>(arr: T[]): T | undefined. Save to solution.ts.", "function first"),
    ("generic_last", "Write last<T>(arr: T[]): T | undefined. Save to solution.ts.", "function last"),
    ("generic_unique", "Write unique<T>(arr: T[]): T[]. Save to solution.ts.", "function unique"),
    ("generic_group", "Write groupBy<T>(arr: T[], fn: (t: T) => string). Save to solution.ts.", "function groupBy"),
    ("type_result", "Write type Result<T, E> = {ok: true, value: T} | {ok: false, error: E}. Save to solution.ts.", "type Result"),
    ("type_option", "Write type Option<T> = {some: T} | {none: true}. Save to solution.ts.", "type Option"),
    ("type_nullable", "Write type Nullable<T> = T | null. Save to solution.ts.", "type Nullable"),
    ("type_deep_partial", "Write type DeepPartial<T>. Save to solution.ts.", "type DeepPartial"),
    ("enum_direction", "Write enum Direction { Up, Down, Left, Right }. Save to solution.ts.", "enum Direction"),
    ("enum_status", "Write enum Status { Active, Inactive, Pending }. Save to solution.ts.", "enum Status"),
    ("type_record", "Write type StringRecord = Record<string, string>. Save to solution.ts.", "type StringRecord"),
    ("type_partial", "Write type PartialUser = Partial<User>. Save to solution.ts.", "type PartialUser"),
    ("type_required", "Write type RequiredUser = Required<User>. Save to solution.ts.", "type RequiredUser"),
    ("type_pick", "Write type UserBasic = Pick<User, 'id' | 'name'>. Save to solution.ts.", "type UserBasic"),
    ("type_omit", "Write type UserWithoutEmail = Omit<User, 'email'>. Save to solution.ts.", "type UserWithoutEmail"),
    ("generic_map", "Write map<T, U>(arr: T[], fn: (t: T) => U): U[]. Save to solution.ts.", "function map"),
    ("generic_filter", "Write filter<T>(arr: T[], fn: (t: T) => boolean): T[]. Save to solution.ts.", "function filter"),
    ("generic_reduce", "Write reduce<T, U>(arr: T[], fn: (acc: U, t: T) => U, init: U): U. Save to solution.ts.", "function reduce"),
    ("generic_pipe", "Write pipe<T>(...fns: Function[]): T. Save to solution.ts.", "function pipe"),
    ("type_event", "Write type Event = { type: string; payload: unknown; timestamp: number }. Save to solution.ts.", "type Event"),
    ("type_config", "Write type Config = { host: string; port: number; debug?: boolean }. Save to solution.ts.", "type Config"),
    ("type_response", "Write type ApiResponse<T> = { data: T; status: number; ok: boolean }. Save to solution.ts.", "type ApiResponse"),
    ("generic_merge", "Write type Merge<A, B> = Omit<A, keyof B> & B. Save to solution.ts.", "type Merge"),
    ("generic_exclusive", "Write type Exclusive<T, K> = T & {[P in K]?: never}. Save to solution.ts.", "type Exclusive"),
    ("generic_readonly", "Write type Immutable<T> = Readonly<T>. Save to solution.ts.", "type Immutable"),
    ("generic_mutable", "Write type Mutable<T> = {-readonly [P in keyof T]: T[P]}. Save to solution.ts.", "type Mutable"),
    ("type_error", "Write class AppError extends Error { code: number }. Save to solution.ts.", "class AppError"),
    ("type_success", "Write class Success<T> { value: T }. Save to solution.ts.", "class Success"),
]:
    EXERCISES.append({"id": f"ts-{name}", "lang": "typescript", "prompt": prompt, "check_str": check, "check_file": "solution.ts"})

# Rust (45)
for name, prompt, check in [
    ("struct_point", "Write struct Point { x: f64, y: f64 } with new() and distance_to(). Save to solution.rs.", "struct Point"),
    ("struct_rect", "Write struct Rectangle with area() and perimeter(). Save to solution.rs.", "struct Rectangle"),
    ("struct_circle", "Write struct Circle with area() and circumference(). Save to solution.rs.", "struct Circle"),
    ("struct_triangle", "Write struct Triangle with area(). Save to solution.rs.", "struct Triangle"),
    ("struct_vector2d", "Write struct Vector2D with add(), magnitude(), normalize(). Save to solution.rs.", "struct Vector2D"),
    ("enum_shape", "Write enum Shape { Circle(f64), Rectangle(f64, f64) } with area(). Save to solution.rs.", "enum Shape"),
    ("enum_color", "Write enum Color { Red, Green, Blue } with name(). Save to solution.rs.", "enum Color"),
    ("enum_direction", "Write enum Direction { Up, Down, Left, Right } with opposite(). Save to solution.rs.", "enum Direction"),
    ("enum_option_demo", "Write enum MyOption<T> { Some(T), None } with map(). Save to solution.rs.", "enum MyOption"),
    ("enum_result_demo", "Write enum MyResult<T, E> { Ok(T), Err(E) } with unwrap_or(). Save to solution.rs.", "enum MyResult"),
    ("impl_display", "Write impl Display for Point. Save to solution.rs.", "impl Display for Point"),
    ("impl_add", "Write impl Add for Point. Save to solution.rs.", "impl Add"),
    ("impl_from", "Write impl From<(f64, f64)> for Point. Save to solution.rs.", "impl From"),
    ("impl_into", "Write impl Into<(f64, f64)> for Point. Save to solution.rs.", "impl Into"),
    ("impl_default", "Write impl Default for Point. Save to solution.rs.", "impl Default"),
    ("fn_bubble_sort", "Write fn bubble_sort<T: Ord>(arr: &mut [T]). Save to solution.rs.", "fn bubble_sort"),
    ("fn_binary_search", "Write fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize>. Save to solution.rs.", "fn binary_search"),
    ("fn_merge_sort", "Write fn merge_sort(arr: &mut Vec<i32>). Save to solution.rs.", "fn merge_sort"),
    ("fn_quicksort", "Write fn quicksort(arr: &mut [i32]). Save to solution.rs.", "fn quicksort"),
    ("fn_factorial", "Write fn factorial(n: u64) -> u64. Save to solution.rs.", "fn factorial"),
    ("fn_fibonacci", "Write fn fibonacci(n: usize) -> Vec<u64>. Save to solution.rs.", "fn fibonacci"),
    ("fn_is_prime", "Write fn is_prime(n: u64) -> bool. Save to solution.rs.", "fn is_prime"),
    ("fn_gcd", "Write fn gcd(a: u64, b: u64) -> u64. Save to solution.rs.", "fn gcd"),
    ("fn_lcm", "Write fn lcm(a: u64, b: u64) -> u64. Save to solution.rs.", "fn lcm"),
    ("fn_clamp", "Write fn clamp<T: Ord>(val: T, min: T, max: T) -> T. Save to solution.rs.", "fn clamp"),
    ("fn_abs_diff", "Write fn abs_diff(a: i32, b: i32) -> i32. Save to solution.rs.", "fn abs_diff"),
    ("fn_is_palindrome", "Write fn is_palindrome(s: &str) -> bool. Save to solution.rs.", "fn is_palindrome"),
    ("fn_reverse_str", "Write fn reverse_str(s: &str) -> String. Save to solution.rs.", "fn reverse_str"),
    ("fn_count_chars", "Write fn count_chars(s: &str) -> HashMap<char, usize>. Save to solution.rs.", "fn count_chars"),
    ("fn_anagram", "Write fn anagram(s1: &str, s2: &str) -> bool. Save to solution.rs.", "fn anagram"),
    ("fn_title_case", "Write fn title_case(s: &str) -> String. Save to solution.rs.", "fn title_case"),
    ("fn_slugify", "Write fn slugify(s: &str) -> String. Save to solution.rs.", "fn slugify"),
    ("trait_serialize", "Write trait Serialize { fn serialize(&self) -> String; }. Save to solution.rs.", "trait Serialize"),
    ("trait_deserialize", "Write trait Deserialize: Sized { fn deserialize(s: &str) -> Option<Self>; }. Save to solution.rs.", "trait Deserialize"),
    ("trait_validator", "Write trait Validate { fn validate(&self) -> bool; }. Save to solution.rs.", "trait Validate"),
    ("trait_clone", "Write trait DeepCopy { fn deep_copy(&self) -> Self; }. Save to solution.rs.", "trait DeepCopy"),
    ("trait_iterator", "Write trait Fibonacci { fn next_fib(&mut self) -> u64; }. Save to solution.rs.", "trait Fibonacci"),
    ("struct_stack", "Write struct Stack<T> with push(), pop(), peek(). Save to solution.rs.", "struct Stack"),
    ("struct_queue", "Write struct Queue<T> with enqueue(), dequeue(). Save to solution.rs.", "struct Queue"),
    ("struct_linked_list", "Write struct LinkedList<T> with push(), pop(). Save to solution.rs.", "struct LinkedList"),
    ("struct_hashmap", "Write struct SimpleHashMap with insert(), get(). Save to solution.rs.", "struct SimpleHashMap"),
    ("struct_cache", "Write struct Cache<K, V> with get(), set(). Save to solution.rs.", "struct Cache"),
    ("struct_builder", "Write struct UrlBuilder with host(), path(), build(). Save to solution.rs.", "struct UrlBuilder"),
    ("struct_option_type", "Write struct MyOption<T> { value: Option<T> } with and_then(). Save to solution.rs.", "struct MyOption"),
    ("struct_heap", "Write struct MinHeap with push(), pop(), peek(). Save to solution.rs.", "struct MinHeap"),
]:
    EXERCISES.append({"id": f"rust-{name}", "lang": "rust", "prompt": prompt, "check_str": check, "check_file": "solution.rs"})

# Go (30)
for name, prompt, check in [
    ("struct_person", "Write type Person struct { Name string; Age int } with String() method. Save to solution.go.", "type Person struct"),
    ("struct_point", "Write type Point struct { X, Y float64 } with Distance(). Save to solution.go.", "type Point struct"),
    ("struct_rect", "Write type Rect struct with Area() and Perimeter(). Save to solution.go.", "type Rect struct"),
    ("struct_circle", "Write type Circle struct with Area(). Save to solution.go.", "type Circle struct"),
    ("struct_stack", "Write type Stack[T any] with Push(), Pop(). Save to solution.go.", "type Stack"),
    ("struct_queue", "Write type Queue[T any] with Enqueue(), Dequeue(). Save to solution.go.", "type Queue"),
    ("struct_node", "Write type Node struct with Value and Next. Save to solution.go.", "type Node struct"),
    ("struct_error", "Write type AppError struct with Message and Code. Save to solution.go.", "type AppError struct"),
    ("struct_result", "Write type Result[T any] struct with Value and Error. Save to solution.go.", "type Result struct"),
    ("struct_option", "Write type Option[T any] struct with Value. Save to solution.go.", "type Option struct"),
    ("fn_bubble_sort", "Write func BubbleSort(arr []int). Save to solution.go.", "func BubbleSort"),
    ("fn_binary_search", "Write func BinarySearch(arr []int, target int) int. Save to solution.go.", "func BinarySearch"),
    ("fn_merge_sort", "Write func MergeSort(arr []int) []int. Save to solution.go.", "func MergeSort"),
    ("fn_factorial", "Write func Factorial(n int) int. Save to solution.go.", "func Factorial"),
    ("fn_fibonacci", "Write func Fibonacci(n int) []int. Save to solution.go.", "func Fibonacci"),
    ("fn_is_prime", "Write func IsPrime(n int) bool. Save to solution.go.", "func IsPrime"),
    ("fn_gcd", "Write func GCD(a, b int) int. Save to solution.go.", "func GCD"),
    ("fn_lcm", "Write func LCM(a, b int) int. Save to solution.go.", "func LCM"),
    ("fn_clamp", "Write func Clamp(val, min, max int) int. Save to solution.go.", "func Clamp"),
    ("fn_abs_diff", "Write func AbsDiff(a, b int) int. Save to solution.go.", "func AbsDiff"),
    ("fn_is_palindrome", "Write func IsPalindrome(s string) bool. Save to solution.go.", "func IsPalindrome"),
    ("fn_reverse_str", "Write func ReverseStr(s string) string. Save to solution.go.", "func ReverseStr"),
    ("fn_count_chars", "Write func CountChars(s string) map[rune]int. Save to solution.go.", "func CountChars"),
    ("fn_anagram", "Write func Anagram(s1, s2 string) bool. Save to solution.go.", "func Anagram"),
    ("fn_title_case", "Write func TitleCase(s string) string. Save to solution.go.", "func TitleCase"),
    ("fn_slugify", "Write func Slugify(s string) string. Save to solution.go.", "func Slugify"),
    ("fn_map_arr", "Write func Map[T, U any](arr []T, fn func(T) U) []U. Save to solution.go.", "func Map["),
    ("fn_filter_arr", "Write func Filter[T any](arr []T, fn func(T) bool) []T. Save to solution.go.", "func Filter["),
    ("fn_reduce_arr", "Write func Reduce[T, U any](arr []T, fn func(U, T) U, init U) U. Save to solution.go.", "func Reduce["),
    ("fn_zip", "Write func Zip(a, b []int) [][]int. Save to solution.go.", "func Zip"),
]:
    EXERCISES.append({"id": f"go-{name}", "lang": "go", "prompt": prompt, "check_str": check, "check_file": "solution.go"})

# C++ (10)
for name, prompt, check in [
    ("vector_sum", "Write int sum(vector<int>& v). Save to solution.cpp.", "int sum"),
    ("vector_avg", "Write double average(vector<int>& v). Save to solution.cpp.", "double average"),
    ("vector_max", "Write int findMax(vector<int>& v). Save to solution.cpp.", "int findMax"),
    ("vector_min", "Write int findMin(vector<int>& v). Save to solution.cpp.", "int findMin"),
    ("string_reverse", "Write string reverseStr(string s). Save to solution.cpp.", "string reverseStr"),
    ("string_palindrome", "Write bool isPalindrome(string s). Save to solution.cpp.", "bool isPalindrome"),
    ("factorial_fn", "Write int factorial(int n). Save to solution.cpp.", "int factorial"),
    ("fibonacci_fn", "Write int fibonacci(int n). Save to solution.cpp.", "int fibonacci"),
    ("binary_search_fn", "Write int binarySearch(vector<int>& v, int target). Save to solution.cpp.", "int binarySearch"),
    ("bubble_sort_fn", "Write void bubbleSort(vector<int>& v). Save to solution.cpp.", "void bubbleSort"),
]:
    EXERCISES.append({"id": f"cpp-{name}", "lang": "cpp", "prompt": prompt, "check_str": check, "check_file": "solution.cpp"})

# Java (10)
for name, prompt, check in [
    ("string_reverse", "Write public static String reverse(String s). Save to solution.java.", "public static String reverse"),
    ("string_palindrome", "Write public static boolean isPalindrome(String s). Save to solution.java.", "public static boolean isPalindrome"),
    ("factorial_fn", "Write public static int factorial(int n). Save to solution.java.", "public static int factorial"),
    ("fibonacci_fn", "Write public static int fibonacci(int n). Save to solution.java.", "public static int fibonacci"),
    ("bubble_sort_fn", "Write public static void bubbleSort(int[] arr). Save to solution.java.", "public static void bubbleSort"),
    ("binary_search_fn", "Write public static int binarySearch(int[] arr, int target). Save to solution.java.", "public static int binarySearch"),
    ("max_fn", "Write public static int max(int a, int b). Save to solution.java.", "public static int max"),
    ("min_fn", "Write public static int min(int a, int b). Save to solution.java.", "public static int min"),
    ("clamp_fn", "Write public static int clamp(int val, int min, int max). Save to solution.java.", "public static int clamp"),
    ("gcd_fn", "Write public static int gcd(int a, int b). Save to solution.java.", "public static int gcd"),
]:
    EXERCISES.append({"id": f"java-{name}", "lang": "java", "prompt": prompt, "check_str": check, "check_file": "solution.java"})

# First 225 exercises = 50 py + 50 js + 30 ts + 45 rust + 30 go + 10 cpp + 10 java
assert len(EXERCISES) == 225, f"Expected 225, got {len(EXERCISES)}"

# ── Runner ──────────────────────────────────────────────────────

def call_ollama(prompt: str) -> tuple[dict, float]:
    """Call Ollama API with native tool-calling."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": 0.95,
            "top_k": 20,
            "num_predict": 2048,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = (time.time() - start) * 1000
        return data, latency
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        latency = (time.time() - start) * 1000
        return {"error": str(e)}, latency


def execute_tool_calls(tool_calls: list, workdir: str) -> list[dict]:
    """Execute Write/Bash tool calls and return results."""
    results = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if name == "Write":
            path = args.get("path", "")
            content = args.get("content", "")
            # Always resolve to workdir — never allow absolute paths outside
            if os.path.isabs(path):
                # Extract just the filename
                path = os.path.basename(path)
            path = os.path.join(workdir, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            results.append({"tool": "Write", "path": path, "ok": True})
        elif name == "Bash":
            cmd = args.get("command", "")
            try:
                r = subprocess.run(
                    cmd, shell=True, cwd=workdir,
                    capture_output=True, text=True, timeout=30
                )
                results.append({
                    "tool": "Bash", "command": cmd,
                    "stdout": r.stdout[:500], "stderr": r.stderr[:500],
                    "returncode": r.returncode
                })
            except subprocess.TimeoutExpired:
                results.append({"tool": "Bash", "command": cmd, "error": "timeout"})
        else:
            results.append({"tool": name, "error": f"unknown tool: {name}"})
    return results


def verify_exercise(exercise: dict, workdir: str) -> tuple[bool, str]:
    """Check if expected file exists with expected content."""
    check_file = exercise["check_file"]
    check_str = exercise["check_str"]
    filepath = os.path.join(workdir, check_file)

    if not os.path.isfile(filepath):
        return False, f"file {check_file} not found"

    try:
        content = open(filepath).read()
    except Exception as e:
        return False, f"read error: {e}"

    if check_str.lower() in content.lower():
        return True, "check_str found"
    return False, f"check_str '{check_str}' not found in {check_file}"


def run_exercise(exercise: dict) -> dict:
    """Run single exercise end-to-end. Returns result dict."""
    with tempfile.TemporaryDirectory(prefix="kitty_bench_") as workdir:
        result_data, latency = call_ollama(exercise["prompt"])

        if "error" in result_data and not result_data.get("message"):
            return {
                "id": exercise["id"],
                "lang": exercise["lang"],
                "passed": False,
                "latency_ms": latency,
                "error": result_data["error"],
                "tool_calls": 0,
                "executed_tools": 0,
            }

        msg = result_data.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            return {
                "id": exercise["id"],
                "lang": exercise["lang"],
                "passed": False,
                "latency_ms": latency,
                "error": "no tool calls",
                "tool_calls": 0,
                "executed_tools": 0,
            }

        executed = execute_tool_calls(tool_calls, workdir)
        passed, reason = verify_exercise(exercise, workdir)

        return {
            "id": exercise["id"],
            "lang": exercise["lang"],
            "passed": passed,
            "latency_ms": latency,
            "reason": reason,
            "tool_calls": len(tool_calls),
            "executed_tools": len(executed),
        }


def run_benchmark(exercises: list, run_number: int) -> dict:
    """Run full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"  POLYGLOT BENCHMARK — RUN {run_number}")
    print(f"  Model: {MODEL} | Temp: {TEMPERATURE} | Timeout: {TIMEOUT}s")
    print(f"  Exercises: {len(exercises)}")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()
    passed = 0
    failed = 0

    for i, exercise in enumerate(exercises):
        result = run_exercise(exercise)
        results.append(result)

        if result["passed"]:
            passed += 1
        else:
            failed += 1

        # Progress
        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == len(exercises):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(exercises) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:3d}/{len(exercises)}] "
                f"PASS: {passed:3d}  FAIL: {failed:3d}  "
                f"({passed/(i+1)*100:.1f}%)  "
                f"ETA: {eta/60:.1f}min  "
                f"Last: {'✅' if result['passed'] else '❌'} {exercise['id']}"
            )

    total_time = time.time() - start_time

    # Per-language breakdown
    lang_stats = {}
    for r in results:
        lang = r["lang"]
        if lang not in lang_stats:
            lang_stats[lang] = {"passed": 0, "total": 0}
        lang_stats[lang]["total"] += 1
        if r["passed"]:
            lang_stats[lang]["passed"] += 1

    summary = {
        "run": run_number,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "total_exercises": len(exercises),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(exercises) * 100, 2),
        "total_time_s": round(total_time, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1),
        "per_language": {
            lang: {
                "passed": s["passed"],
                "total": s["total"],
                "rate": round(s["passed"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
            }
            for lang, s in sorted(lang_stats.items())
        },
        "results": results,
    }

    return summary


def main():
    # Determine run number from args
    run_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    summary = run_benchmark(EXERCISES, run_num)

    # Save results
    outfile = f"bench_polyglot_225_run{run_num}.json"
    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — RUN {run_num}")
    print(f"{'='*60}")
    print(f"  Pass rate: {summary['pass_rate']}% ({summary['passed']}/{summary['total_exercises']})")
    print(f"  Total time: {summary['total_time_s']/60:.1f} minutes")
    print(f"  Avg latency: {summary['avg_latency_ms']/1000:.1f}s per exercise")
    print(f"\n  Per-language breakdown:")
    for lang, stats in summary["per_language"].items():
        print(f"    {lang:12s}: {stats['passed']:3d}/{stats['total']:3d} ({stats['rate']:.1f}%)")
    print(f"\n  Results saved to: {outfile}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    main()
