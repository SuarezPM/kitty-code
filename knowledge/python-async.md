# Python Async/Await — Quick Reference

## Basic async def and await

```python
import asyncio

async def fetch(url):
    await asyncio.sleep(0.1)  # simulate I/O
    return f"data from {url}"

async def main():
    result = await fetch("https://example.com")
    print(result)

asyncio.run(main())
```

## Concurrent execution

```python
# asyncio.gather — run coroutines concurrently
async def main():
    results = await asyncio.gather(
        fetch("https://a.com"),
        fetch("https://b.com"),
        fetch("https://c.com"),
    )
    print(results)

# asyncio.create_task for background work
async def main():
    task = asyncio.create_task(fetch("https://example.com"))
    # do other work
    await task
```

## Async context and iteration

```python
# async with and async for
async def process(stream):
    async with stream as s:
        async for item in s:
            await handle(item)

# Timeout and error handling
async def safe_fetch(url):
    try:
        return await asyncio.wait_for(fetch(url), timeout=2.0)
    except asyncio.TimeoutError:
        return None
```
