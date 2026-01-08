# Threading and Parallelism

NumPy QuadDType is designed to be thread-safe and supports Python's free-threading (GIL-free) mode introduced in Python 3.13.

## Thread Safety

All QuadDType operations are thread-safe:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType
from concurrent.futures import ThreadPoolExecutor

def compute_sum(arr):
    """Thread-safe computation."""
    return np.sum(arr)

# Create shared array
arr = np.arange(1000, dtype=QuadPrecDType())

# Run computations in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    # Split array and compute sums in parallel
    chunks = np.array_split(arr, 4)
    futures = [executor.submit(compute_sum, chunk) for chunk in chunks]
    results = [f.result() for f in futures]
    
total = sum(results)
print(f"Total sum: {total}")
```

## QuadBLAS Threading Control

NumPy QuadDType uses QuadBLAS for optimized linear algebra operations. You can control the number of threads used:

```python
from numpy_quaddtype import set_num_threads, get_num_threads, get_quadblas_version

# Check QuadBLAS version
version = get_quadblas_version()
if version:
    print(f"QuadBLAS version: {version}")
else:
    print("QuadBLAS not available (DISABLE_QUADBLAS was set)")

# Get current thread count
current_threads = get_num_threads()
print(f"Current threads: {current_threads}")

# Set thread count
set_num_threads(4)
print(f"Threads after setting: {get_num_threads()}")

# Use single thread for reproducibility
set_num_threads(1)
```

```{note}
QuadBLAS is disabled on Windows builds due to MSVC compatibility issues.
Use `get_quadblas_version()` to check if it's available.
```

## Free-Threading Support (Python 3.13+)

NumPy QuadDType fully supports Python's experimental free-threading mode (GIL-free Python).

### Checking Free-Threading Mode

```python
import sys

if hasattr(sys, '_is_gil_enabled'):
    if sys._is_gil_enabled():
        print("Running with GIL enabled")
    else:
        print("Running in free-threaded mode (no GIL)")
else:
    print("Free-threading not available (Python < 3.13)")
```

### Using Free-Threading

When running with free-threading enabled, true parallel execution is possible:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType
import threading

results = []
lock = threading.Lock()

def parallel_compute(arr):
    """Compute in parallel."""
    result = np.sum(np.sin(arr))
    with lock:
        results.append(result)

# Create arrays for parallel processing
arrays = [np.arange(i * 1000, (i + 1) * 1000, dtype=QuadPrecDType()) 
          for i in range(4)]

# Run in parallel threads
threads = [threading.Thread(target=parallel_compute, args=(arr,)) 
           for arr in arrays]

for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Results from {len(results)} threads: {results}")
```

## Building with Thread Sanitizer

For development and testing thread safety, you can build with Thread Sanitizer (TSan):

### Prerequisites

```bash
# Use clang compiler
export CC=clang
export CXX=clang++
```

### Build Steps

1. Build CPython with TSan support (see [Python Free-Threading Guide](https://py-free-threading.github.io/thread_sanitizer/))

2. Build NumPy with TSan:
   ```bash
   pip install "numpy @ git+https://github.com/numpy/numpy" \
       -C'setup-args=-Db_sanitize=thread'
   ```

3. Build SLEEF with TSan:
   ```bash
   cmake \
       -DCMAKE_C_COMPILER=clang \
       -DCMAKE_C_FLAGS="-fsanitize=thread -g -O1" \
       -DSLEEF_BUILD_QUAD=ON \
       -S sleef -B sleef/build
   
   cmake --build sleef/build -j
   sudo cmake --install sleef/build --prefix=/usr/local
   ```

4. Build numpy-quaddtype with TSan:
   ```bash
   export CFLAGS="-fsanitize=thread -g -O0"
   export CXXFLAGS="-fsanitize=thread -g -O0"
   export LDFLAGS="-fsanitize=thread"
   
   pip install . -vv --no-build-isolation \
       -Csetup-args=-Db_sanitize=thread
   ```

## Best Practices

### Do's

- ✅ Use `set_num_threads()` to control parallelism
- ✅ Use thread-local storage for intermediate results when needed
- ✅ Test with TSan during development
- ✅ Use proper synchronization for shared mutable state

### Don'ts

- ❌ Don't assume operations are atomic
- ❌ Don't modify arrays while other threads are reading them
- ❌ Don't ignore thread sanitizer warnings

## Performance Considerations

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType, set_num_threads
import time

arr = np.random.randn(100000).astype(QuadPrecDType())

# Benchmark with different thread counts
for threads in [1, 2, 4, 8]:
    set_num_threads(threads)
    
    start = time.time()
    for _ in range(10):
        result = np.sum(arr)
    elapsed = time.time() - start
    
    print(f"Threads: {threads}, Time: {elapsed:.3f}s")
```

The optimal thread count depends on your specific workload and hardware.
