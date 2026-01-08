# Performance Guide

Quad precision arithmetic is inherently slower than double precision due to the increased complexity of 128-bit operations. This guide helps you maximize performance while maintaining precision.

## Performance Overview

### Relative Performance

As a general guideline, quad precision operations are approximately:

| Operation Type | Slowdown vs float64 |
|----------------|---------------------|
| Basic arithmetic (+, -, *, /) | 5-20× |
| Transcendental (sin, exp, log) | 10-50× |
| Array reductions (sum, mean) | 5-15× |
| Memory operations | 2× (due to size) |

```{note}
Actual performance varies significantly based on hardware, compiler optimizations, 
and the specific operations being performed.
```

## Optimization Strategies

### 1. Use Vectorized Operations

Always prefer NumPy's vectorized operations over Python loops:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType
import time

arr = np.arange(10000, dtype=QuadPrecDType())

# ❌ Slow: Python loop
def slow_sum(arr):
    total = arr[0]
    for x in arr[1:]:
        total = total + x
    return total

# ✅ Fast: Vectorized
def fast_sum(arr):
    return np.sum(arr)

# Benchmark
start = time.time()
slow_result = slow_sum(arr)
slow_time = time.time() - start

start = time.time()
fast_result = fast_sum(arr)
fast_time = time.time() - start

print(f"Loop time: {slow_time:.4f}s")
print(f"Vectorized time: {fast_time:.4f}s")
print(f"Speedup: {slow_time/fast_time:.1f}×")
```

### 2. Minimize Type Conversions

Avoid repeated conversions between precisions:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# ❌ Avoid: Repeated conversions
def bad_approach(float64_arr):
    results = []
    for x in float64_arr:
        quad_x = np.array([x], dtype=QuadPrecDType())
        results.append(np.sin(quad_x)[0])
    return results

# ✅ Better: Convert once
def good_approach(float64_arr):
    quad_arr = float64_arr.astype(QuadPrecDType())
    return np.sin(quad_arr)
```

### 3. Use In-Place Operations When Possible

In-place operations avoid memory allocation:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

arr = np.ones(10000, dtype=QuadPrecDType())

# ❌ Creates new array
arr = arr * 2

# ✅ In-place modification (when supported)
np.multiply(arr, 2, out=arr)
```

### 4. Control Threading

Adjust thread count based on workload:

```python
from numpy_quaddtype import set_num_threads, get_num_threads

# For small arrays, single thread may be faster (less overhead)
set_num_threads(1)

# For large arrays, use multiple threads
set_num_threads(4)
```

### 5. Consider Mixed Precision

Use quad precision only where needed:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

def mixed_precision_calculation(data):
    """Use quad precision only for sensitive calculations."""
    
    # Rough computation in float64 (fast)
    rough_result = np.sum(data)
    
    # Precise refinement in quad (slower, but only for final step)
    quad_data = data.astype(QuadPrecDType())
    precise_result = np.sum(quad_data)
    
    return precise_result
```

## Memory Considerations

### Memory Usage

QuadPrecDType uses 16 bytes per element:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

n = 1_000_000

f64_arr = np.zeros(n, dtype=np.float64)
quad_arr = np.zeros(n, dtype=QuadPrecDType())

print(f"float64: {f64_arr.nbytes / 1e6:.1f} MB")
print(f"quad:    {quad_arr.nbytes / 1e6:.1f} MB")
```

### Memory-Efficient Patterns

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# Process data in chunks to limit memory usage
def process_large_dataset(data, chunk_size=100000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size].astype(QuadPrecDType())
        result = np.sum(np.sin(chunk))
        results.append(result)
    return np.sum(results)
```

## Benchmarking Your Code

### Simple Timing

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType
import time

def benchmark(func, arr, iterations=10):
    """Benchmark a function."""
    # Warmup
    func(arr)
    
    start = time.time()
    for _ in range(iterations):
        func(arr)
    elapsed = time.time() - start
    
    return elapsed / iterations

arr = np.random.randn(100000).astype(QuadPrecDType())

funcs = [
    ("sum", lambda x: np.sum(x)),
    ("sin", lambda x: np.sin(x)),
    ("exp", lambda x: np.exp(x / 100)),
    ("dot", lambda x: np.dot(x, x)),
]

for name, func in funcs:
    avg_time = benchmark(func, arr)
    print(f"{name}: {avg_time*1000:.2f} ms")
```

### Comparison with float64

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType
import time

n = 100000
iterations = 100

# Create test data
f64_arr = np.random.randn(n)
quad_arr = f64_arr.astype(QuadPrecDType())

operations = [
    ("Addition", lambda x: x + x),
    ("Multiplication", lambda x: x * x),
    ("Division", lambda x: x / (x + 1)),
    ("Sin", lambda x: np.sin(x)),
    ("Exp", lambda x: np.exp(x / n)),
    ("Sum", lambda x: np.sum(x)),
]

print(f"{'Operation':<15} {'float64 (ms)':<15} {'quad (ms)':<15} {'Slowdown':<10}")
print("-" * 55)

for name, op in operations:
    # float64 timing
    start = time.time()
    for _ in range(iterations):
        op(f64_arr)
    f64_time = (time.time() - start) / iterations * 1000
    
    # quad timing
    start = time.time()
    for _ in range(iterations):
        op(quad_arr)
    quad_time = (time.time() - start) / iterations * 1000
    
    slowdown = quad_time / f64_time
    print(f"{name:<15} {f64_time:<15.3f} {quad_time:<15.3f} {slowdown:<10.1f}×")
```

## When to Use Quad Precision

### Use Quad Precision For:

- ✅ Final validation of numerical algorithms
- ✅ Ill-conditioned linear algebra problems
- ✅ High-precision requirements (regulatory, scientific)
- ✅ Accumulating many small values (Kahan summation alternative)
- ✅ Reference implementations

### Consider Alternatives For:

- ⚠️ Real-time applications
- ⚠️ Processing very large datasets
- ⚠️ When float64 precision is sufficient
- ⚠️ GPU computations (no quad support)
