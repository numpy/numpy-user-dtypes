import numpy_quaddtype as npq
import numpy as np


def test_scalar_ops(backend):
    print(f"\nTesting scalar operations for {backend} backend:")

    # Create QuadPrecision instances
    q1 = npq.QuadPrecision(
        "3.14159265358979323846264338327950288", backend=backend)
    q2 = npq.QuadPrecision(
        "-2.71828182845904523536028747135266250", backend=backend)

    # Test unary operations
    print("\nUnary operations:")
    print(f"  Negation of q1: {-q1}")
    print(f"  Absolute value of q2: {abs(q2)}")

    # Test binary operations
    print("\nBinary operations:")
    print(f"  Addition: {q1 + q2}")
    print(f"  Subtraction: {q1 - q2}")
    print(f"  Multiplication: {q1 * q2}")
    print(f"  Division: {q1 / q2}")

    # Test comparison operations
    print("\nComparison operations:")
    print(f"  q1 == q2: {q1 == q2}")
    print(f"  q1 != q2: {q1 != q2}")
    print(f"  q1 < q2: {q1 < q2}")
    print(f"  q1 <= q2: {q1 <= q2}")
    print(f"  q1 > q2: {q1 > q2}")
    print(f"  q1 >= q2: {q1 >= q2}")

    # Test operations with Python numbers
    print("\nOperations with Python numbers:")
    print(f"  q1 + 1: {q1 + 1}")
    print(f"  q1 - 2.5: {q1 - 2.5}")
    print(f"  q1 * 3: {q1 * 3}")
    print(f"  q1 / 2: {q1 / 2}")

    # Test boolean conversion
    print("\nBoolean conversion:")
    print(f"  bool(q1): {bool(q1)}")
    print(
        f"  bool(npq.QuadPrecision('0', backend=backend)): {bool(npq.QuadPrecision('0', backend=backend))}")


def test_casting(backend):
    print(f"\nTesting {backend} backend:")

    # Create QuadPrecision instances
    q1 = npq.QuadPrecision(
        "3.14159265358979323846264338327950288", backend=backend)
    q2 = npq.QuadPrecision(
        "-2.71828182845904523536028747135266250", backend=backend)

    # Test casting from QuadPrecision to numpy dtypes
    print("Casting from QuadPrecision to numpy dtypes:")
    print(f"  float32: {np.float32(q1)}")
    print(f"  float64: {np.float64(q1)}")
    print(f"  int64: {np.int64(q1)}")
    print(f"  uint64: {np.uint64(q1)}")

    # Test casting from numpy dtypes to QuadPrecision
    print("\nCasting from numpy dtypes to QuadPrecision:")
    print(
        f"  float32: {np.float32(3.14159).astype(npq.QuadPrecDType(backend=backend))}")
    print(
        f"  float64: {np.float64(2.71828182845904).astype(npq.QuadPrecDType(backend=backend))}")
    print(
        f"  int64: {np.int64(-1234567890).astype(npq.QuadPrecDType(backend=backend))}")
    print(
        f"  uint64: {np.uint64(9876543210).astype(npq.QuadPrecDType(backend=backend))}")

    # Test array operations
    print("\nArray operations:")
    q_array = np.array([q1, q2], dtype=npq.QuadPrecDType(backend=backend))
    print(f"  QuadPrecision array: {q_array}")

    np_array = np.array([3.14, -2.71, 1.41, -1.73], dtype=np.float64)
    q_from_np = np_array.astype(npq.QuadPrecDType(backend=backend))
    print(f"  Numpy to QuadPrecision: {q_from_np}")

    back_to_np = np.array(q_from_np, dtype=np.float64)
    print(f"  QuadPrecision to Numpy: {back_to_np}")

    # Test precision maintenance
    large_int = 12345678901234567890
    q_large = np.array([large_int], dtype=np.uint64).astype(
        npq.QuadPrecDType(backend=backend))[0]
    print(f"\nPrecision test:")
    print(f"  Original large int: {large_int}")
    print(f"  QuadPrecision: {q_large}")
    print(f"  Back to int: {np.int64(q_large)}")

    # Test edge cases


def test_edge_cases(backend):
    print(f"\nTesting negative numbers for {backend} backend:")

    # Test various negative numbers
    test_values = [
        -1.0,
        -1e10,
        -1e100,
        -1e300,
        np.nextafter(np.finfo(np.float64).min, 0),
        np.finfo(np.float64).min
    ]

    for value in test_values:
        q_value = npq.QuadPrecision(str(value), backend=backend)
        print(f"  Original: {value}")
        print(f"  QuadPrecision: {q_value}")
        print(f"  Back to float64: {np.float64(q_value)}")
        print()

    # Test value beyond float64 precision
    beyond_float64_precision = "1.7976931348623157081452742373170435e+308"
    q_beyond = npq.QuadPrecision(beyond_float64_precision, backend=backend)
    print(f"  Beyond float64 precision: {q_beyond}")
    q_float64_max = npq.QuadPrecision(
        str(np.finfo(np.float64).max), backend=backend)
    diff = q_beyond - q_float64_max
    print(f"  Difference from float64 max: {diff}")
    print(
        f"  Difference is positive: {diff > npq.QuadPrecision('0', backend=backend)}")

    # Test epsilon (smallest representable difference between two numbers)
    q_epsilon = npq.QuadPrecision(
        str(np.finfo(np.float64).eps), backend=backend)
    print(f"  Float64 epsilon in QuadPrecision: {q_epsilon}")
    q_one = npq.QuadPrecision("1", backend=backend)
    q_one_plus_epsilon = q_one + q_epsilon
    print(f"  1 + epsilon != 1: {q_one_plus_epsilon != q_one}")
    print(f"  (1 + epsilon) - 1: {q_one_plus_epsilon - q_one}")


def test_ufuncs(backend):
    print(f"\nTesting ufuncs for {backend} backend:")

    # Create QuadPrecision arrays
    q_array1 = np.array([1, 2, 3], dtype=npq.QuadPrecDType(backend=backend))
    q_array2 = np.array([1, 2, 3], dtype=npq.QuadPrecDType(backend=backend))

    # Test unary ufuncs
    print("\nUnary ufuncs:")
    print(f"  negative: {np.negative(q_array1)}")
    print(f"  absolute: {np.absolute(q_array1)}")
    print(f"  rint: {np.rint(q_array1)}")
    print(f"  floor: {np.floor(q_array1)}")
    print(f"  ceil: {np.ceil(q_array1)}")
    print(f"  trunc: {np.trunc(q_array1)}")
    print(f"  sqrt: {np.sqrt(q_array1)}")
    print(f"  square: {np.square(q_array1)}")
    print(f"  log: {np.log(q_array1)}")
    print(f"  log2: {np.log2(q_array1)}")
    print(f"  log10: {np.log10(q_array1)}")
    print(f"  exp: {np.exp(q_array1)}")
    print(f"  exp2: {np.exp2(q_array1)}")

    # Test binary ufuncs
    print("\nBinary ufuncs:")
    print(f"  add: {np.add(q_array1, q_array2)}")
    print(f"  subtract: {np.subtract(q_array1, q_array2)}")
    print(f"  multiply: {np.multiply(q_array1, q_array2)}")
    print(f"  divide: {np.divide(q_array1, q_array2)}")
    print(f"  power: {np.power(q_array1, q_array2)}")
    print(f"  mod: {np.mod(q_array1, q_array2)}")
    print(f"  minimum: {np.minimum(q_array1, q_array2)}")
    print(f"  maximum: {np.maximum(q_array1, q_array2)}")

    # Test comparison ufuncs
    print("\nComparison ufuncs:")
    print(f"  equal: {np.equal(q_array1, q_array2)}")
    print(f"  not_equal: {np.not_equal(q_array1, q_array2)}")
    print(f"  less: {np.less(q_array1, q_array2)}")
    print(f"  less_equal: {np.less_equal(q_array1, q_array2)}")
    print(f"  greater: {np.greater(q_array1, q_array2)}")
    print(f"  greater_equal: {np.greater_equal(q_array1, q_array2)}")

    # Test mixed operations with numpy arrays
    print(f"Testing backend: {backend}")
    print("\nMixed operations with numpy arrays:")
    np_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    print(f"  add: {np.add(q_array1, np_array)}")
    print(f"  multiply: {np.multiply(q_array1, np_array)}")
    print(f"  divide: {np.divide(q_array1, np_array)}")

    # Test reduction operations
    print("\nReduction operations:")
    print(f"  sum: {np.sum(q_array1)}")
    print(f"  prod: {np.prod(q_array1)}")
    print(f"  min: {np.min(q_array1)}")
    print(f"  max: {np.max(q_array1)}")


# Run tests for both backends
for backend in ['longdouble']:
    test_scalar_ops(backend)
    test_casting(backend)
    test_edge_cases(backend)
    test_ufuncs(backend)

print("All tests completed successfully")
