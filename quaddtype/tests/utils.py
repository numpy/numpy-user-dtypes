from numpy_quaddtype import QuadPrecision, QuadPrecDType
import numpy as np

def assert_quad_equal(a, b, rtol=1e-15, atol=1e-15):
    """Assert two quad precision values are equal within tolerance"""
    # Ensure both operands are QuadPrecision objects for the comparison
    if not isinstance(a, QuadPrecision):
        a = QuadPrecision(str(a), backend='sleef')
    if not isinstance(b, QuadPrecision):
        b = QuadPrecision(str(b), backend='sleef')

    # Use quad-precision arithmetic to calculate the difference
    diff = abs(a - b)
    tolerance = QuadPrecision(str(atol), backend='sleef') + QuadPrecision(str(rtol), backend='sleef') * max(abs(a), abs(b))
    
    # Assert using quad-precision objects
    assert diff <= tolerance, f"Values not equal: {a} != {b} (diff: {diff}, tol: {tolerance})"


def assert_quad_array_equal(a, b, rtol=1e-25, atol=1e-25):
    """Assert two quad precision arrays are equal within tolerance"""
    assert a.shape == b.shape, f"Shapes don't match: {a.shape} vs {b.shape}"
    
    flat_a = a.flatten()
    flat_b = b.flatten()
    
    for i, (val_a, val_b) in enumerate(zip(flat_a, flat_b)):
        try:
            assert_quad_equal(val_a, val_b, rtol, atol)
        except AssertionError as e:
            raise AssertionError(f"Arrays differ at index {i}: {e}")


def create_quad_array(values, shape=None):
    """Create a QuadPrecision array from values using Sleef backend"""
    dtype = QuadPrecDType(backend='sleef')
    
    if isinstance(values, (list, tuple)):
        if shape is None:
            # 1D array
            quad_values = [QuadPrecision(str(float(v)), backend='sleef') for v in values]
            return np.array(quad_values, dtype=dtype)
        else:
            # Reshape to specified shape
            if len(shape) == 1:
                quad_values = [QuadPrecision(str(float(v)), backend='sleef') for v in values]
                return np.array(quad_values, dtype=dtype)
            elif len(shape) == 2:
                m, n = shape
                assert len(values) == m * n, f"Values length {len(values)} doesn't match shape {shape}"
                quad_matrix = []
                for i in range(m):
                    row = [QuadPrecision(str(float(values[i * n + j])), backend='sleef') for j in range(n)]
                    quad_matrix.append(row)
                return np.array(quad_matrix, dtype=dtype)
    
    raise ValueError("Unsupported values or shape")


def is_special_value(val):
    """Check if a value is NaN or infinite"""
    try:
        float_val = float(val)
        return np.isnan(float_val) or np.isinf(float_val)
    except:
        return False


def arrays_equal_with_nan(a, b, rtol=1e-15, atol=1e-15):
    """Compare arrays that may contain NaN values"""
    if a.shape != b.shape:
        return False
    
    flat_a = a.flatten()
    flat_b = b.flatten()
    
    for i, (val_a, val_b) in enumerate(zip(flat_a, flat_b)):
        # Handle NaN cases
        if is_special_value(val_a) and is_special_value(val_b):
            float_a = float(val_a)
            float_b = float(val_b)
            # Both NaN
            if np.isnan(float_a) and np.isnan(float_b):
                continue
            # Both infinite with same sign
            elif np.isinf(float_a) and np.isinf(float_b) and np.sign(float_a) == np.sign(float_b):
                continue
            else:
                return False
        elif is_special_value(val_a) or is_special_value(val_b):
            return False
        else:
            try:
                assert_quad_equal(val_a, val_b, rtol, atol)
            except AssertionError:
                return False
    
    return True