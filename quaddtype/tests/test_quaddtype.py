import pytest
import sys
import numpy as np
import operator

from mpmath import mp

import numpy_quaddtype
from numpy_quaddtype import QuadPrecDType, QuadPrecision
from numpy_quaddtype import pi as quad_pi


def test_create_scalar_simple():
    assert isinstance(QuadPrecision("12.0"), QuadPrecision)
    assert isinstance(QuadPrecision(1.63), QuadPrecision)
    assert isinstance(QuadPrecision(1), QuadPrecision)


@pytest.mark.parametrize("int_val", [
    # Very large integers that exceed long double range
    2 ** 1024,
    2 ** 2048,
    10 ** 308,
    10 ** 4000,
    # Edge cases
    0,
    1,
    -1,
    # Negative large integers
    -(2 ** 1024),
])
def test_create_scalar_from_large_int(int_val):
    """Test that QuadPrecision can handle very large integers beyond long double range.
    
    This test ensures that integers like 2**1024, which overflow standard long double,
    are properly converted via string representation to QuadPrecision without raising
    overflow errors. The conversion should match the string-based conversion.
    """
    # Convert large int to QuadPrecision
    result = QuadPrecision(int_val)
    assert isinstance(result, QuadPrecision)
    
    # String conversion should give the same result
    str_val = str(int_val)
    result_from_str = QuadPrecision(str_val)
    
    # Both conversions should produce the same value
    # (can be inf==inf on some platforms for very large values)
    assert result == result_from_str
    
    # For zero and small values, verify exact conversion
    if int_val == 0:
        assert float(result) == 0.0
    elif abs(int_val) == 1:
        assert float(result) == float(int_val)


def test_create_scalar_from_int_with_broken_str():
    """Test that QuadPrecision handles errors when __str__ fails on large integers.
    
    This test checks the error handling path in scalar.c where PyObject_Str(py_int)
    returns NULL. We simulate this by subclassing int with a __str__ method
    that raises an exception.
    """
    class BrokenInt(int):
        def __str__(self):
            raise RuntimeError("Intentionally broken __str__ method")
    
    # Create an instance with a value that will overflow long long (> 2**63 - 1)
    # This triggers the string conversion path in quad_from_py_int
    broken_int = BrokenInt(2 ** 1024)
    
    # When PyLong_AsLongLongAndOverflow returns overflow,
    # it tries to convert to string, which should fail and propagate the error
    with pytest.raises(RuntimeError, match="Intentionally broken __str__ method"):
        QuadPrecision(broken_int)


class TestQuadPrecisionArrayCreation:
    """Test suite for QuadPrecision array creation from sequences and arrays."""
    
    def test_create_array_from_list(self):
        """Test that QuadPrecision can create arrays from lists."""
        # Test with simple list
        result = QuadPrecision([3, 4, 5])
        assert isinstance(result, np.ndarray)
        assert result.dtype.name == "QuadPrecDType128"
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, np.array([3, 4, 5], dtype=QuadPrecDType(backend='sleef')))
        
        # Test with float list
        result = QuadPrecision([1.5, 2.5, 3.5])
        assert isinstance(result, np.ndarray)
        assert result.dtype.name == "QuadPrecDType128"
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, np.array([1.5, 2.5, 3.5], dtype=QuadPrecDType(backend='sleef')))

    def test_create_array_from_tuple(self):
        """Test that QuadPrecision can create arrays from tuples."""
        result = QuadPrecision((10, 20, 30))
        assert isinstance(result, np.ndarray)
        assert result.dtype.name == "QuadPrecDType128"
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, np.array([10, 20, 30], dtype=QuadPrecDType(backend='sleef')))

    def test_create_array_from_ndarray(self):
        """Test that QuadPrecision can create arrays from numpy arrays."""
        arr = np.array([1, 2, 3, 4])
        result = QuadPrecision(arr)
        assert isinstance(result, np.ndarray)
        assert result.dtype.name == "QuadPrecDType128"
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, arr.astype(QuadPrecDType(backend='sleef')))

    def test_create_2d_array_from_nested_list(self):
        """Test that QuadPrecision can create 2D arrays from nested lists."""
        result = QuadPrecision([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)
        assert result.dtype.name == "QuadPrecDType128"
        assert result.shape == (2, 2)
        expected = np.array([[1, 2], [3, 4]], dtype=QuadPrecDType(backend='sleef'))
        np.testing.assert_array_equal(result, expected)

    def test_create_array_with_backend(self):
        """Test that QuadPrecision respects backend parameter for arrays."""
        # Test with sleef backend (default)
        result_sleef = QuadPrecision([1, 2, 3], backend='sleef')
        assert isinstance(result_sleef, np.ndarray)
        assert result_sleef.dtype == QuadPrecDType(backend='sleef')
        
        # Test with longdouble backend
        result_ld = QuadPrecision([1, 2, 3], backend='longdouble')
        assert isinstance(result_ld, np.ndarray)
        assert result_ld.dtype == QuadPrecDType(backend='longdouble')

    def test_quad_precision_array_vs_astype_equivalence(self):
        """Test that QuadPrecision(array) is equivalent to array.astype(QuadPrecDType)."""
        test_arrays = [
            [1, 2, 3],
            [1.5, 2.5, 3.5],
            [[1, 2], [3, 4]],
            np.array([10, 20, 30]),
        ]
        
        for arr in test_arrays:
            result_quad = QuadPrecision(arr)
            result_astype = np.array(arr).astype(QuadPrecDType(backend='sleef'))
            np.testing.assert_array_equal(result_quad, result_astype)
            assert result_quad.dtype == result_astype.dtype

    def test_create_empty_array(self):
        """Test that QuadPrecision can create arrays from empty sequences."""
        result = QuadPrecision([])
        assert isinstance(result, np.ndarray)
        assert result.dtype.name == "QuadPrecDType128"
        assert result.shape == (0,)
        expected = np.array([], dtype=QuadPrecDType(backend='sleef'))
        np.testing.assert_array_equal(result, expected)

    def test_create_from_numpy_int_scalars(self):
        """Test that QuadPrecision can create scalars from numpy integer types."""
        # Test np.int32
        result = QuadPrecision(np.int32(42))
        assert isinstance(result, QuadPrecision)
        assert float(result) == 42.0
        
        # Test np.int64
        result = QuadPrecision(np.int64(100))
        assert isinstance(result, QuadPrecision)
        assert float(result) == 100.0
        
        # Test np.uint32
        result = QuadPrecision(np.uint32(255))
        assert isinstance(result, QuadPrecision)
        assert float(result) == 255.0
        
        # Test np.int8
        result = QuadPrecision(np.int8(-128))
        assert isinstance(result, QuadPrecision)
        assert float(result) == -128.0

    def test_create_from_numpy_float_scalars(self):
        """Test that QuadPrecision can create scalars from numpy floating types."""
        # Test np.float64
        result = QuadPrecision(np.float64(3.14))
        assert isinstance(result, QuadPrecision)
        assert abs(float(result) - 3.14) < 1e-10
        
        # Test np.float32
        result = QuadPrecision(np.float32(2.71))
        assert isinstance(result, QuadPrecision)
        # Note: float32 has limited precision, so we use a looser tolerance
        assert abs(float(result) - 2.71) < 1e-5
        
        # Test np.float16
        result = QuadPrecision(np.float16(1.5))
        assert isinstance(result, QuadPrecision)
        assert abs(float(result) - 1.5) < 1e-3

    def test_create_from_numpy_bool_scalars(self):
        """Test that QuadPrecision can create scalars from numpy boolean types."""
        # Test np.bool_(True) converts to 1.0
        result = QuadPrecision(np.bool_(True))
        assert isinstance(result, QuadPrecision)
        assert float(result) == 1.0
        
        # Test np.bool_(False) converts to 0.0
        result = QuadPrecision(np.bool_(False))
        assert isinstance(result, QuadPrecision)
        assert float(result) == 0.0

    def test_create_from_zero_dimensional_array(self):
        """Test that QuadPrecision can create from 0-d numpy arrays."""
        # 0-d array from scalar
        arr_0d = np.array(5.5)
        result = QuadPrecision(arr_0d)
        assert isinstance(result, np.ndarray)
        assert result.shape == ()  # 0-d array
        assert result.dtype.name == "QuadPrecDType128"
        expected = np.array(5.5, dtype=QuadPrecDType(backend='sleef'))
        np.testing.assert_array_equal(result, expected)
        
        # Another test with integer
        arr_0d = np.array(42)
        result = QuadPrecision(arr_0d)
        assert isinstance(result, np.ndarray)
        assert result.shape == ()
        expected = np.array(42, dtype=QuadPrecDType(backend='sleef'))
        np.testing.assert_array_equal(result, expected)

    def test_numpy_scalar_with_backend(self):
        """Test that numpy scalars respect the backend parameter."""
        # Test with sleef backend
        result = QuadPrecision(np.int32(10), backend='sleef')
        assert isinstance(result, QuadPrecision)
        assert "backend='sleef'" in repr(result)
        
        # Test with longdouble backend
        result = QuadPrecision(np.float64(3.14), backend='longdouble')
        assert isinstance(result, QuadPrecision)
        assert "backend='longdouble'" in repr(result)

    def test_numpy_scalar_types_coverage(self):
        """Test a comprehensive set of numpy scalar types."""
        # Integer types
        int_types = [
            (np.int8, 10),
            (np.int16, 1000),
            (np.int32, 100000),
            (np.int64, 10000000),
            (np.uint8, 200),
            (np.uint16, 50000),
            (np.uint32, 4000000000),
        ]
        
        for dtype, value in int_types:
            result = QuadPrecision(dtype(value))
            assert isinstance(result, QuadPrecision), f"Failed for {dtype.__name__}"
            assert float(result) == float(value), f"Value mismatch for {dtype.__name__}"
        
        # Float types
        float_types = [
            (np.float16, 1.5),
            (np.float32, 2.5),
            (np.float64, 3.5),
        ]
        
        for dtype, value in float_types:
            result = QuadPrecision(dtype(value))
            assert isinstance(result, QuadPrecision), f"Failed for {dtype.__name__}"
            # Use appropriate tolerance based on dtype precision
            expected = float(dtype(value))
            assert abs(float(result) - expected) < 1e-5, f"Value mismatch for {dtype.__name__}"


def test_string_roundtrip():
    # Test with various values that require full quad precision
    test_values = [
        QuadPrecision("0.417022004702574000667425480060047"),  # Random value
        QuadPrecision("1.23456789012345678901234567890123456789"),  # Many digits
        numpy_quaddtype.pi,  # Mathematical constant
        numpy_quaddtype.e,
        QuadPrecision("1e-100"),  # Very small
        QuadPrecision("1e100"),   # Very large
        QuadPrecision("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233"),  # very precise pi
    ]
    
    for original in test_values:
        string_repr = str(original)
        reconstructed = QuadPrecision(string_repr)
        
        # Values should be exactly equal (bit-for-bit identical)
        assert reconstructed == original, (
            f"Round-trip failed for {repr(original)}:\n"
            f"  Original:      {repr(original)}\n"
            f"  String:        {string_repr}\n"
            f"  Reconstructed: {repr(reconstructed)}"
        )
        
        # Also verify repr() preserves value
        repr_str = repr(original)
        # Extract the string value from repr format: QuadPrecision('value', backend='...')
        value_from_repr = repr_str.split("'")[1]
        reconstructed_from_repr = QuadPrecision(value_from_repr)
        
        assert reconstructed_from_repr == original, (
            f"Round-trip from repr() failed for {repr(original)}"
        )


class TestBytesSupport:
    """Test suite for QuadPrecision bytes input support."""
    
    @pytest.mark.parametrize("original", [
        QuadPrecision("0.417022004702574000667425480060047"),  # Random value
        QuadPrecision("1.23456789012345678901234567890123456789"),  # Many digits
        pytest.param(numpy_quaddtype.pi, id="pi"),  # Mathematical constant
        pytest.param(numpy_quaddtype.e, id="e"),
        QuadPrecision("1e-100"),  # Very small
        QuadPrecision("1e100"),   # Very large
        QuadPrecision("-3.14159265358979323846264338327950288419"),  # Negative pi
        QuadPrecision("0.0"),  # Zero
        QuadPrecision("-0.0"),  # Negative zero
        QuadPrecision("1.0"),  # One
        QuadPrecision("-1.0"),  # Negative one
    ])
    def test_bytes_roundtrip(self, original):
        """Test that bytes representations of quad precision values roundtrip correctly."""
        string_repr = str(original)
        bytes_repr = string_repr.encode("ascii")
        reconstructed = QuadPrecision(bytes_repr)
        
        # Values should be exactly equal (bit-for-bit identical)
        assert reconstructed == original, (
            f"Bytes round-trip failed for {repr(original)}:\n"
            f"  Original:      {repr(original)}\n"
            f"  Bytes:         {bytes_repr}\n"
            f"  Reconstructed: {repr(reconstructed)}"
        )
    
    @pytest.mark.parametrize("bytes_val,expected_str", [
        # Simple numeric values
        (b"1.0", "1.0"),
        (b"-1.0", "-1.0"),
        (b"0.0", "0.0"),
        (b"3.14159", "3.14159"),
        # Scientific notation
        (b"1e10", "1e10"),
        (b"1e-10", "1e-10"),
        (b"2.5e100", "2.5e100"),
        (b"-3.7e-50", "-3.7e-50"),
    ])
    def test_bytes_creation_basic(self, bytes_val, expected_str):
        """Test basic creation of QuadPrecision from bytes objects."""
        assert QuadPrecision(bytes_val) == QuadPrecision(expected_str)
    
    @pytest.mark.parametrize("bytes_val,check_func", [
        # Very large and very small numbers
        (b"1e308", lambda x: x == QuadPrecision("1e308")),
        (b"1e-308", lambda x: x == QuadPrecision("1e-308")),
        # Special values
        (b"inf", lambda x: np.isinf(x)),
        (b"-inf", lambda x: np.isinf(x) and x < 0),
        (b"nan", lambda x: np.isnan(x)),
    ])
    def test_bytes_creation_edge_cases(self, bytes_val, check_func):
        """Test edge cases for QuadPrecision creation from bytes."""
        val = QuadPrecision(bytes_val)
        assert check_func(val)
    
    @pytest.mark.parametrize("invalid_bytes", [
        b"",  # Empty bytes
        b"not_a_number",  # Invalid format
        b"1.23abc",  # Trailing garbage
        b"abc1.23",  # Leading garbage
    ])
    def test_bytes_invalid_input(self, invalid_bytes):
        """Test that invalid bytes input raises appropriate errors."""
        with pytest.raises(ValueError, match="Unable to parse bytes to QuadPrecision"):
            QuadPrecision(invalid_bytes)
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    @pytest.mark.parametrize("bytes_val", [
        b"1.0",
        b"-1.0",
        b"3.141592653589793238462643383279502884197",
        b"1e100",
        b"1e-100",
        b"0.0",
    ])
    def test_bytes_backend_consistency(self, backend, bytes_val):
        """Test that bytes parsing works consistently across backends."""
        quad_val = QuadPrecision(bytes_val, backend=backend)
        str_val = QuadPrecision(bytes_val.decode("ascii"), backend=backend)
        
        # Bytes and string should produce identical results
        assert quad_val == str_val, (
            f"Backend {backend}: bytes and string parsing differ for {bytes_val}\n"
            f"  From bytes:  {repr(quad_val)}\n"
            f"  From string: {repr(str_val)}"
        )
    
    @pytest.mark.parametrize("bytes_val,expected_str", [
        # Leading whitespace is OK (consumed by parser)
        (b" 1.0", "1.0"),
        (b"  3.14", "3.14"),
    ])
    def test_bytes_whitespace_valid(self, bytes_val, expected_str):
        """Test handling of valid whitespace in bytes input."""
        assert QuadPrecision(bytes_val) == QuadPrecision(expected_str)
    
    @pytest.mark.parametrize("invalid_bytes", [
        b"1.0 ",  # Trailing whitespace
        b"1.0  ",  # Multiple trailing spaces
        b"1 .0",  # Internal whitespace
        b"1. 0",  # Internal whitespace
    ])
    def test_bytes_whitespace_invalid(self, invalid_bytes):
        """Test that invalid whitespace in bytes input raises errors."""
        with pytest.raises(ValueError, match="Unable to parse bytes to QuadPrecision"):
            QuadPrecision(invalid_bytes)
    
    @pytest.mark.parametrize("test_str", [
        "1.0",
        "-3.14159265358979323846264338327950288419",
        "1e100",
        "2.71828182845904523536028747135266249775",
    ])
    def test_bytes_encoding_compatibility(self, test_str):
        """Test that bytes created from different encodings work correctly."""
        from_string = QuadPrecision(test_str)
        from_bytes = QuadPrecision(test_str.encode("ascii"))
        from_bytes_utf8 = QuadPrecision(test_str.encode("utf-8"))
        
        assert from_string == from_bytes
        assert from_string == from_bytes_utf8


def test_string_subclass_parsing():
    """Test that QuadPrecision handles string subclasses correctly.
    
    This tests the PyUnicode_Check path in scalar.c lines 195-209,
    verifying that string subclasses work and that parsing errors
    are properly handled.
    """
    class MyString(str):
        """A custom string subclass"""
        pass
    
    # Test valid string subclass - should parse correctly
    valid_str = MyString("3.14159265358979323846")
    result = QuadPrecision(valid_str)
    assert isinstance(result, QuadPrecision)
    expected = QuadPrecision("3.14159265358979323846")
    assert result == expected
    
    # Test with scientific notation
    sci_str = MyString("1.23e-100")
    result = QuadPrecision(sci_str)
    assert isinstance(result, QuadPrecision)
    
    # Test with negative value
    neg_str = MyString("-42.5")
    result = QuadPrecision(neg_str)
    assert float(result) == -42.5
    
    # Test invalid string - should raise ValueError
    invalid_str = MyString("not a number")
    with pytest.raises(ValueError, match="Unable to parse string to QuadPrecision"):
        QuadPrecision(invalid_str)
    
    # Test partially valid string (has trailing garbage)
    partial_str = MyString("3.14abc")
    with pytest.raises(ValueError, match="Unable to parse string to QuadPrecision"):
        QuadPrecision(partial_str)
    
    # Test empty string
    empty_str = MyString("")
    with pytest.raises(ValueError, match="Unable to parse string to QuadPrecision"):
        QuadPrecision(empty_str)
    
    # Test string with leading garbage
    leading_garbage = MyString("abc3.14")
    with pytest.raises(ValueError, match="Unable to parse string to QuadPrecision"):
        QuadPrecision(leading_garbage)
    
    # Test special values
    inf_str = MyString("inf")
    result = QuadPrecision(inf_str)
    assert np.isinf(float(result))
    
    neg_inf_str = MyString("-inf")
    result = QuadPrecision(neg_inf_str)
    assert np.isinf(float(result)) and float(result) < 0
    
    nan_str = MyString("nan")
    result = QuadPrecision(nan_str)
    assert np.isnan(float(result))


@pytest.mark.parametrize("name,expected", [("pi", np.pi), ("e", np.e), ("log2e", np.log2(np.e)), ("log10e", np.log10(np.e)), ("ln2", np.log(2.0)), ("ln10", np.log(10.0))])
def test_math_constant(name, expected):
    assert isinstance(getattr(numpy_quaddtype, name), QuadPrecision)

    assert np.float64(getattr(numpy_quaddtype, name)) == expected


def test_smallest_subnormal_value():
    """Test that smallest_subnormal has the correct value across all platforms."""
    smallest_sub = numpy_quaddtype.smallest_subnormal
    repr_str = repr(smallest_sub)
    
    # The repr should show QuadPrecision('6.0e-4966', backend='sleef')
    assert "6.0e-4966" in repr_str, f"Expected '6.0e-4966' in repr, got {repr_str}"
    
    assert smallest_sub > 0, "smallest_subnormal should be positive"


@pytest.mark.parametrize("dtype", [
    "bool",
    "byte", "int8", "ubyte", "uint8",
    "short", "int16", "ushort", "uint16",
    "int", "int32", "uint", "uint32",
    "long", "ulong",
    "longlong", "int64", "ulonglong", "uint64",
    "half", "float16",
    "float", "float32",
    "double", "float64",
    "longdouble", "float96", "float128",
])
def test_supported_astype(dtype):
    if dtype in ("float96", "float128") and getattr(np, dtype, None) is None:
        pytest.skip(f"{dtype} is unsupported on the current platform")

    orig = np.array(1, dtype=dtype)
    quad = orig.astype(QuadPrecDType, casting="safe")
    back = quad.astype(dtype, casting="unsafe")

    assert quad == 1
    assert back == orig


@pytest.mark.parametrize("dtype", ["V10", "datetime64[ms]", "timedelta64[ms]"])
def test_unsupported_astype(dtype):
    if dtype == "V10":
        with pytest.raises(TypeError, match="cast"):
          np.ones((3, 3), dtype="V10").astype(QuadPrecDType, casting="unsafe")
    else:
      with pytest.raises(TypeError, match="cast"):
          np.array(1, dtype=dtype).astype(QuadPrecDType, casting="unsafe")

      with pytest.raises(TypeError, match="cast"):
          np.array(QuadPrecision(1)).astype(dtype, casting="unsafe")

class TestArrayCastStringBytes:
    @pytest.mark.parametrize("strtype", [np.str_, str])
    @pytest.mark.parametrize("input_val", [
        "3.141592653589793238462643383279502884197",
        "2.71828182845904523536028747135266249775",
        "1e100",
        "1e-100",
        "0.0",
        "-0.0",
        "inf",
        "-inf",
        "nan",
        "-nan",
    ])
    def test_cast_string_to_quad_roundtrip(self, input_val, strtype):
        str_array = np.array(input_val, dtype=strtype)
        quad_array = str_array.astype(QuadPrecDType())
        expected = np.array(input_val, dtype=QuadPrecDType())
        
        if np.isnan(float(expected)):
            np.testing.assert_array_equal(np.isnan(quad_array), np.isnan(expected))
        else:
            np.testing.assert_array_equal(quad_array, expected)
        
        quad_to_string_array = quad_array.astype(strtype)
        
        # Round-trip - String -> Quad -> String -> Quad should preserve value
        roundtrip_quad_array = quad_to_string_array.astype(QuadPrecDType())
        
        if np.isnan(float(expected)):
            np.testing.assert_array_equal(np.isnan(roundtrip_quad_array), np.isnan(quad_array))
        else:
            np.testing.assert_array_equal(roundtrip_quad_array, quad_array, 
                                         err_msg=f"Round-trip failed for {input_val}")
        
        # Verify the string representation can be parsed back
        # (This ensures the quad->string cast produces valid parseable strings)
        scalar_str = str(quad_array[()])
        scalar_from_str = QuadPrecision(scalar_str)
        
        if np.isnan(float(quad_array[()])):
            assert np.isnan(float(scalar_from_str))
        else:
            assert scalar_from_str == quad_array[()], \
                f"Scalar round-trip failed: {scalar_str} -> {scalar_from_str} != {quad_array[()]}"
    
    @pytest.mark.parametrize("input_val", [
        b"3.141592653589793238462643383279502884197",
        b"2.71828182845904523536028747135266249775",
        b"1e100",
        b"1e-100",
        b"0.0",
        b"-0.0",
        b"inf",
        b"-inf",
        b"nan",
        b"-nan",
    ])
    def test_cast_bytes_to_quad_roundtrip(self, input_val):
        """Test bytes -> quad -> bytes round-trip conversion"""
        bytes_array = np.array(input_val, dtype='S50')
        quad_array = bytes_array.astype(QuadPrecDType())
        expected = np.array(input_val.decode('utf-8'), dtype=QuadPrecDType())
        
        if np.isnan(float(expected)):
            np.testing.assert_array_equal(np.isnan(quad_array), np.isnan(expected))
        else:
            np.testing.assert_array_equal(quad_array, expected)
        
        quad_to_bytes_array = quad_array.astype('S50')
        
        # Round-trip - Bytes -> Quad -> Bytes -> Quad should preserve value
        roundtrip_quad_array = quad_to_bytes_array.astype(QuadPrecDType())
        
        if np.isnan(float(expected)):
            np.testing.assert_array_equal(np.isnan(roundtrip_quad_array), np.isnan(quad_array))
        else:
            np.testing.assert_array_equal(roundtrip_quad_array, quad_array, 
                                         err_msg=f"Round-trip failed for {input_val}")
    
    @pytest.mark.parametrize("dtype_str", ['S10', 'S20', 'S30', 'S50', 'S100'])
    def test_bytes_different_sizes(self, dtype_str):
        """Test bytes casting with different buffer sizes"""
        quad_val = np.array([1.23456789012345678901234567890], dtype=QuadPrecDType())
        bytes_val = quad_val.astype(dtype_str)
        
        # Should not raise error
        assert bytes_val.dtype.str.startswith('|S')
        
        # Should be able to parse back
        roundtrip = bytes_val.astype(QuadPrecDType())
        
        # For smaller sizes, precision may be truncated, so use approximate comparison
        # For larger sizes (S50+), should be exact
        if dtype_str in ['S50', 'S100']:
            np.testing.assert_array_equal(roundtrip, quad_val)
        else:
            # Smaller sizes may lose precision due to string truncation
            np.testing.assert_allclose(roundtrip, quad_val, rtol=1e-8)
    
    @pytest.mark.parametrize("input_bytes", [
        b'1.5',
        b'2.25',
        b'3.14159265358979323846',
        b'-1.5',
        b'-2.25',
        b'1.23e50',
        b'-4.56e-100',
    ])
    def test_bytes_to_quad_basic_values(self, input_bytes):
        """Test basic numeric bytes to quad conversion"""
        bytes_array = np.array([input_bytes], dtype='S50')
        quad_array = bytes_array.astype(QuadPrecDType())
        
        # Should successfully convert
        assert quad_array.dtype.name == "QuadPrecDType128"
        
        # Value should match string conversion
        str_val = input_bytes.decode('utf-8')
        expected = QuadPrecision(str_val)
        assert quad_array[0] == expected
    
    @pytest.mark.parametrize("special_bytes,check_func", [
        (b'inf', lambda x: np.isinf(float(str(x))) and float(str(x)) > 0),
        (b'-inf', lambda x: np.isinf(float(str(x))) and float(str(x)) < 0),
        (b'nan', lambda x: np.isnan(float(str(x)))),
        (b'Infinity', lambda x: np.isinf(float(str(x))) and float(str(x)) > 0),
        (b'-Infinity', lambda x: np.isinf(float(str(x))) and float(str(x)) < 0),
        (b'NaN', lambda x: np.isnan(float(str(x)))),
    ])
    def test_bytes_special_values(self, special_bytes, check_func):
        """Test special values (inf, nan) in bytes format"""
        bytes_array = np.array([special_bytes], dtype='S20')
        quad_array = bytes_array.astype(QuadPrecDType())
        
        assert check_func(quad_array[0]), f"Failed for {special_bytes}"
    
    def test_bytes_array_vectorized(self):
        """Test vectorized bytes to quad conversion"""
        bytes_array = np.array([b'1.5', b'2.25', b'3.14159', b'-1.0', b'1e100'], dtype='S50')
        quad_array = bytes_array.astype(QuadPrecDType())
        
        assert quad_array.shape == (5,)
        assert quad_array.dtype.name == "QuadPrecDType128"
        
        # Check individual values
        assert quad_array[0] == QuadPrecision('1.5')
        assert quad_array[1] == QuadPrecision('2.25')
        assert quad_array[2] == QuadPrecision('3.14159')
        assert quad_array[3] == QuadPrecision('-1.0')
        assert quad_array[4] == QuadPrecision('1e100')
    
    def test_quad_to_bytes_preserves_precision(self):
        """Test that quad to bytes conversion preserves high precision"""
        # Use a high-precision value
        quad_val = np.array([QuadPrecision("3.141592653589793238462643383279502884197")], 
                           dtype=QuadPrecDType())
        bytes_val = quad_val.astype('S50')
        
        # Convert back and verify precision is maintained
        roundtrip = bytes_val.astype(QuadPrecDType())
        np.testing.assert_array_equal(roundtrip, quad_val)
    
    @pytest.mark.parametrize("invalid_bytes", [
        b'not_a_number',
        b'1.23.45',
        b'abc123',
        b'1e',
        b'++1.0',
        b'1.0abc',
    ])
    def test_invalid_bytes_raise_error(self, invalid_bytes):
        """Test that invalid bytes raise ValueError"""
        bytes_array = np.array([invalid_bytes], dtype='S50')
        
        with pytest.raises(ValueError, match="could not convert bytes to QuadPrecision"):
            bytes_array.astype(QuadPrecDType())
    
    def test_bytes_with_null_terminator(self):
        """Test bytes with embedded null terminators are handled correctly"""
        # Bytes arrays can have null padding
        bytes_array = np.array([b'1.5'], dtype='S20')
        # This creates a 20-byte array with '1.5' followed by null bytes
        
        quad_array = bytes_array.astype(QuadPrecDType())
        assert quad_array[0] == QuadPrecision('1.5')
    
    def test_empty_bytes_raises_error(self):
        """Test that empty bytes raise ValueError"""
        bytes_array = np.array([b''], dtype='S50')
        
        with pytest.raises(ValueError):
            bytes_array.astype(QuadPrecDType())

    @pytest.mark.parametrize('dtype', ['S50', 'U50'])
    @pytest.mark.parametrize('size', [500, 1000, 10000])
    def test_large_array_casting(self, dtype, size):
        """Test long array casting won't lead segfault, GIL enabled"""
        arr = np.arange(size).astype(np.float32).astype(dtype)
        quad_arr = arr.astype(QuadPrecDType())
        assert quad_arr.dtype == QuadPrecDType()
        assert quad_arr.size == size

        # check roundtrip
        roundtrip = quad_arr.astype(dtype)
        np.testing.assert_array_equal(arr, roundtrip)

class TestStringParsingEdgeCases:
    """Test edge cases in NumPyOS_ascii_strtoq string parsing"""
    @pytest.mark.parametrize("input_str", ['3.14', '-2.71', '0.0', '1e10', '-1e-10'])
    @pytest.mark.parametrize("byte_order", ['<', '>'])
    def test_numeric_string_parsing(self, input_str, byte_order):
        """Test that numeric strings are parsed correctly regardless of byte order"""
        strtype = np.dtype(f'{byte_order}U20')
        arr = np.array([input_str], dtype=strtype)
        result = arr.astype(QuadPrecDType())
        
        expected = np.array(input_str, dtype=np.float64)
        
        np.testing.assert_allclose(result, expected,
                                      err_msg=f"Failed parsing '{input_str}' with byte order '{byte_order}'")


    @pytest.mark.parametrize("input_str,expected_sign", [
        ("inf", 1),
        ("+inf", 1),
        ("-inf", -1),
        ("Inf", 1),
        ("+Inf", 1),
        ("-Inf", -1),
        ("INF", 1),
        ("+INF", 1),
        ("-INF", -1),
        ("infinity", 1),
        ("+infinity", 1),
        ("-infinity", -1),
        ("Infinity", 1),
        ("+Infinity", 1),
        ("-Infinity", -1),
        ("INFINITY", 1),
        ("+INFINITY", 1),
        ("-INFINITY", -1),
    ])
    def test_infinity_sign_preservation(self, input_str, expected_sign):
        """Test that +/- signs are correctly applied to infinity values"""
        arr = np.array([input_str], dtype='U20')
        result = arr.astype(QuadPrecDType())
        
        assert np.isinf(float(str(result[0]))), f"Expected inf for '{input_str}'"
        
        actual_sign = 1 if float(str(result[0])) > 0 else -1
        assert actual_sign == expected_sign, \
            f"Sign mismatch for '{input_str}': got {actual_sign}, expected {expected_sign}"
    
    @pytest.mark.parametrize("input_str", [
        "nan", "+nan", "-nan",  # Note: NaN sign is typically ignored
        "NaN", "+NaN", "-NaN",
        "NAN", "+NAN", "-NAN",
        "nan()", "nan(123)", "nan(abc_)", "NAN(XYZ)",
    ])
    def test_nan_case_insensitive(self, input_str):
        """Test case-insensitive NaN parsing with optional payloads"""
        arr = np.array([input_str], dtype='U20')
        result = arr.astype(QuadPrecDType())
        
        assert np.isnan(float(str(result[0]))), f"Expected NaN for '{input_str}'"
    
    @pytest.mark.parametrize("input_str,expected_val", [
        ("3.14", 3.14),
        ("+3.14", 3.14),
        ("-3.14", -3.14),
        ("0.0", 0.0),
        ("+0.0", 0.0),
        ("-0.0", -0.0),
        ("1e10", 1e10),
        ("+1e10", 1e10),
        ("-1e10", -1e10),
        ("1.23e-45", 1.23e-45),
        ("+1.23e-45", 1.23e-45),
        ("-1.23e-45", -1.23e-45),
    ])
    def test_numeric_sign_handling(self, input_str, expected_val):
        """Test that +/- signs are correctly handled for numeric values"""
        arr = np.array([input_str], dtype='U20')
        result = arr.astype(QuadPrecDType())
        
        result_val = float(str(result[0]))
        
        # For zero, check sign separately
        if expected_val == 0.0:
            assert result_val == 0.0
            if input_str.startswith('-'):
                assert np.signbit(result_val), f"Expected negative zero for '{input_str}'"
            else:
                assert not np.signbit(result_val), f"Expected positive zero for '{input_str}'"
        else:
            np.testing.assert_allclose(result_val, expected_val, rtol=1e-10)
    
    @pytest.mark.parametrize("input_str", [
        "  3.14  ",
        "\t3.14\t",
        "\n3.14\n",
        "\r3.14\r",
        "  \t\n\r  3.14  \t\n\r  ",
        "  inf  ",
        "\t-inf\t",
        "  nan  ",
    ])
    def test_whitespace_handling(self, input_str):
        """Test that leading/trailing whitespace is handled correctly"""
        arr = np.array([input_str], dtype='U20')
        result = arr.astype(QuadPrecDType())
        
        # Should not raise an error
        result_str = str(result[0])
        assert result_str  # Should have a value
    
    @pytest.mark.parametrize("invalid_str", [
        "abc",           # Non-numeric
        "3.14.15",       # Multiple decimals
        "1.23e",         # Incomplete scientific notation
        "e10",           # Scientific notation without base
        "3.14abc",       # Trailing non-numeric
        "++3.14",        # Double sign
        "--3.14",        # Double sign
        "+-3.14",        # Mixed signs
        "in",            # Incomplete inf
        "na",            # Incomplete nan
        "infinit",       # Incomplete infinity
    ])
    def test_invalid_strings_raise_error(self, invalid_str):
        """Test that invalid strings raise ValueError"""
        arr = np.array([invalid_str], dtype='U20')
        
        with pytest.raises(ValueError):
            arr.astype(QuadPrecDType())
    
    @pytest.mark.parametrize("input_str", [
        "3.14ñ",         # Trailing non-ASCII
        "ñ3.14",         # Leading non-ASCII  
        "3.1€4",         # Mid non-ASCII
        "π",             # Greek pi
    ])
    def test_non_ascii_raises_error(self, input_str):
        """Test that non-ASCII characters raise ValueError"""
        arr = np.array([input_str], dtype='U20')
        
        with pytest.raises(ValueError):
            arr.astype(QuadPrecDType())
    
    def test_numpy_longdouble_compatibility(self):
        """Test that our parsing matches NumPy's longdouble for common cases"""
        test_cases = [
            "inf", "INF", "Inf", "-inf", "+infinity",
            "nan", "NAN", "NaN", 
            "3.14", "-2.718", "1e10", "-1.23e-45",
        ]
        
        for test_str in test_cases:
            arr = np.array([test_str], dtype='U20')
            
            # NumPy's built-in
            np_result = arr.astype(np.longdouble)
            
            # Our QuadPrecision
            quad_result = arr.astype(QuadPrecDType())
            
            np_val = np_result[0]
            quad_val = float(str(quad_result[0]))
            
            if np.isnan(np_val):
                assert np.isnan(quad_val), f"NumPy gives NaN but QuadPrec doesn't for '{test_str}'"
            elif np.isinf(np_val):
                assert np.isinf(quad_val), f"NumPy gives inf but QuadPrec doesn't for '{test_str}'"
                assert np.sign(np_val) == np.sign(quad_val), \
                    f"Inf sign mismatch for '{test_str}': NumPy={np.sign(np_val)}, Quad={np.sign(quad_val)}"
            else:
                np.testing.assert_allclose(quad_val, np_val, rtol=1e-10)
    
    def test_locale_independence(self):
        """Test that parsing always uses '.' for decimal, not locale-specific separator"""
        # This should parse correctly (period as decimal)
        arr = np.array(["3.14"], dtype='U20')
        result = arr.astype(QuadPrecDType())
        assert float(str(result[0])) == 3.14
        
        # In some locales ',' is decimal separator, but we should always use '.'
        # So "3,14" should either error or parse as "3" (stopping at comma)
        # Since require_full_parse validation happens in casts.cpp, and we check
        # for trailing content, this should raise ValueError
        arr_comma = np.array(["3,14"], dtype='U20')
        with pytest.raises(ValueError):
            arr_comma.astype(QuadPrecDType())

    @pytest.mark.parametrize("input_str,description", [
        ("  1.23  ", "space - leading and trailing"),
        ("\t1.23\t", "tab - leading and trailing"),
        ("\n1.23\n", "newline - leading and trailing"),
        ("\r1.23\r", "carriage return - leading and trailing"),
        ("\v1.23\v", "vertical tab - leading and trailing"),
        ("\f1.23\f", "form feed - leading and trailing"),
        (" \t\n\r\v\f1.23 \t\n\r\v\f", "all 6 whitespace chars - mixed"),
        ("\t\t\t3.14\t\t\t", "multiple tabs"),
        ("   inf   ", "infinity with spaces"),
        ("\t\t-inf\t\t", "negative infinity with tabs"),
        ("\n\nnan\n\n", "nan with newlines"),
        ("\r\r-nan\r\r", "negative nan with carriage returns"),
        ("\v\v1e10\v\v", "scientific notation with vertical tabs"),
        ("\f\f-1.23e-45\f\f", "negative scientific with form feeds"),
    ])
    def test_all_six_whitespace_characters(self, input_str, description):
        """Test all 6 ASCII whitespace characters (space, tab, newline, carriage return, vertical tab, form feed)
        
        This tests the ascii_isspace() helper function in casts.cpp which matches
        CPython's Py_ISSPACE and NumPy's NumPyOS_ascii_isspace behavior.
        The 6 characters are: 0x09(\t), 0x0A(\n), 0x0B(\v), 0x0C(\f), 0x0D(\r), 0x20(space)
        """
        arr = np.array([input_str], dtype='U50')
        result = arr.astype(QuadPrecDType())
        
        # Should successfully parse without errors
        result_val = str(result[0])
        assert result_val, f"Failed to parse with {description}"
        
        # Verify the value is correct (strip whitespace and compare)
        stripped = input_str.strip(' \t\n\r\v\f')
        expected_arr = np.array([stripped], dtype='U50')
        expected = expected_arr.astype(QuadPrecDType())
        
        if np.isnan(float(str(expected[0]))):
            assert np.isnan(float(str(result[0]))), f"NaN parsing failed for {description}"
        elif np.isinf(float(str(expected[0]))):
            assert np.isinf(float(str(result[0]))), f"Inf parsing failed for {description}"
            assert np.sign(float(str(expected[0]))) == np.sign(float(str(result[0]))), \
                f"Inf sign mismatch for {description}"
        else:
            assert result[0] == expected[0], f"Value mismatch for {description}"

    @pytest.mark.parametrize("invalid_str,description", [
        ("1.23 abc", "trailing non-whitespace after number"),
        ("  1.23xyz  ", "trailing garbage with surrounding whitespace"),
        ("abc 123", "leading garbage before number"),
        ("1.23\x01", "control char (SOH) after number"),
        ("1.23    a", "letter after multiple spaces"),
        ("\t1.23\tabc\t", "tabs with garbage in middle"),
    ])
    def test_whitespace_with_invalid_trailing_content(self, invalid_str, description):
        """Test that strings with invalid trailing content are rejected even with whitespace
        
        This ensures the trailing whitespace check in casts.cpp properly validates
        that only whitespace follows the parsed number, not other characters.
        """
        arr = np.array([invalid_str], dtype='U50')
        
        with pytest.raises(ValueError, match="could not convert string to QuadPrecision"):
            arr.astype(QuadPrecDType())
        
    @pytest.mark.parametrize("input_bytes", [
        b'3.14', b'-2.71', b'0.0', b'1e10', b'-1e-10'
    ])
    def test_bytes_numeric_parsing(self, input_bytes):
        """Test that numeric bytes are parsed correctly"""
        arr = np.array([input_bytes], dtype='S20')
        result = arr.astype(QuadPrecDType())
        
        expected = np.array(input_bytes.decode('utf-8'), dtype=np.float64)
        np.testing.assert_allclose(result, expected,
                                  err_msg=f"Failed parsing bytes {input_bytes}")
    
    @pytest.mark.parametrize("input_bytes,expected_sign", [
        (b"inf", 1),
        (b"+inf", 1),
        (b"-inf", -1),
        (b"Inf", 1),
        (b"+Inf", 1),
        (b"-Inf", -1),
        (b"INF", 1),
        (b"+INF", 1),
        (b"-INF", -1),
        (b"infinity", 1),
        (b"+infinity", 1),
        (b"-infinity", -1),
        (b"Infinity", 1),
        (b"+Infinity", 1),
        (b"-Infinity", -1),
        (b"INFINITY", 1),
        (b"+INFINITY", 1),
        (b"-INFINITY", -1),
    ])
    def test_bytes_infinity_sign_preservation(self, input_bytes, expected_sign):
        """Test that +/- signs are correctly applied to infinity values in bytes"""
        arr = np.array([input_bytes], dtype='S20')
        result = arr.astype(QuadPrecDType())
        
        assert np.isinf(float(str(result[0]))), f"Expected inf for bytes {input_bytes}"
        
        actual_sign = 1 if float(str(result[0])) > 0 else -1
        assert actual_sign == expected_sign, \
            f"Sign mismatch for bytes {input_bytes}: got {actual_sign}, expected {expected_sign}"
    
    @pytest.mark.parametrize("input_bytes", [
        b"nan", b"+nan", b"-nan",
        b"NaN", b"+NaN", b"-NaN",
        b"NAN", b"+NAN", b"-NAN",
        b"nan()", b"nan(123)", b"nan(abc_)", b"NAN(XYZ)",
    ])
    def test_bytes_nan_case_insensitive(self, input_bytes):
        """Test case-insensitive NaN parsing with optional payloads in bytes"""
        arr = np.array([input_bytes], dtype='S20')
        result = arr.astype(QuadPrecDType())
        
        assert np.isnan(float(str(result[0]))), f"Expected NaN for bytes {input_bytes}"
    
    @pytest.mark.parametrize("input_bytes,expected_val", [
        (b"3.14", 3.14),
        (b"+3.14", 3.14),
        (b"-3.14", -3.14),
        (b"0.0", 0.0),
        (b"+0.0", 0.0),
        (b"-0.0", -0.0),
        (b"1e10", 1e10),
        (b"+1e10", 1e10),
        (b"-1e10", -1e10),
        (b"1.23e-45", 1.23e-45),
        (b"+1.23e-45", 1.23e-45),
        (b"-1.23e-45", -1.23e-45),
    ])
    def test_bytes_numeric_sign_handling(self, input_bytes, expected_val):
        """Test that +/- signs are correctly handled for numeric values in bytes"""
        arr = np.array([input_bytes], dtype='S20')
        result = arr.astype(QuadPrecDType())
        
        result_val = float(str(result[0]))
        
        # For zero, check sign separately
        if expected_val == 0.0:
            assert result_val == 0.0
            if input_bytes.startswith(b'-'):
                assert np.signbit(result_val), f"Expected negative zero for bytes {input_bytes}"
            else:
                assert not np.signbit(result_val), f"Expected positive zero for bytes {input_bytes}"
        else:
            np.testing.assert_allclose(result_val, expected_val, rtol=1e-10)
    
    @pytest.mark.parametrize("input_bytes", [
        b"  3.14  ",
        b"\t3.14\t",
        b"\n3.14\n",
        b"\r3.14\r",
        b"  \t\n\r  3.14  \t\n\r  ",
        b"  inf  ",
        b"\t-inf\t",
        b"  nan  ",
    ])
    def test_bytes_whitespace_handling(self, input_bytes):
        """Test that leading/trailing whitespace is handled correctly in bytes"""
        arr = np.array([input_bytes], dtype='S50')
        result = arr.astype(QuadPrecDType())
        
        # Should not raise an error
        result_str = str(result[0])
        assert result_str  # Should have a value
    
    @pytest.mark.parametrize("invalid_bytes", [
        b"abc",           # Non-numeric
        b"3.14.15",       # Multiple decimals
        b"1.23e",         # Incomplete scientific notation
        b"e10",           # Scientific notation without base
        b"3.14abc",       # Trailing non-numeric
        b"++3.14",        # Double sign
        b"--3.14",        # Double sign
        b"+-3.14",        # Mixed signs
        b"in",            # Incomplete inf
        b"na",            # Incomplete nan
        b"infinit",       # Incomplete infinity
    ])
    def test_bytes_invalid_raises_error(self, invalid_bytes):
        """Test that invalid bytes raise ValueError"""
        arr = np.array([invalid_bytes], dtype='S20')
        
        with pytest.raises(ValueError, match="could not convert bytes to QuadPrecision"):
            arr.astype(QuadPrecDType())
    
    @pytest.mark.parametrize("input_bytes,description", [
        (b"  1.23  ", "space - leading and trailing"),
        (b"\t1.23\t", "tab - leading and trailing"),
        (b"\n1.23\n", "newline - leading and trailing"),
        (b"\r1.23\r", "carriage return - leading and trailing"),
        (b" \t\n\r1.23 \t\n\r", "mixed whitespace"),
        (b"\t\t\t3.14\t\t\t", "multiple tabs"),
        (b"   inf   ", "infinity with spaces"),
        (b"\t\t-inf\t\t", "negative infinity with tabs"),
        (b"\n\nnan\n\n", "nan with newlines"),
    ])
    def test_bytes_whitespace_characters(self, input_bytes, description):
        """Test all ASCII whitespace characters in bytes format"""
        arr = np.array([input_bytes], dtype='S50')
        result = arr.astype(QuadPrecDType())
        
        # Should successfully parse without errors
        result_val = str(result[0])
        assert result_val, f"Failed to parse bytes with {description}"
    
    @pytest.mark.parametrize("invalid_bytes,description", [
        (b"1.23 abc", "trailing non-whitespace after number"),
        (b"  1.23xyz  ", "trailing garbage with surrounding whitespace"),
        (b"abc 123", "leading garbage before number"),
        (b"1.23    a", "letter after multiple spaces"),
        (b"\t1.23\tabc\t", "tabs with garbage in middle"),
    ])
    def test_bytes_whitespace_with_invalid_trailing(self, invalid_bytes, description):
        """Test that bytes with invalid trailing content are rejected even with whitespace"""
        arr = np.array([invalid_bytes], dtype='S50')
        
        with pytest.raises(ValueError, match="could not convert bytes to QuadPrecision"):
            arr.astype(QuadPrecDType())
    
    def test_bytes_null_padding(self):
        """Test that null-padded bytes are handled correctly"""
        # Create a bytes array with explicit null padding
        arr = np.array([b'1.5'], dtype='S20')  # 20 bytes with '1.5' followed by nulls
        result = arr.astype(QuadPrecDType())
        
        assert result[0] == QuadPrecision('1.5')
    
    def test_bytes_exact_size_no_null(self):
        """Test bytes array where content exactly fills the buffer"""
        # Create a string that exactly fits
        test_str = b'1.234567890'  # 11 bytes
        arr = np.array([test_str], dtype='S11')
        result = arr.astype(QuadPrecDType())
        
        expected = QuadPrecision(test_str.decode('utf-8'))
        assert result[0] == expected
    
    @pytest.mark.parametrize("size", [10, 20, 50, 100])
    def test_bytes_various_buffer_sizes(self, size):
        """Test bytes parsing with various buffer sizes"""
        test_bytes = b'3.14159'
        dtype_str = f'S{size}'
        arr = np.array([test_bytes], dtype=dtype_str)
        result = arr.astype(QuadPrecDType())
        
        expected = QuadPrecision(test_bytes.decode('utf-8'))
        assert result[0] == expected
    
    def test_bytes_scientific_notation_variations(self):
        """Test various scientific notation formats in bytes"""
        test_cases = [
            (b'1e10', 1e10),
            (b'1E10', 1e10),
            (b'1.23e-45', 1.23e-45),
            (b'1.23E-45', 1.23e-45),
            (b'1.23e+45', 1.23e45),
            (b'1.23E+45', 1.23e45),
        ]
        
        for input_bytes, expected_val in test_cases:
            arr = np.array([input_bytes], dtype='S20')
            result = arr.astype(QuadPrecDType())
            result_val = float(str(result[0]))
            np.testing.assert_allclose(result_val, expected_val, rtol=1e-10,
                                      err_msg=f"Failed for {input_bytes}")
    
    def test_bytes_high_precision_values(self):
        """Test bytes with very high precision numeric strings"""
        high_precision_bytes = b'3.141592653589793238462643383279502884197'
        arr = np.array([high_precision_bytes], dtype='S50')
        result = arr.astype(QuadPrecDType())
        
        # Should not raise error and produce a valid quad value
        assert result.dtype.name == "QuadPrecDType128"
        
        # Round-trip should preserve value
        back_to_bytes = result.astype('S50')
        roundtrip = back_to_bytes.astype(QuadPrecDType())
        np.testing.assert_array_equal(roundtrip, result)

    def test_empty_string_and_whitespace_only(self):
        """Test that empty strings and whitespace-only strings raise errors"""
        test_cases = [
            "",           # Empty string
            " ",          # Single space
            "  ",         # Multiple spaces
            "\t",         # Single tab
            "\n",         # Single newline
            "\r",         # Single carriage return
            "\v",         # Single vertical tab
            "\f",         # Single form feed
            " \t\n\r\v\f", # All whitespace characters
            "   \t\t\n\n  ", # Mixed whitespace
        ]
        
        for test_str in test_cases:
            arr = np.array([test_str], dtype='U20')
            with pytest.raises(ValueError, match="could not convert string to QuadPrecision"):
                arr.astype(QuadPrecDType())

    @pytest.mark.parametrize("boundary_str,description", [
        ("1e4932", "near max exponent for quad precision"),
        ("1e-4932", "near min exponent for quad precision"),
        ("1.189731495357231765085759326628007016196477" + "e4932", "very large number"),
        ("3.362103143112093506262677817321752602596e-4932", "very small number"),
        ("-1.189731495357231765085759326628007016196477" + "e4932", "very large negative"),
        ("-3.362103143112093506262677817321752602596e-4932", "very small negative"),
    ])
    def test_extreme_exponent_values(self, boundary_str, description):
        """Test parsing of numbers with extreme exponents near quad precision limits
        
        IEEE 754 binary128 has exponent range of approximately ±4932
        """
        arr = np.array([boundary_str], dtype='U100')
        result = arr.astype(QuadPrecDType())
        
        # Should parse successfully (may result in inf for overflow cases)
        result_str = str(result[0])
        assert result_str, f"Failed to parse {description}"

    @pytest.mark.parametrize("precision_str", [
        "3.141592653589793238462643383279502884197",  # 36 digits (quad precision)
        "2.718281828459045235360287471352662497757",  # e to 36 digits
        "1.414213562373095048801688724209698078569",  # sqrt(2) to 36 digits
        "-1.732050807568877293527446341505872366942", # -sqrt(3) to 36 digits
    ])
    def test_full_precision_parsing(self, precision_str):
        """Test that strings with full quad precision (36 decimal digits) parse correctly
        
        This ensures the full precision is preserved during string -> quad conversion
        """
        arr = np.array([precision_str], dtype='U50')
        result = arr.astype(QuadPrecDType())
        
        # Convert back to string and verify roundtrip preserves precision
        back_to_str = result.astype('U50')
        roundtrip = back_to_str.astype(QuadPrecDType())
        
        # Roundtrip should preserve the value
        assert result[0] == roundtrip[0], \
            f"Precision lost in roundtrip for {precision_str}"


def test_basic_equality():
    assert QuadPrecision("12") == QuadPrecision(
        "12.0") == QuadPrecision("12.00")


@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "pow", "copysign"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_binary_ops(op, a, b):
    if op == "truediv" and float(b) == 0:
        pytest.xfail("float division by zero")

    op_func = getattr(operator, op, None) or getattr(np, op)
    quad_a = QuadPrecision(a)
    quad_b = QuadPrecision(b)
    float_a = float(a)
    float_b = float(b)

    quad_result = op_func(quad_a, quad_b)
    float_result = op_func(float_a, float_b)

    np.testing.assert_allclose(np.float64(quad_result), float_result, atol=1e-10, rtol=0, equal_nan=True)

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({a}, {b})"


@pytest.mark.parametrize("op", ["eq", "ne", "le", "lt", "ge", "gt"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_comparisons(op, a, b):
    op_func = getattr(operator, op)
    quad_a = QuadPrecision(a)
    quad_b = QuadPrecision(b)
    float_a = float(a)
    float_b = float(b)

    assert op_func(quad_a, quad_b) == op_func(float_a, float_b)


@pytest.mark.parametrize("op", ["eq", "ne", "le", "lt", "ge", "gt"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_array_comparisons(op, a, b):
    op_func = getattr(operator, op)
    quad_a = np.array(QuadPrecision(a))
    quad_b = np.array(QuadPrecision(b))
    float_a = np.array(float(a))
    float_b = np.array(float(b))

    assert np.array_equal(op_func(quad_a, quad_b), op_func(float_a, float_b))


@pytest.mark.parametrize("op", ["minimum", "maximum", "fmin", "fmax"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_array_minmax(op, a, b):
    op_func = getattr(np, op)
    quad_a = np.array([QuadPrecision(a)])
    quad_b = np.array([QuadPrecision(b)])
    float_a = np.array([float(a)])
    float_b = np.array([float(b)])

    quad_res = op_func(quad_a, quad_b)
    float_res = op_func(float_a, float_b)

    # native implementation may not be sensitive to zero signs
    #  but we want to enforce it for the quad dtype
    # e.g. min(+0.0, -0.0) = -0.0
    if float_a == 0.0 and float_b == 0.0:
        assert float_res == 0.0
        float_res = np.copysign(0.0, op_func(np.copysign(1.0, float_a), np.copysign(1.0, float_b)))

    np.testing.assert_array_equal(quad_res.astype(float), float_res)

    # Check sign for zero results
    if float_res == 0.0:
        assert np.signbit(float_res) == np.signbit(
            quad_res), f"Zero sign mismatch for {op}({a}, {b})"

class TestComparisonReductionOps:
    """Test suite for comparison reduction operations on QuadPrecision arrays."""
    
    @pytest.mark.parametrize("op", ["all", "any"])
    @pytest.mark.parametrize("input_array", [
        (["1.0", "2.0", "3.0"]),
        (["1.0", "0.0", "3.0"]),
        (["0.0", "0.0", "0.0"]),
        # Including negative zero
        (["-0.0", "0.0"]),
        # Including NaN (should be treated as true)
        (["nan", "1.0"]),
        (["nan", "0.0"]),
        (["nan", "nan"]),
        # inf cases
        (["inf", "1.0"]),
        (["-inf", "0.0"]),
        (["inf", "-inf"]),
        # Mixed cases
        (["1.0", "-0.0", "nan", "inf"]),
        (["0.0", "-0.0", "nan", "-inf"]),
    ])
    def test_reduction_ops(self, op, input_array):
        """Test all and any reduction operations."""
        quad_array = np.array([QuadPrecision(x) for x in input_array])
        float_array = np.array([float(x) for x in input_array])
        op = getattr(np, op)
        result = op(quad_array)
        expected = op(float_array)
        
        assert result == expected, (
            f"Reduction op '{op}' failed for input {input_array}: "
            f"expected {expected}, got {result}"
        )

    @pytest.mark.parametrize("val_str", [
        "0.0",
        "-0.0",
        "1.0",
        "-1.0",
        "nan",
        "inf",
        "-inf",
    ])
    def test_scalar_reduction_ops(self, val_str):
        """Test reduction operations on scalar QuadPrecision values."""
        quad_val = QuadPrecision(val_str)
        float_val = np.float64(val_str)

        result_all = quad_val.all()
        expected_all_result = float_val.all()
        assert result_all == expected_all_result, (
            f"Scalar all failed for {val_str}: expected {expected_all_result}, got {result_all}"
        )
        
        result_any = quad_val.any()
        expected_any_result = float_val.any()
        assert result_any == expected_any_result, (
            f"Scalar any failed for {val_str}: expected {expected_any_result}, got {result_any}"
        )


# Logical operations tests
@pytest.mark.parametrize("op", ["logical_and", "logical_or", "logical_xor"])
@pytest.mark.parametrize("x1,x2", [
    # Basic cases
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 2.0),
    (2.5, 3.7),
    # Negative values
    (-1.0, 1.0),
    (-2.0, -3.0),
    # Negative zero (also falsy)
    (-0.0, 0.0),
    (-0.0, 1.0),
    (1.0, -0.0),
    (-0.0, -0.0),
    # Special values: NaN and inf are truthy
    (np.nan, 0.0),
    (0.0, np.nan),
    (np.nan, 1.0),
    (1.0, np.nan),
    (np.nan, np.nan),
    (np.inf, 0.0),
    (0.0, np.inf),
    (np.inf, 1.0),
    (np.inf, np.inf),
    (-np.inf, 1.0),
    (-np.inf, -np.inf),
])
def test_binary_logical_ops(op, x1, x2):
    """Test binary logical operations (and, or, xor) against NumPy's behavior"""
    op_func = getattr(np, op)
    
    # QuadPrecision values
    quad_x1 = QuadPrecision(str(x1))
    quad_x2 = QuadPrecision(str(x2))
    quad_result = op_func(quad_x1, quad_x2)
    
    # NumPy float64 values for comparison
    float_x1 = np.float64(x1)
    float_x2 = np.float64(x2)
    float_result = op_func(float_x1, float_x2)
    
    # Results should match NumPy's behavior
    assert quad_result == float_result, f"{op}({x1}, {x2}): quad={quad_result}, float64={float_result}"
    assert isinstance(quad_result, (bool, np.bool_)), f"Result should be bool, got {type(quad_result)}"


@pytest.mark.parametrize("x", [
    # Zeros are falsy
    0.0,
    -0.0,
    # Non-zero values are truthy
    1.0,
    -1.0,
    2.5,
    -3.7,
    0.001,
    # Special values: NaN and inf are truthy
    np.nan,
    np.inf,
    -np.inf,
])
def test_unary_logical_not(x):
    """Test logical_not operation against NumPy's behavior"""
    # QuadPrecision value
    quad_x = QuadPrecision(str(x))
    quad_result = np.logical_not(quad_x)
    
    # NumPy float64 value for comparison
    float_x = np.float64(x)
    float_result = np.logical_not(float_x)
    
    # Results should match NumPy's behavior
    assert quad_result == float_result, f"logical_not({x}): quad={quad_result}, float64={float_result}"
    assert isinstance(quad_result, (bool, np.bool_)), f"Result should be bool, got {type(quad_result)}"


@pytest.mark.parametrize("op", ["amin", "amax", "nanmin", "nanmax"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_array_aminmax(op, a, b):
    op_func = getattr(np, op)
    quad_ab = np.array([QuadPrecision(a), QuadPrecision(b)])
    float_ab = np.array([float(a), float(b)])

    quad_res = op_func(quad_ab)
    float_res = op_func(float_ab)

    # native implementation may not be sensitive to zero signs
    #  but we want to enforce it for the quad dtype
    # e.g. min(+0.0, -0.0) = -0.0
    if float(a) == 0.0 and float(b) == 0.0:
        assert float_res == 0.0
        float_res = np.copysign(0.0, op_func(np.array([np.copysign(1.0, float(a)), np.copysign(1.0, float(b))])))

    np.testing.assert_array_equal(np.array(quad_res).astype(float), float_res)

    # Check sign for zero results
    if float_res == 0.0:
        assert np.signbit(float_res) == np.signbit(
            quad_res), f"Zero sign mismatch for {op}({a}, {b})"


@pytest.mark.parametrize("op", ["negative", "positive", "absolute", "sign", "signbit", "isfinite", "isinf", "isnan", "sqrt", "square", "reciprocal"])
@pytest.mark.parametrize("val", ["3.0", "-3.0", "12.5", "100.0", "1e100", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_unary_ops(op, val):
    op_func = dict(negative=operator.neg, positive=operator.pos, absolute=operator.abs).get(op, None)
    nop_func = getattr(np, op)

    quad_val = QuadPrecision(val)
    float_val = float(val)

    for of in [op_func, nop_func]:
        if of is None:
            continue

        quad_result = of(quad_val)
        float_result = of(float_val)

        np.testing.assert_array_equal(np.array(quad_result).astype(float), float_result)

        if (float_result == 0.0) and (op not in ["signbit", "isfinite", "isinf", "isnan"]):
            assert np.signbit(float_result) == np.signbit(quad_result)


@pytest.mark.parametrize("op", ["floor", "ceil", "trunc", "rint"])
@pytest.mark.parametrize("val", [
    # Basic cases
    "3.2", "-3.2", "3.8", "-3.8", "0.1", "-0.1",
    # Edge cases around integers
    "3.0", "-3.0", "0.0", "-0.0", "1.0", "-1.0",
    # Halfway cases (important for rint)
    "2.5", "-2.5", "3.5", "-3.5", "0.5", "-0.5",
    # Large numbers
    "1e10", "-1e10", "1e15", "-1e15",
    # Small fractional numbers
    "1e-10", "-1e-10", "1e-15", "-1e-15",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_rounding_functions(op, val):
    """Comprehensive test for rounding functions: floor, ceil, trunc, rint"""
    op_func = getattr(np, op)

    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = op_func(quad_val)
    float_result = op_func(float_val)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for {op}({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for {op}({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for {op}({val})"
        return

    # For finite results, check value and sign
    np.testing.assert_allclose(float(quad_result), float_result, rtol=1e-15, atol=1e-15,
                               err_msg=f"Value mismatch for {op}({val})")

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({val})"


def test_rint_near_halfway():
    assert np.rint(QuadPrecision("7.4999999999999999")) == 7
    assert np.rint(QuadPrecision("7.49999999999999999")) == 7
    assert np.rint(QuadPrecision("7.5")) == 8


@pytest.mark.parametrize("val", [
    # Perfect cubes
    "1.0", "8.0", "27.0", "64.0", "125.0", "1000.0",
    # Negative perfect cubes
    "-1.0", "-8.0", "-27.0", "-64.0", "-125.0", "-1000.0",
    # Small positive values
    "0.001", "0.008", "0.027", "1e-9", "1e-15", "1e-100",
    # Small negative values
    "-0.001", "-0.008", "-0.027", "-1e-9", "-1e-15", "-1e-100",
    # Large positive values
    "1e10", "1e15", "1e100", "1e300",
    # Large negative values
    "-1e10", "-1e15", "-1e100", "-1e300",
    # Fractional values
    "0.5", "2.5", "3.5", "10.5", "100.5",
    "-0.5", "-2.5", "-3.5", "-10.5", "-100.5",
    # Edge cases
    "0.0", "-0.0",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_cbrt(val):
    """Comprehensive test for cube root function"""
    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = np.cbrt(quad_val)
    float_result = np.cbrt(float_val)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for cbrt({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for cbrt({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for cbrt({val})"
        return

    # For finite results, check value and sign
    # Use relative tolerance for cbrt
    if float_result != 0.0:
        rtol = 1e-14 if abs(float_result) < 1e100 else 1e-10
        np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=1e-15,
                                   err_msg=f"Value mismatch for cbrt({val})")
    else:
        # For zero results
        assert float(quad_result) == 0.0, f"Expected 0 for cbrt({val}), got {float(quad_result)}"
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for cbrt({val})"


def test_cbrt_accuracy():
    """Test that cbrt gives accurate results for perfect cubes"""
    # Test perfect cubes
    for i in [1, 2, 3, 4, 5, 10, 100]:
        val = QuadPrecision(i ** 3)
        result = np.cbrt(val)
        expected = QuadPrecision(i)
        np.testing.assert_allclose(float(result), float(expected), rtol=1e-14, atol=1e-15,
                                   err_msg=f"cbrt({i}^3) should equal {i}")
    
    # Test negative perfect cubes
    for i in [1, 2, 3, 4, 5, 10, 100]:
        val = QuadPrecision(-(i ** 3))
        result = np.cbrt(val)
        expected = QuadPrecision(-i)
        np.testing.assert_allclose(float(result), float(expected), rtol=1e-14, atol=1e-15,
                                   err_msg=f"cbrt(-{i}^3) should equal -{i}")


@pytest.mark.parametrize("op", ["exp", "exp2"])
@pytest.mark.parametrize("val", [
    # Basic cases
    "0.0", "-0.0", "1.0", "-1.0", "2.0", "-2.0",
    # Small values (should be close to 1)
    "1e-10", "-1e-10", "1e-15", "-1e-15",
    # Medium values
    "10.0", "-10.0", "20.0", "-20.0",
    # Values that might cause overflow
    "100.0", "200.0", "700.0", "1000.0",
    # Values that might cause underflow
    "-100.0", "-200.0", "-700.0", "-1000.0",
    # Fractional values
    "0.5", "-0.5", "1.5", "-1.5", "2.5", "-2.5",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_exponential_functions(op, val):
    """Comprehensive test for exponential functions: exp, exp2"""
    op_func = getattr(np, op)

    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = op_func(quad_val)
    float_result = op_func(float_val)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for {op}({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for {op}({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for {op}({val})"
        return

    # Handle underflow to zero
    if float_result == 0.0:
        assert float(
            quad_result) == 0.0, f"Expected 0 for {op}({val}), got {float(quad_result)}"
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({val})"
        return

    # For finite non-zero results
    # Use relative tolerance for exponential functions due to their rapid growth
    rtol = 1e-14 if abs(float_result) < 1e100 else 1e-10
    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=1e-15,
                               err_msg=f"Value mismatch for {op}({val})")


@pytest.mark.parametrize("op", ["log", "log2", "log10"])
@pytest.mark.parametrize("val", [
    # Basic positive cases
    "1.0", "2.0", "10.0", "100.0", "1000.0",
    # Values close to 1 (important for log accuracy)
    "1.01", "0.99", "1.001", "0.999", "1.0001", "0.9999",
    # Small positive values
    "1e-10", "1e-15", "1e-100", "1e-300",
    # Large positive values
    "1e10", "1e15", "1e100", "1e300",
    # Fractional values
    "0.5", "0.1", "0.01", "2.5", "5.5", "25.0",
    # Edge cases
    "0.0", "-0.0",  # Should give -inf
    # Invalid domain (negative values) - should give NaN
    "-1.0", "-2.0", "-0.5", "-10.0",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_logarithmic_functions(op, val):
    """Comprehensive test for logarithmic functions: log, log2, log10"""
    op_func = getattr(np, op)

    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = op_func(quad_val)
    float_result = op_func(float_val)

    # Handle NaN cases (negative values, NaN input)
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for {op}({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for {op}({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for {op}({val})"
        return

    # For finite results
    # Use higher tolerance for values very close to 1 where log is close to 0
    if abs(float(val) - 1.0) < 1e-10:
        rtol = 1e-10
        atol = 1e-15
    else:
        rtol = 1e-14
        atol = 1e-15

    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=atol,
                               err_msg=f"Value mismatch for {op}({val})")

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch"


@pytest.mark.parametrize("val", [
    # Basic cases around -1 (critical point for log1p)
    "-0.5", "-0.1", "-0.01", "-0.001", "-0.0001",
    # Cases close to 0 (where log1p is most accurate)
    "1e-10", "-1e-10", "1e-15", "-1e-15", "1e-20", "-1e-20",
    # Larger positive values
    "0.1", "0.5", "1.0", "2.0", "10.0", "100.0",
    # Edge case at -1 (should give -inf)
    "-1.0",
    # Invalid domain (< -1) - should give NaN
    "-1.1", "-2.0", "-10.0",
    # Large positive values
    "1e10", "1e15", "1e100",
    # Edge cases
    "0.0", "-0.0",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_log1p(val):
    """Comprehensive test for log1p function"""
    op = "log1p"
    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = np.log1p(quad_val)
    float_result = np.log1p(float_val)

    # Handle NaN cases (values < -1, NaN input)
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for log1p({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for log1p({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for log1p({val})"
        return

    # For finite results
    # log1p is designed for high accuracy near 0, so use tight tolerances
    if abs(float(val)) < 1e-10:
        rtol = 1e-15
        atol = 1e-20
    else:
        rtol = 1e-14
        atol = 1e-15

    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=atol,
                               err_msg=f"Value mismatch for log1p({val})")

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({val})"


@pytest.mark.parametrize("val", [
    # Cases close to 0 (where expm1 is most accurate and important)
    "0.0", "-0.0",
    "1e-10", "-1e-10", "1e-15", "-1e-15", "1e-20", "-1e-20",
    "1e-100", "-1e-100", "1e-300", "-1e-300",
    # Small values
    "0.001", "-0.001", "0.01", "-0.01", "0.1", "-0.1",
    # Moderate values
    "0.5", "-0.5", "1.0", "-1.0", "2.0", "-2.0",
    # Larger values
    "5.0", "-5.0", "10.0", "-10.0", "20.0", "-20.0",
    # Values that test exp behavior
    "50.0", "-50.0", "100.0", "-100.0",
    # Large positive values (exp(x) grows rapidly)
    "200.0", "500.0", "700.0",
    # Large negative values (should approach -1)
    "-200.0", "-500.0", "-700.0", "-1000.0",
    # Special values
    "inf",   # Should give inf
    "-inf",  # Should give -1
    "nan", "-nan"
])
def test_expm1(val):
    """Comprehensive test for expm1 function: exp(x) - 1
    
    This function provides greater precision than exp(x) - 1 for small values of x.
    """
    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = np.expm1(quad_val)
    float_result = np.expm1(float_val)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for expm1({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for expm1({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for expm1({val})"
        return

    # For finite results
    # expm1 is designed for high accuracy near 0, so use tight tolerances for small inputs
    if abs(float(val)) < 1e-10:
        rtol = 1e-15
        atol = 1e-20
    elif abs(float_result) < 1:
        rtol = 1e-14
        atol = 1e-15
    else:
        # For larger results, use relative tolerance
        rtol = 1e-14
        atol = 1e-15

    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=atol,
                               err_msg=f"Value mismatch for expm1({val})")

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for expm1({val})"


@pytest.mark.parametrize("x", [
    # Regular values
    "0.0", "1.0", "2.0", "-1.0", "-2.0", "0.5", "-0.5",
    # Large values (test numerical stability)
    "100.0", "1000.0", "-100.0", "-1000.0",
    # Small values
    "1e-10", "-1e-10", "1e-20", "-1e-20",
    # Special values
    "inf", "-inf", "nan", "-nan", "-0.0"
])
@pytest.mark.parametrize("y", [
    # Regular values
    "0.0", "1.0", "2.0", "-1.0", "-2.0", "0.5", "-0.5",
    # Large values
    "100.0", "1000.0", "-100.0", "-1000.0",
    # Small values
    "1e-10", "-1e-10", "1e-20", "-1e-20",
    # Special values
    "inf", "-inf", "nan", "-nan", "-0.0"
])
def test_logaddexp(x, y):
    """Comprehensive test for logaddexp function: log(exp(x) + exp(y))"""
    quad_x = QuadPrecision(x)
    quad_y = QuadPrecision(y)
    float_x = float(x)
    float_y = float(y)
    
    quad_result = np.logaddexp(quad_x, quad_y)
    float_result = np.logaddexp(float_x, float_y)
    
    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(float(quad_result)), \
            f"Expected NaN for logaddexp({x}, {y}), got {float(quad_result)}"
        return
    
    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(float(quad_result)), \
            f"Expected inf for logaddexp({x}, {y}), got {float(quad_result)}"
        if not np.isnan(float_result):
            assert np.sign(float_result) == np.sign(float(quad_result)), \
                f"Infinity sign mismatch for logaddexp({x}, {y})"
        return
    
    # For finite results, check with appropriate tolerance
    # logaddexp is numerically sensitive, especially for large differences
    if abs(float_x - float_y) > 50:
        # When values differ greatly, result should be close to max(x, y)
        rtol = 1e-10
        atol = 1e-10
    else:
        rtol = 1e-13
        atol = 1e-15
    
    np.testing.assert_allclose(
        float(quad_result), float_result, 
        rtol=rtol, atol=atol,
        err_msg=f"Value mismatch for logaddexp({x}, {y})"
    )


def test_logaddexp_special_properties():
    """Test special mathematical properties of logaddexp"""
    # logaddexp(x, x) = x + log(2)
    x = QuadPrecision("2.0")
    result = np.logaddexp(x, x)
    expected = float(x) + np.log(2.0)
    np.testing.assert_allclose(float(result), expected, rtol=1e-14)
    
    # logaddexp(x, -inf) = x
    x = QuadPrecision("5.0")
    result = np.logaddexp(x, QuadPrecision("-inf"))
    np.testing.assert_allclose(float(result), float(x), rtol=1e-14)
    
    # logaddexp(-inf, x) = x
    result = np.logaddexp(QuadPrecision("-inf"), x)
    np.testing.assert_allclose(float(result), float(x), rtol=1e-14)
    
    # logaddexp(-inf, -inf) = -inf
    result = np.logaddexp(QuadPrecision("-inf"), QuadPrecision("-inf"))
    assert np.isinf(float(result)) and float(result) < 0
    
    # logaddexp(inf, anything) = inf
    result = np.logaddexp(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # logaddexp(anything, inf) = inf
    result = np.logaddexp(QuadPrecision("100.0"), QuadPrecision("inf"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # Commutativity: logaddexp(x, y) = logaddexp(y, x)
    x = QuadPrecision("3.0")
    y = QuadPrecision("5.0")
    result1 = np.logaddexp(x, y)
    result2 = np.logaddexp(y, x)
    np.testing.assert_allclose(float(result1), float(result2), rtol=1e-14)


@pytest.mark.parametrize("x", [
    # Regular values
    "0.0", "1.0", "2.0", "-1.0", "-2.0", "0.5", "-0.5",
    # Large values (test numerical stability)
    "100.0", "1000.0", "-100.0", "-1000.0",
    # Small values
    "1e-10", "-1e-10", "1e-20", "-1e-20",
    # Special values
    "inf", "-inf", "nan", "-nan", "-0.0"
])
@pytest.mark.parametrize("y", [
    # Regular values
    "0.0", "1.0", "2.0", "-1.0", "-2.0", "0.5", "-0.5",
    # Large values
    "100.0", "1000.0", "-100.0", "-1000.0",
    # Small values
    "1e-10", "-1e-10", "1e-20", "-1e-20",
    # Special values
    "inf", "-inf", "nan", "-nan", "-0.0"
])
def test_logaddexp2(x, y):
    """Comprehensive test for logaddexp2 function: log2(2^x + 2^y)"""
    quad_x = QuadPrecision(x)
    quad_y = QuadPrecision(y)
    float_x = float(x)
    float_y = float(y)
    
    quad_result = np.logaddexp2(quad_x, quad_y)
    float_result = np.logaddexp2(float_x, float_y)
    
    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(float(quad_result)), \
            f"Expected NaN for logaddexp2({x}, {y}), got {float(quad_result)}"
        return
    
    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(float(quad_result)), \
            f"Expected inf for logaddexp2({x}, {y}), got {float(quad_result)}"
        if not np.isnan(float_result):
            assert np.sign(float_result) == np.sign(float(quad_result)), \
                f"Infinity sign mismatch for logaddexp2({x}, {y})"
        return
    
    # For finite results, check with appropriate tolerance
    # logaddexp2 is numerically sensitive, especially for large differences
    if abs(float_x - float_y) > 50:
        # When values differ greatly, result should be close to max(x, y)
        rtol = 1e-10
        atol = 1e-10
    else:
        rtol = 1e-13
        atol = 1e-15
    
    np.testing.assert_allclose(
        float(quad_result), float_result, 
        rtol=rtol, atol=atol,
        err_msg=f"Value mismatch for logaddexp2({x}, {y})"
    )


def test_logaddexp2_special_properties():
    """Test special mathematical properties of logaddexp2"""
    # logaddexp2(x, x) = x + 1 (since log2(2^x + 2^x) = log2(2 * 2^x) = log2(2) + log2(2^x) = 1 + x)
    x = QuadPrecision("2.0")
    result = np.logaddexp2(x, x)
    expected = float(x) + 1.0
    np.testing.assert_allclose(float(result), expected, rtol=1e-14)
    
    # logaddexp2(x, -inf) = x
    x = QuadPrecision("5.0")
    result = np.logaddexp2(x, QuadPrecision("-inf"))
    np.testing.assert_allclose(float(result), float(x), rtol=1e-14)
    
    # logaddexp2(-inf, x) = x
    result = np.logaddexp2(QuadPrecision("-inf"), x)
    np.testing.assert_allclose(float(result), float(x), rtol=1e-14)
    
    # logaddexp2(-inf, -inf) = -inf
    result = np.logaddexp2(QuadPrecision("-inf"), QuadPrecision("-inf"))
    assert np.isinf(float(result)) and float(result) < 0
    
    # logaddexp2(inf, anything) = inf
    result = np.logaddexp2(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # logaddexp2(anything, inf) = inf
    result = np.logaddexp2(QuadPrecision("100.0"), QuadPrecision("inf"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # Commutativity: logaddexp2(x, y) = logaddexp2(y, x)
    x = QuadPrecision("3.0")
    y = QuadPrecision("5.0")
    result1 = np.logaddexp2(x, y)
    result2 = np.logaddexp2(y, x)
    np.testing.assert_allclose(float(result1), float(result2), rtol=1e-14)
    
    # Relationship with logaddexp: logaddexp2(x, y) = logaddexp(x*ln2, y*ln2) / ln2
    x = QuadPrecision("2.0")
    y = QuadPrecision("3.0")
    result_logaddexp2 = np.logaddexp2(x, y)
    ln2 = np.log(2.0)
    result_logaddexp = np.logaddexp(float(x) * ln2, float(y) * ln2) / ln2
    np.testing.assert_allclose(float(result_logaddexp2), result_logaddexp, rtol=1e-13)


@pytest.mark.parametrize(
    "x_val",
    [
        0.0, 1.0, 2.0, -1.0, -2.0,
        0.5, -0.5,
        100.0, 1000.0, -100.0, -1000.0,
        1e-10, -1e-10, 1e-20, -1e-20,
        float("inf"), float("-inf"), float("nan"), float("-nan"), -0.0
    ]
)
@pytest.mark.parametrize(
    "y_val",
    [
        0.0, 1.0, 2.0, -1.0, -2.0,
        0.5, -0.5,
        100.0, 1000.0, -100.0, -1000.0,
        1e-10, -1e-10, 1e-20, -1e-20,
        float("inf"), float("-inf"), float("nan"), float("-nan"), -0.0
    ]
)
def test_true_divide(x_val, y_val):
    """Test true_divide ufunc with comprehensive edge cases"""
    x_quad = QuadPrecision(str(x_val))
    y_quad = QuadPrecision(str(y_val))
    
    # Compute using QuadPrecision
    result_quad = np.true_divide(x_quad, y_quad)
    
    # Compute using float64 for comparison
    result_float64 = np.true_divide(np.float64(x_val), np.float64(y_val))
    
    # Compare results
    if np.isnan(result_float64):
        assert np.isnan(float(result_quad)), f"Expected NaN for true_divide({x_val}, {y_val})"
    elif np.isinf(result_float64):
        assert np.isinf(float(result_quad)), f"Expected inf for true_divide({x_val}, {y_val})"
        assert np.sign(float(result_quad)) == np.sign(result_float64), f"Sign mismatch for true_divide({x_val}, {y_val})"
    else:
        # For finite results, check relative tolerance
        np.testing.assert_allclose(
            float(result_quad), result_float64, rtol=1e-14,
            err_msg=f"Mismatch for true_divide({x_val}, {y_val})"
        )


def test_true_divide_special_properties():
    """Test special mathematical properties of true_divide"""
    # Division by 1 returns the original value
    x = QuadPrecision("42.123456789")
    result = np.true_divide(x, QuadPrecision("1.0"))
    np.testing.assert_allclose(float(result), float(x), rtol=1e-30)
    
    # Division of 0 by any non-zero number is 0
    result = np.true_divide(QuadPrecision("0.0"), QuadPrecision("5.0"))
    assert float(result) == 0.0
    
    # Division by 0 gives inf (with appropriate sign)
    result = np.true_divide(QuadPrecision("1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    result = np.true_divide(QuadPrecision("-1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(result)) and float(result) < 0
    
    # 0 / 0 = NaN
    result = np.true_divide(QuadPrecision("0.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    # inf / inf = NaN
    result = np.true_divide(QuadPrecision("inf"), QuadPrecision("inf"))
    assert np.isnan(float(result))
    
    # inf / finite = inf
    result = np.true_divide(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # finite / inf = 0
    result = np.true_divide(QuadPrecision("100.0"), QuadPrecision("inf"))
    assert float(result) == 0.0
    
    # Self-division (x / x) = 1 for finite non-zero x
    x = QuadPrecision("7.123456789")
    result = np.true_divide(x, x)
    np.testing.assert_allclose(float(result), 1.0, rtol=1e-30)
    
    # Sign preservation: (-x) / y = -(x / y)
    x = QuadPrecision("5.5")
    y = QuadPrecision("2.2")
    result1 = np.true_divide(-x, y)
    result2 = -np.true_divide(x, y)
    np.testing.assert_allclose(float(result1), float(result2), rtol=1e-30)
    
    # Sign rule: negative / negative = positive
    result = np.true_divide(QuadPrecision("-6.0"), QuadPrecision("-2.0"))
    assert float(result) > 0
    np.testing.assert_allclose(float(result), 3.0, rtol=1e-30)


@pytest.mark.parametrize(
    "x_val",
    [
        0.0, 1.0, 2.0, -1.0, -2.0,
        0.5, -0.5,
        100.0, 1000.0, -100.0, -1000.0,
        1e-10, -1e-10, 1e-20, -1e-20,
        float("inf"), float("-inf"), float("nan"), float("-nan"), -0.0
    ]
)
@pytest.mark.parametrize(
    "y_val",
    [
        0.0, 1.0, 2.0, -1.0, -2.0,
        0.5, -0.5,
        100.0, 1000.0, -100.0, -1000.0,
        1e-10, -1e-10, 1e-20, -1e-20,
        float("inf"), float("-inf"), float("nan"), float("-nan"), -0.0
    ]
)
def test_floor_divide(x_val, y_val):
    """Test floor_divide ufunc with comprehensive edge cases"""
    x_quad = QuadPrecision(str(x_val))
    y_quad = QuadPrecision(str(y_val))
    
    # Compute using QuadPrecision
    result_quad = np.floor_divide(x_quad, y_quad)
    
    # Compute using float64 for comparison
    result_float64 = np.floor_divide(np.float64(x_val), np.float64(y_val))
    
    # Compare results
    if np.isnan(result_float64):
        assert np.isnan(float(result_quad)), f"Expected NaN for floor_divide({x_val}, {y_val})"
    elif np.isinf(result_float64):
        assert np.isinf(float(result_quad)), f"Expected inf for floor_divide({x_val}, {y_val})"
        assert np.sign(float(result_quad)) == np.sign(result_float64), f"Sign mismatch for floor_divide({x_val}, {y_val})"
    else:
        # For finite results, check relative tolerance
        # Use absolute tolerance for large numbers due to float64 precision limits
        atol = max(1e-10, abs(result_float64) * 1e-9) if abs(result_float64) > 1e6 else 1e-10
        np.testing.assert_allclose(
            float(result_quad), result_float64, rtol=1e-12, atol=atol,
            err_msg=f"Mismatch for floor_divide({x_val}, {y_val})"
        )
def test_floor_divide_special_properties():
    """Test special mathematical properties of floor_divide"""
    # floor_divide(x, 1) = floor(x)
    x = QuadPrecision("42.7")
    result = np.floor_divide(x, QuadPrecision("1.0"))
    np.testing.assert_allclose(float(result), 42.0, rtol=1e-30)
    
    # floor_divide(0, non-zero) = 0
    result = np.floor_divide(QuadPrecision("0.0"), QuadPrecision("5.0"))
    assert float(result) == 0.0
    
    # floor_divide by 0 gives inf (with appropriate sign)
    result = np.floor_divide(QuadPrecision("1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    result = np.floor_divide(QuadPrecision("-1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(result)) and float(result) < 0
    
    # 0 / 0 = NaN
    result = np.floor_divide(QuadPrecision("0.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    # inf / inf = NaN
    result = np.floor_divide(QuadPrecision("inf"), QuadPrecision("inf"))
    assert np.isnan(float(result))
    
    # inf / finite_nonzero = NaN (NumPy behavior)
    result = np.floor_divide(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isnan(float(result))
    
    # finite / inf = 0
    result = np.floor_divide(QuadPrecision("100.0"), QuadPrecision("inf"))
    assert float(result) == 0.0
    
    # floor_divide rounds toward negative infinity
    result = np.floor_divide(QuadPrecision("7.0"), QuadPrecision("3.0"))
    assert float(result) == 2.0  # floor(7/3) = floor(2.333...) = 2
    
    result = np.floor_divide(QuadPrecision("-7.0"), QuadPrecision("3.0"))
    assert float(result) == -3.0  # floor(-7/3) = floor(-2.333...) = -3
    
    result = np.floor_divide(QuadPrecision("7.0"), QuadPrecision("-3.0"))
    assert float(result) == -3.0  # floor(7/-3) = floor(-2.333...) = -3
    
    result = np.floor_divide(QuadPrecision("-7.0"), QuadPrecision("-3.0"))
    assert float(result) == 2.0  # floor(-7/-3) = floor(2.333...) = 2
    
    # floor_divide(x, x) = 1 for positive finite non-zero x
    x = QuadPrecision("7.123456789")
    result = np.floor_divide(x, x)
    np.testing.assert_allclose(float(result), 1.0, rtol=1e-30)
    
    # Relationship with floor and true_divide
    x = QuadPrecision("10.5")
    y = QuadPrecision("3.2")
    result_floor_divide = np.floor_divide(x, y)
    result_floor_true_divide = np.floor(np.true_divide(x, y))
    np.testing.assert_allclose(float(result_floor_divide), float(result_floor_true_divide), rtol=1e-30)


@pytest.mark.parametrize("x_val,y_val", [
    (x, y) for x in [-1e10, -100.0, -7.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 7.0, 100.0, 1e10, 
                      float('inf'), float('-inf'), float('nan'),
                      -6.0, 6.0, -0.1, 0.1, -3.14159, 3.14159]
    for y in [-1e10, -100.0, -3.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 3.0, 100.0, 1e10,
              float('inf'), float('-inf'), float('nan'),
              -2.0, 2.0, -0.25, 0.25, -1.5, 1.5]
])
def test_fmod(x_val, y_val):
    """Test fmod ufunc with comprehensive edge cases"""
    x_quad = QuadPrecision(str(x_val))
    y_quad = QuadPrecision(str(y_val))
    
    # Compute using QuadPrecision
    result_quad = np.fmod(x_quad, y_quad)
    
    # Compute using float64 for comparison
    result_float64 = np.fmod(np.float64(x_val), np.float64(y_val))
    
    # Compare results
    if np.isnan(result_float64):
        assert np.isnan(float(result_quad)), f"Expected NaN for fmod({x_val}, {y_val})"
    elif np.isinf(result_float64):
        assert np.isinf(float(result_quad)), f"Expected inf for fmod({x_val}, {y_val})"
        assert np.sign(float(result_quad)) == np.sign(result_float64), f"Sign mismatch for fmod({x_val}, {y_val})"
    else:
        # For finite results, check relative tolerance
        atol = max(1e-10, abs(result_float64) * 1e-9) if abs(result_float64) > 1e6 else 1e-10
        np.testing.assert_allclose(
            float(result_quad), result_float64, rtol=1e-12, atol=atol,
            err_msg=f"Mismatch for fmod({x_val}, {y_val})"
        )
        
        # Critical: Check sign preservation for zero results
        if result_float64 == 0.0:
            assert np.signbit(result_quad) == np.signbit(result_float64), \
                f"Sign mismatch for zero result: fmod({x_val}, {y_val}), " \
                f"expected signbit={np.signbit(result_float64)}, got signbit={np.signbit(result_quad)}"


def test_fmod_special_properties():
    """Test special mathematical properties of fmod"""
    # fmod(x, 1) gives fractional part of x (with sign preserved)
    x = QuadPrecision("42.7")
    result = np.fmod(x, QuadPrecision("1.0"))
    np.testing.assert_allclose(float(result), 0.7, rtol=1e-15, atol=1e-15)
    
    # fmod(0, non-zero) = 0 with correct sign
    result = np.fmod(QuadPrecision("0.0"), QuadPrecision("5.0"))
    assert float(result) == 0.0 and not np.signbit(result)
    
    result = np.fmod(QuadPrecision("-0.0"), QuadPrecision("5.0"))
    assert float(result) == 0.0 and np.signbit(result)
    
    # fmod by 0 gives NaN
    result = np.fmod(QuadPrecision("1.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    result = np.fmod(QuadPrecision("-1.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    # 0 fmod 0 = NaN
    result = np.fmod(QuadPrecision("0.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    # inf fmod x = NaN
    result = np.fmod(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isnan(float(result))
    
    result = np.fmod(QuadPrecision("-inf"), QuadPrecision("100.0"))
    assert np.isnan(float(result))
    
    # x fmod inf = x (for finite x)
    result = np.fmod(QuadPrecision("100.0"), QuadPrecision("inf"))
    np.testing.assert_allclose(float(result), 100.0, rtol=1e-30)
    
    result = np.fmod(QuadPrecision("-100.0"), QuadPrecision("inf"))
    np.testing.assert_allclose(float(result), -100.0, rtol=1e-30)
    
    # inf fmod inf = NaN
    result = np.fmod(QuadPrecision("inf"), QuadPrecision("inf"))
    assert np.isnan(float(result))
    
    # fmod uses truncated division (rounds toward zero)
    # Result has same sign as dividend (first argument)
    result = np.fmod(QuadPrecision("7.0"), QuadPrecision("3.0"))
    assert float(result) == 1.0  # 7 - trunc(7/3)*3 = 7 - 2*3 = 1
    
    result = np.fmod(QuadPrecision("-7.0"), QuadPrecision("3.0"))
    assert float(result) == -1.0  # -7 - trunc(-7/3)*3 = -7 - (-2)*3 = -1
    
    result = np.fmod(QuadPrecision("7.0"), QuadPrecision("-3.0"))
    assert float(result) == 1.0  # 7 - trunc(7/-3)*(-3) = 7 - (-2)*(-3) = 1
    
    result = np.fmod(QuadPrecision("-7.0"), QuadPrecision("-3.0"))
    assert float(result) == -1.0  # -7 - trunc(-7/-3)*(-3) = -7 - 2*(-3) = -1
    
    # Sign preservation when result is exactly zero
    result = np.fmod(QuadPrecision("6.0"), QuadPrecision("3.0"))
    assert float(result) == 0.0 and not np.signbit(result)
    
    result = np.fmod(QuadPrecision("-6.0"), QuadPrecision("3.0"))
    assert float(result) == 0.0 and np.signbit(result)
    
    result = np.fmod(QuadPrecision("6.0"), QuadPrecision("-3.0"))
    assert float(result) == 0.0 and not np.signbit(result)
    
    result = np.fmod(QuadPrecision("-6.0"), QuadPrecision("-3.0"))
    assert float(result) == 0.0 and np.signbit(result)
    
    # Difference from mod/remainder (which uses floor division)
    # fmod result has sign of dividend, mod result has sign of divisor
    x = QuadPrecision("-7.0")
    y = QuadPrecision("3.0")
    fmod_result = np.fmod(x, y)
    mod_result = np.remainder(x, y)
    
    assert float(fmod_result) == -1.0  # sign of dividend (negative)
    assert float(mod_result) == 2.0    # sign of divisor (positive)
    
    # Relationship: x = trunc(x/y) * y + fmod(x, y)
    x = QuadPrecision("10.5")
    y = QuadPrecision("3.2")
    quotient = np.trunc(np.true_divide(x, y))
    remainder = np.fmod(x, y)
    reconstructed = np.add(np.multiply(quotient, y), remainder)
    np.testing.assert_allclose(float(reconstructed), float(x), rtol=1e-30)


def test_inf():
    assert QuadPrecision("inf") > QuadPrecision("1e1000")
    assert np.signbit(QuadPrecision("inf")) == 0
    assert QuadPrecision("-inf") < QuadPrecision("-1e1000")
    assert np.signbit(QuadPrecision("-inf")) == 1


def test_dtype_creation():
    dtype = QuadPrecDType()
    assert isinstance(dtype, np.dtype)
    assert dtype.name == "QuadPrecDType128"


def test_array_creation():
    arr = np.array([1, 2, 3], dtype=QuadPrecDType())
    assert arr.dtype.name == "QuadPrecDType128"
    assert all(isinstance(x, QuadPrecision) for x in arr)


def test_array_operations():
    arr1 = np.array(
        [QuadPrecision("1.5"), QuadPrecision("2.5"), QuadPrecision("3.5")])
    arr2 = np.array(
        [QuadPrecision("0.5"), QuadPrecision("1.0"), QuadPrecision("1.5")])

    result = arr1 + arr2
    expected = np.array(
        [QuadPrecision("2.0"), QuadPrecision("3.5"), QuadPrecision("5.0")])
    assert all(x == y for x, y in zip(result, expected))


@pytest.mark.parametrize("backend", ["sleef", "longdouble"])
@pytest.mark.parametrize("op", [np.mod, np.remainder])
@pytest.mark.parametrize("a,b", [
    # Basic cases - positive/negative combinations
    (7.0, 3.0), (-7.0, 3.0), (7.0, -3.0), (-7.0, -3.0),

    # Zero dividend cases
    (0.0, 3.0), (-0.0, 3.0), (0.0, -3.0), (-0.0, -3.0),

    # Cases that result in zero (sign testing)
    (6.0, 3.0), (-6.0, 3.0), (6.0, -3.0), (-6.0, -3.0),
    (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),

    # Fractional cases
    (7.5, 2.5), (-7.5, 2.5), (7.5, -2.5), (-7.5, -2.5),
    (0.75, 0.25), (-0.1, 0.3), (0.9, -1.0), (-1.1, -1.0),

    # Large/small numbers
    (1e10, 1e5), (-1e10, 1e5), (1e-10, 1e-5), (-1e-10, 1e-5),

    # Finite % infinity cases
    (5.0, float('inf')), (-5.0, float('inf')),
    (5.0, float('-inf')), (-5.0, float('-inf')),
    (0.0, float('inf')), (-0.0, float('-inf')),

    # NaN cases (should return NaN)
    (float('nan'), 3.0), (3.0, float('nan')), (float('nan'), float('nan')),

    # Division by zero cases (should return NaN)
    (5.0, 0.0), (-5.0, 0.0), (0.0, 0.0), (-0.0, 0.0),

    # Infinity dividend cases (should return NaN)
    (float('inf'), 3.0), (float('-inf'), 3.0),
    (float('inf'), float('inf')), (float('-inf'), float('-inf')),
])
def test_mod(a, b, backend, op):
    """Comprehensive test for mod operation against NumPy behavior"""
    if backend == "sleef":
        quad_a = QuadPrecision(str(a))
        quad_b = QuadPrecision(str(b))
    elif backend == "longdouble":
        quad_a = QuadPrecision(a, backend='longdouble')
        quad_b = QuadPrecision(b, backend='longdouble')
    float_a = np.float64(a)
    float_b = np.float64(b)

    quad_result = op(quad_a, quad_b)
    numpy_result = op(float_a, float_b)

    # Handle NaN cases
    if np.isnan(numpy_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for {a} % {b}, got {float(quad_result)}"
        return

    if np.isinf(numpy_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for {a} % {b}, got {float(quad_result)}"
        assert np.sign(numpy_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for {a} % {b}"
        return

    np.testing.assert_allclose(float(quad_result), numpy_result, rtol=1e-10, atol=1e-15,
                               err_msg=f"Value mismatch for {a} % {b}")

    if numpy_result == 0.0:
        numpy_sign = np.signbit(numpy_result)
        quad_sign = np.signbit(quad_result)
        assert numpy_sign == quad_sign, f"Zero sign mismatch for {a} % {b}: numpy={numpy_sign}, quad={quad_sign}"

    # Check that non-zero results have correct sign relative to divisor
    if numpy_result != 0.0 and not np.isnan(b) and not np.isinf(b) and b != 0.0:
        # In Python mod, non-zero result should have same sign as divisor (or be zero)
        result_negative = float(quad_result) < 0
        divisor_negative = b < 0
        numpy_negative = numpy_result < 0

        assert result_negative == numpy_negative, f"Sign mismatch for {a} % {b}: quad={result_negative}, numpy={numpy_negative}"


@pytest.mark.parametrize("backend", ["sleef", "longdouble"])
@pytest.mark.parametrize("a,b", [
    # Basic cases - positive/positive
    (7.0, 3.0), (10.5, 3.2), (21.0, 4.0),
    
    # Positive/negative combinations
    (-7.0, 3.0), (7.0, -3.0), (-7.0, -3.0),
    (-10.5, 3.2), (10.5, -3.2), (-10.5, -3.2),

    # Zero dividend cases
    (0.0, 3.0), (-0.0, 3.0), (0.0, -3.0), (-0.0, -3.0),

    # Cases that result in zero remainder (exact division)
    (6.0, 3.0), (-6.0, 3.0), (6.0, -3.0), (-6.0, -3.0),
    (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
    (10.0, 2.0), (-10.0, 2.0), (10.0, -2.0), (-10.0, -2.0),

    # Fractional cases
    (7.5, 2.5), (-7.5, 2.5), (7.5, -2.5), (-7.5, -2.5),
    (0.75, 0.25), (-0.1, 0.3), (0.9, -1.0), (-1.1, -1.0),
    (3.14159, 1.0), (-3.14159, 1.0), (3.14159, -1.0), (-3.14159, -1.0),

    # Large/small numbers
    (1e10, 1e5), (-1e10, 1e5), (1e-10, 1e-5), (-1e-10, 1e-5),
    (1e15, 1e10), (1e-15, 1e-10),

    # Finite % infinity cases
    (5.0, float('inf')), (-5.0, float('inf')),
    (5.0, float('-inf')), (-5.0, float('-inf')),
    (0.0, float('inf')), (-0.0, float('-inf')),

    # NaN cases (should return NaN for both quotient and remainder)
    (float('nan'), 3.0), (3.0, float('nan')), (float('nan'), float('nan')),

    # Division by zero cases (should return inf/NaN)
    (5.0, 0.0), (-5.0, 0.0), (0.0, 0.0), (-0.0, 0.0),

    # Infinity dividend cases (should return NaN for both)
    (float('inf'), 3.0), (float('-inf'), 3.0),
    (float('inf'), float('inf')), (float('-inf'), float('-inf')),
    
    # Cases with dividend < divisor
    (1.0, 10.0), (-1.0, 10.0), (1.0, -10.0), (-1.0, -10.0),
    (0.5, 1.0), (0.1, 1.0), (0.001, 0.01),
])
def test_divmod(a, b, backend):
    """Comprehensive test for divmod operation against NumPy behavior"""
    if backend == "sleef":
        quad_a = QuadPrecision(str(a))
        quad_b = QuadPrecision(str(b))
    elif backend == "longdouble":
        quad_a = QuadPrecision(a, backend='longdouble')
        quad_b = QuadPrecision(b, backend='longdouble')
    
    float_a = np.float64(a)
    float_b = np.float64(b)

    # Compute divmod
    quad_quotient, quad_remainder = np.divmod(quad_a, quad_b)
    numpy_quotient, numpy_remainder = np.divmod(float_a, float_b)

    # Verify quotient
    if np.isnan(numpy_quotient):
        assert np.isnan(float(quad_quotient)), \
            f"Expected NaN quotient for divmod({a}, {b})"
    elif np.isinf(numpy_quotient):
        assert np.isinf(float(quad_quotient)) and \
               np.sign(numpy_quotient) == np.sign(float(quad_quotient)), \
            f"Expected inf quotient with matching sign for divmod({a}, {b})"
    else:
        # Adaptive tolerance for large quotients due to float64 conversion precision loss
        atol_q = abs(numpy_quotient) * 1e-8 if abs(numpy_quotient) > 1e6 else 1e-15
        np.testing.assert_allclose(
            float(quad_quotient), numpy_quotient, rtol=1e-9, atol=atol_q,
            err_msg=f"Quotient mismatch for divmod({a}, {b})"
        )
        if numpy_quotient == 0.0:
            assert np.signbit(numpy_quotient) == np.signbit(quad_quotient), \
                f"Zero quotient sign mismatch for divmod({a}, {b})"

    # Verify remainder
    if np.isnan(numpy_remainder):
        assert np.isnan(float(quad_remainder)), \
            f"Expected NaN remainder for divmod({a}, {b})"
    elif np.isinf(numpy_remainder):
        assert np.isinf(float(quad_remainder)) and \
               np.sign(numpy_remainder) == np.sign(float(quad_remainder)), \
            f"Expected inf remainder with matching sign for divmod({a}, {b})"
    else:
        # Standard tolerance for remainder comparison
        np.testing.assert_allclose(
            float(quad_remainder), numpy_remainder, rtol=1e-9, atol=1e-15,
            err_msg=f"Remainder mismatch for divmod({a}, {b})"
        )
        if numpy_remainder == 0.0:
            assert np.signbit(numpy_remainder) == np.signbit(quad_remainder), \
                f"Zero remainder sign mismatch for divmod({a}, {b})"
        elif not np.isnan(b) and not np.isinf(b) and b != 0.0:
            assert (float(quad_remainder) < 0) == (numpy_remainder < 0), \
                f"Remainder sign mismatch for divmod({a}, {b})"

    # Verify the fundamental property: a = quotient * b + remainder (for finite values)
    if not np.isnan(numpy_quotient) and not np.isinf(numpy_quotient) and \
       not np.isnan(numpy_remainder) and not np.isinf(numpy_remainder) and \
       not np.isnan(b) and not np.isinf(b) and b != 0.0:
        reconstructed = float(quad_quotient) * float(quad_b) + float(quad_remainder)
        np.testing.assert_allclose(
            reconstructed, float(quad_a), rtol=1e-10, atol=1e-15,
            err_msg=f"Property a = q*b + r failed for divmod({a}, {b})"
        )


def test_divmod_special_properties():
    """Test special mathematical properties of divmod"""
    # divmod(x, 1) should give (floor(x), 0)
    x = QuadPrecision("42.7")
    quotient, remainder = np.divmod(x, QuadPrecision("1.0"))
    np.testing.assert_allclose(float(quotient), 42.0, rtol=1e-30)
    np.testing.assert_allclose(float(remainder), 0.7, rtol=1e-14)
    
    # divmod(0, non-zero) should give (0, 0)
    quotient, remainder = np.divmod(QuadPrecision("0.0"), QuadPrecision("5.0"))
    assert float(quotient) == 0.0
    assert float(remainder) == 0.0
    
    # divmod by 0 gives (inf, NaN) for positive dividend
    quotient, remainder = np.divmod(QuadPrecision("1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(quotient)) and float(quotient) > 0
    assert np.isnan(float(remainder))
    
    quotient, remainder = np.divmod(QuadPrecision("-1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(quotient)) and float(quotient) < 0
    assert np.isnan(float(remainder))
    
    # divmod(inf, finite) gives (NaN, NaN)
    quotient, remainder = np.divmod(QuadPrecision("inf"), QuadPrecision("5.0"))
    assert np.isnan(float(quotient))
    assert np.isnan(float(remainder))
    
    # divmod(finite, inf) gives (0, dividend)
    quotient, remainder = np.divmod(QuadPrecision("5.0"), QuadPrecision("inf"))
    np.testing.assert_allclose(float(quotient), 0.0, rtol=1e-30)
    np.testing.assert_allclose(float(remainder), 5.0, rtol=1e-30)
    
    # Verify equivalence with floor_divide and mod
    a = QuadPrecision("10.5")
    b = QuadPrecision("3.2")
    quotient, remainder = np.divmod(a, b)
    expected_quotient = np.floor_divide(a, b)
    expected_remainder = np.mod(a, b)
    np.testing.assert_allclose(float(quotient), float(expected_quotient), rtol=1e-30)
    np.testing.assert_allclose(float(remainder), float(expected_remainder), rtol=1e-30)


def test_divmod_array():
    """Test divmod with arrays"""
    a = np.array([10.5, 21.0, -7.5, 0.0], dtype=QuadPrecDType())
    b = np.array([3.2, 4.0, 2.5, 5.0], dtype=QuadPrecDType())
    
    quotients, remainders = np.divmod(a, b)
    
    # Check dtype
    assert quotients.dtype.name == "QuadPrecDType128"
    assert remainders.dtype.name == "QuadPrecDType128"
    
    # Check against NumPy float64
    a_float = np.array([10.5, 21.0, -7.5, 0.0], dtype=np.float64)
    b_float = np.array([3.2, 4.0, 2.5, 5.0], dtype=np.float64)
    expected_quotients, expected_remainders = np.divmod(a_float, b_float)
    
    for i in range(len(a)):
        np.testing.assert_allclose(
            float(quotients[i]), expected_quotients[i], rtol=1e-10, atol=1e-15,
            err_msg=f"Quotient mismatch at index {i}"
        )
        np.testing.assert_allclose(
            float(remainders[i]), expected_remainders[i], rtol=1e-10, atol=1e-15,
            err_msg=f"Remainder mismatch at index {i}"
        )


def test_divmod_broadcasting():
    """Test divmod with broadcasting"""
    # Scalar with array
    a = np.array([10.5, 21.0, 31.5], dtype=QuadPrecDType())
    b = QuadPrecision("3.0")
    
    quotients, remainders = np.divmod(a, b)
    
    assert quotients.dtype.name == "QuadPrecDType128"
    assert remainders.dtype.name == "QuadPrecDType128"
    assert len(quotients) == 3
    assert len(remainders) == 3
    
    # Check values
    expected_quotients = [3.0, 7.0, 10.0]
    expected_remainders = [1.5, 0.0, 1.5]
    
    for i in range(3):
        np.testing.assert_allclose(float(quotients[i]), expected_quotients[i], rtol=1e-14)
        np.testing.assert_allclose(float(remainders[i]), expected_remainders[i], rtol=1e-14)

class TestTrignometricFunctions:
  @pytest.mark.parametrize("op", ["sin", "cos", "tan", "atan"])
  @pytest.mark.parametrize("val", [
      # Basic cases
      "0.0", "-0.0", "1.0", "-1.0", "2.0", "-2.0",
      # pi multiples
      str(quad_pi), str(-quad_pi), str(2*quad_pi), str(-2*quad_pi), str(quad_pi/2), str(-quad_pi/2), str(3*quad_pi/2), str(-3*quad_pi/2),
      # Small values
      "1e-10", "-1e-10", "1e-15", "-1e-15",
      # Values near one
      "0.9", "-0.9", "0.9999", "-0.9999",
      "1.1", "-1.1", "1.0001", "-1.0001",
      # Medium values
      "10.0", "-10.0", "20.0", "-20.0",
      # Large values
      "100.0", "200.0", "700.0", "1000.0", "1e100", "1e308",
      "-100.0", "-200.0", "-700.0", "-1000.0", "-1e100", "-1e308",
      # Fractional values
      "0.5", "-0.5", "1.5", "-1.5", "2.5", "-2.5",
      # Special values
      "inf", "-inf", "nan",
  ])
  def test_sin_cos_tan(self, op, val):
    mp.prec = 113  # Set precision to 113 bits (~34 decimal digits)
    numpy_op = getattr(np, op)
    mpmath_op = getattr(mp, op)
    
    quad_val = QuadPrecision(val)
    mpf_val = mp.mpf(val)

    quad_result = numpy_op(quad_val)
    mpmath_result = mpmath_op(mpf_val)
    # convert mpmath result to quad for comparison
    # Use mp.nstr to get full precision (40 digits for quad precision)
    mpmath_result = QuadPrecision(mp.nstr(mpmath_result, 40))

    # Handle NaN cases
    if np.isnan(mpmath_result):
        assert np.isnan(quad_result), f"Expected NaN for {op}({val}), got {quad_result}"
        return

    # Handle infinity cases
    if np.isinf(mpmath_result):
        assert np.isinf(quad_result), f"Expected inf for {op}({val}), got {quad_result}"
        assert np.sign(mpmath_result) == np.sign(quad_result), f"Infinity sign mismatch for {op}({val})"
        return

    # For finite non-zero results
    np.testing.assert_allclose(quad_result, mpmath_result, rtol=1e-32, atol=1e-34,
                              err_msg=f"Value mismatch for {op}({val}), expected {mpmath_result}, got {quad_result}")

  # their domain is [-1 , 1]
  @pytest.mark.parametrize("op", ["asin", "acos"])
  @pytest.mark.parametrize("val", [
    # Basic cases (valid domain)
    "0.0", "-0.0", "1.0", "-1.0",
    # Small values
    "1e-10", "-1e-10", "1e-15", "-1e-15",
    # Values near domain boundaries
    "0.9", "-0.9", "0.9999", "-0.9999",
    "0.99999999", "-0.99999999",
    "0.999999999999", "-0.999999999999",
    # Fractional values (within domain)
    "0.5", "-0.5",
    # Special values
    "nan"
  ])
  def test_inverse_sin_cos(self, op, val):
    mp.prec = 113  # Set precision to 113 bits (~34 decimal digits)
    numpy_op = getattr(np, op)
    mpmath_op = getattr(mp, op)
    
    quad_val = QuadPrecision(val)
    mpf_val = mp.mpf(val)

    quad_result = numpy_op(quad_val)
    mpmath_result = mpmath_op(mpf_val)
    # convert mpmath result to quad for comparison
    # Use mp.nstr to get full precision (40 digits for quad precision)
    mpmath_result = QuadPrecision(mp.nstr(mpmath_result, 40))

    # Handle NaN cases
    if np.isnan(mpmath_result):
        assert np.isnan(quad_result), f"Expected NaN for {op}({val}), got {quad_result}"
        return

    # For finite non-zero results
    np.testing.assert_allclose(quad_result, mpmath_result, rtol=1e-32, atol=1e-34,
                              err_msg=f"Value mismatch for {op}({val}), expected {mpmath_result}, got {quad_result}")
  
  # mpmath's atan2 does not follow IEEE standards so hardcoding the edge cases
  # for special edge cases check reference here: https://en.cppreference.com/w/cpp/numeric/math/atan2.html
  # atan2: [Real x Real] -> [-pi , pi]
  @pytest.mark.parametrize("y", [
    # Basic cases
    "0.0", "-0.0", "1.0", "-1.0", 
    # Small values
    "1e-10", "-1e-10", "1e-15", "-1e-15",
    # Medium/Large values
    "10.0", "-10.0", "100.0", "-100.0", "1000.0", "-1000.0",
    # Fractional
    "0.5", "-0.5", "2.5", "-2.5",
    # Special
    "inf", "-inf", "nan",
  ])
  @pytest.mark.parametrize("x", [
      "0.0", "-0.0", "1.0", "-1.0",
      "1e-10", "-1e-10",
      "10.0", "-10.0", "100.0", "-100.0",
      "0.5", "-0.5",
      "inf", "-inf", "nan",
  ])
  def test_atan2(self, y, x):
    mp.prec = 113
    
    quad_y = QuadPrecision(y)
    quad_x = QuadPrecision(x)
    mpf_y = mp.mpf(y)
    mpf_x = mp.mpf(x)

    quad_result = np.arctan2(quad_y, quad_x)
    
    # IEEE 754 special cases - hardcoded expectations
    y_val = float(y)
    x_val = float(x)
    
    # If either x is NaN or y is NaN, NaN is returned
    if np.isnan(y_val) or np.isnan(x_val):
        assert np.isnan(quad_result), f"Expected NaN for atan2({y}, {x}), got {quad_result}"
        return
    
    # If y is ±0 and x is negative or -0, ±π is returned
    if y_val == 0.0 and (x_val < 0.0 or (x_val == 0.0 and np.signbit(x_val))):
        expected = quad_pi if not np.signbit(y_val) else -quad_pi
        np.testing.assert_allclose(quad_result, expected, rtol=1e-32, atol=1e-34,
                                  err_msg=f"Value mismatch for atan2({y}, {x}), expected {expected}, got {quad_result}")
        return
    
    # If y is ±0 and x is positive or +0, ±0 is returned
    if y_val == 0.0 and (x_val > 0.0 or (x_val == 0.0 and not np.signbit(x_val))):
        assert quad_result == 0.0, f"Expected ±0 for atan2({y}, {x}), got {quad_result}"
        assert np.signbit(quad_result) == np.signbit(y_val), f"Sign mismatch for atan2({y}, {x})"
        return
    
    # If y is ±∞ and x is finite, ±π/2 is returned
    if np.isinf(y_val) and np.isfinite(x_val):
        expected = quad_pi / 2 if y_val > 0 else -quad_pi / 2
        np.testing.assert_allclose(quad_result, expected, rtol=1e-32, atol=1e-34,
                                  err_msg=f"Value mismatch for atan2({y}, {x}), expected {expected}, got {quad_result}")
        return
    
    # If y is ±∞ and x is -∞, ±3π/4 is returned
    if np.isinf(y_val) and np.isinf(x_val) and x_val < 0:
        expected = 3 * quad_pi / 4 if y_val > 0 else -3 * quad_pi / 4
        np.testing.assert_allclose(quad_result, expected, rtol=1e-32, atol=1e-34,
                                  err_msg=f"Value mismatch for atan2({y}, {x}), expected {expected}, got {quad_result}")
        return
    
    # If y is ±∞ and x is +∞, ±π/4 is returned
    if np.isinf(y_val) and np.isinf(x_val) and x_val > 0:
        expected = quad_pi / 4 if y_val > 0 else -quad_pi / 4
        np.testing.assert_allclose(quad_result, expected, rtol=1e-32, atol=1e-34,
                                  err_msg=f"Value mismatch for atan2({y}, {x}), expected {expected}, got {quad_result}")
        return
    
    # If x is ±0 and y is negative, -π/2 is returned
    if x_val == 0.0 and y_val < 0.0:
        expected = -quad_pi / 2
        np.testing.assert_allclose(quad_result, expected, rtol=1e-32, atol=1e-34,
                                  err_msg=f"Value mismatch for atan2({y}, {x}), expected {expected}, got {quad_result}")
        return
    
    # If x is ±0 and y is positive, +π/2 is returned
    if x_val == 0.0 and y_val > 0.0:
        expected = quad_pi / 2
        np.testing.assert_allclose(quad_result, expected, rtol=1e-32, atol=1e-34,
                                  err_msg=f"Value mismatch for atan2({y}, {x}), expected {expected}, got {quad_result}")
        return
    
    # If x is -∞ and y is finite and positive, +π is returned
    if np.isinf(x_val) and x_val < 0 and np.isfinite(y_val) and y_val > 0.0:
        expected = quad_pi
        np.testing.assert_allclose(quad_result, expected, rtol=1e-32, atol=1e-34,
                                  err_msg=f"Value mismatch for atan2({y}, {x}), expected {expected}, got {quad_result}")
        return
    
    # If x is -∞ and y is finite and negative, -π is returned
    if np.isinf(x_val) and x_val < 0 and np.isfinite(y_val) and y_val < 0.0:
        expected = -quad_pi
        np.testing.assert_allclose(quad_result, expected, rtol=1e-32, atol=1e-34,
                                  err_msg=f"Value mismatch for atan2({y}, {x}), expected {expected}, got {quad_result}")
        return
    
    # If x is +∞ and y is finite and positive, +0 is returned
    if np.isinf(x_val) and x_val > 0 and np.isfinite(y_val) and y_val > 0.0:
        assert quad_result == 0.0 and not np.signbit(quad_result), f"Expected +0 for atan2({y}, {x}), got {quad_result}"
        return
    
    # If x is +∞ and y is finite and negative, -0 is returned
    if np.isinf(x_val) and x_val > 0 and np.isfinite(y_val) and y_val < 0.0:
        assert quad_result == 0.0 and np.signbit(quad_result), f"Expected -0 for atan2({y}, {x}), got {quad_result}"
        return
    
    # For all other cases, compare with mpmath
    mpmath_result = mp.atan2(mpf_y, mpf_x)
    # Use mp.nstr to get full precision (40 digits for quad precision)
    mpmath_result = QuadPrecision(mp.nstr(mpmath_result, 40))

    if np.isnan(mpmath_result):
        assert np.isnan(quad_result), f"Expected NaN for atan2({y}, {x}), got {quad_result}"
        return

    if np.isinf(mpmath_result):
        assert np.isinf(quad_result), f"Expected inf for atan2({y}, {x}), got {quad_result}"
        assert np.sign(mpmath_result) == np.sign(quad_result), f"Infinity sign mismatch for atan2({y}, {x})"
        return

    np.testing.assert_allclose(quad_result, mpmath_result, rtol=1e-32, atol=1e-34,
                              err_msg=f"Value mismatch for atan2({y}, {x}), expected {mpmath_result}, got {quad_result}")

@pytest.mark.parametrize("op", ["sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh"])
@pytest.mark.parametrize("val", [
    # Basic cases
    "0.0", "-0.0", "1.0", "-1.0", "2.0", "-2.0",
    # Small values
    "1e-10", "-1e-10", "1e-15", "-1e-15",
    # Values near one
    "0.9", "-0.9", "0.9999", "-0.9999",
    "1.1", "-1.1", "1.0001", "-1.0001",
    # Medium values
    "10.0", "-10.0", "20.0", "-20.0",
    # Large values
    "100.0", "200.0", "700.0", "1000.0", "1e100", "1e308",
    "-100.0", "-200.0", "-700.0", "-1000.0", "-1e100", "-1e308",
    # Fractional values
    "0.5", "-0.5", "1.5", "-1.5", "2.5", "-2.5",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_hyperbolic_functions(op, val):
    """Comprehensive test for hyperbolic functions: sinh, cosh, tanh, arcsinh, arccosh, arctanh"""
    op_func = getattr(np, op)

    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = op_func(quad_val)
    float_result = op_func(float_val)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for {op}({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for {op}({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for {op}({val})"
        return

    # For finite non-zero results
    # Use relative tolerance for exponential functions due to their rapid growth
    rtol = 1e-13 if abs(float_result) < 1e100 else 1e-10
    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=1e-15,
                               err_msg=f"Value mismatch for {op}({val})")

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({val})"


class TestTypePomotionWithPythonAbstractTypes:
    """Tests for common_dtype handling of Python abstract dtypes (PyLongDType, PyFloatDType)"""
    
    def test_promotion_with_python_int(self):
        """Test that Python int promotes to QuadPrecDType"""
        # Create array from Python int
        arr = np.array([1, 2, 3], dtype=QuadPrecDType)
        assert arr.dtype.name == "QuadPrecDType128"
        assert len(arr) == 3
        assert float(arr[0]) == 1.0
        assert float(arr[1]) == 2.0
        assert float(arr[2]) == 3.0
    
    def test_promotion_with_python_float(self):
        """Test that Python float promotes to QuadPrecDType"""
        # Create array from Python float
        arr = np.array([1.5, 2.7, 3.14], dtype=QuadPrecDType)
        assert arr.dtype.name == "QuadPrecDType128"
        assert len(arr) == 3
        np.testing.assert_allclose(float(arr[0]), 1.5, rtol=1e-15)
        np.testing.assert_allclose(float(arr[1]), 2.7, rtol=1e-15)
        np.testing.assert_allclose(float(arr[2]), 3.14, rtol=1e-15)
    
    def test_result_dtype_binary_ops_with_python_types(self):
        """Test that binary operations between QuadPrecDType and Python scalars return QuadPrecDType"""
        quad_arr = np.array([QuadPrecision("1.0"), QuadPrecision("2.0")])
        
        # Addition with Python int
        result = quad_arr + 5
        assert result.dtype.name == "QuadPrecDType128"
        assert float(result[0]) == 6.0
        assert float(result[1]) == 7.0
        
        # Multiplication with Python float
        result = quad_arr * 2.5
        assert result.dtype.name == "QuadPrecDType128"
        np.testing.assert_allclose(float(result[0]), 2.5, rtol=1e-15)
        np.testing.assert_allclose(float(result[1]), 5.0, rtol=1e-15)
    
    def test_concatenate_with_python_types(self):
        """Test concatenation handles Python numeric types correctly"""
        quad_arr = np.array([QuadPrecision("1.0")])
        # This should work if promotion is correct
        int_arr = np.array([2], dtype=np.int64)
        
        # The result dtype should be QuadPrecDType
        result = np.concatenate([quad_arr, int_arr.astype(QuadPrecDType)])
        assert result.dtype.name == "QuadPrecDType128"
        assert len(result) == 2


@pytest.mark.parametrize("func,args,expected", [
    # arange tests
    (np.arange, (0, 10), list(range(10))),
    (np.arange, (0, 10, 2), [0, 2, 4, 6, 8]),
    (np.arange, (0.0, 5.0, 0.5), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]),
    (np.arange, (10, 0, -1), [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
    (np.arange, (-5, 5), list(range(-5, 5))),
    # linspace tests
    (np.linspace, (0, 10, 11), list(range(11))),
    (np.linspace, (0, 1, 5), [0.0, 0.25, 0.5, 0.75, 1.0]),
])
def test_fill_function(func, args, expected):
    """Test quadprec_fill function with arange and linspace"""
    arr = func(*args, dtype=QuadPrecDType())
    assert arr.dtype.name == "QuadPrecDType128"
    assert len(arr) == len(expected)
    for i, exp_val in enumerate(expected):
        np.testing.assert_allclose(float(arr[i]), float(exp_val), rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize("base,exponent", [
    # Basic integer powers
    (2.0, 3.0), (3.0, 2.0), (10.0, 5.0), (5.0, 10.0),
    
    # Fractional powers
    (4.0, 0.5), (9.0, 0.5), (27.0, 1.0/3.0), (16.0, 0.25),
    (8.0, 2.0/3.0), (100.0, 0.5),
    
    # Negative bases with integer exponents
    (-2.0, 3.0), (-3.0, 2.0), (-2.0, 4.0), (-5.0, 3.0),
    
    # Negative bases with fractional exponents (should return NaN)
    (-1.0, 0.5), (-4.0, 0.5), (-1.0, 1.5), (-4.0, 1.5),
    (-2.0, 0.25), (-8.0, 1.0/3.0), (-5.0, 2.5), (-10.0, 0.75),
    (-1.0, -0.5), (-4.0, -1.5), (-2.0, -2.5),
    
    # Zero base cases
    (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 10.0),
    (0.0, 0.5), (0.0, -0.0),
    
    # Negative zero base
    (-0.0, 0.0), (-0.0, 1.0), (-0.0, 2.0), (-0.0, 3.0),
    
    # Base of 1
    (1.0, 0.0), (1.0, 1.0), (1.0, 100.0), (1.0, -100.0),
    (1.0, float('inf')), (1.0, float('-inf')), (1.0, float('nan')),
    
    # Base of -1
    (-1.0, 0.0), (-1.0, 1.0), (-1.0, 2.0), (-1.0, 3.0),
    (-1.0, float('inf')), (-1.0, float('-inf')),
    
    # Exponent of 0
    (2.0, 0.0), (100.0, 0.0), (-5.0, 0.0), (0.5, 0.0),
    (float('inf'), 0.0), (float('-inf'), 0.0), (float('nan'), 0.0),
    
    # Exponent of 1
    (2.0, 1.0), (100.0, 1.0), (-5.0, 1.0), (0.5, 1.0),
    (float('inf'), 1.0), (float('-inf'), 1.0),
    
    # Negative exponents
    (2.0, -1.0), (2.0, -2.0), (10.0, -3.0), (0.5, -1.0),
    (4.0, -0.5), (9.0, -0.5),
    
    # Infinity base
    (float('inf'), 0.0), (float('inf'), 1.0), (float('inf'), 2.0),
    (float('inf'), -1.0), (float('inf'), -2.0), (float('inf'), 0.5),
    (float('inf'), float('inf')), (float('inf'), float('-inf')),
    
    # Negative infinity base
    (float('-inf'), 0.0), (float('-inf'), 1.0), (float('-inf'), 2.0),
    (float('-inf'), 3.0), (float('-inf'), -1.0), (float('-inf'), -2.0),
    (float('-inf'), float('inf')), (float('-inf'), float('-inf')),
    
    # Infinity exponent
    (2.0, float('inf')), (0.5, float('inf')), (1.5, float('inf')),
    (2.0, float('-inf')), (0.5, float('-inf')), (1.5, float('-inf')),
    (0.0, float('inf')), (0.0, float('-inf')),
    
    # NaN cases
    (float('nan'), 0.0), (float('nan'), 1.0), (float('nan'), 2.0),
    (2.0, float('nan')), (0.0, float('nan')),
    (float('nan'), float('nan')), (float('nan'), float('inf')),
    (float('inf'), float('nan')),
    
    # Small and large values
    (1e-10, 2.0), (1e10, 2.0), (1e-10, 0.5), (1e10, 0.5),
    (2.0, 100.0), (2.0, -100.0), (0.5, 100.0), (0.5, -100.0),
])
def test_float_power(base, exponent):
    """
    Comprehensive test for float_power ufunc.
    
    float_power differs from power in that it always promotes to floating point.
    For floating-point dtypes like QuadPrecDType, it should behave identically to power.
    """
    quad_base = QuadPrecision(str(base)) if not (np.isnan(base) or np.isinf(base)) else QuadPrecision(base)
    quad_exp = QuadPrecision(str(exponent)) if not (np.isnan(exponent) or np.isinf(exponent)) else QuadPrecision(exponent)

    float_base = np.float64(base)
    float_exp = np.float64(exponent)

    quad_result = np.float_power(quad_base, quad_exp)
    float_result = np.float_power(float_base, float_exp)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(float(quad_result)), \
            f"Expected NaN for float_power({base}, {exponent}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(float(quad_result)), \
            f"Expected inf for float_power({base}, {exponent}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(float(quad_result)), \
            f"Infinity sign mismatch for float_power({base}, {exponent})"
        return

    # For finite results
    np.testing.assert_allclose(
        float(quad_result), float_result, 
        rtol=1e-13, atol=1e-15,
        err_msg=f"Value mismatch for float_power({base}, {exponent})"
    )

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(quad_result), \
            f"Zero sign mismatch for float_power({base}, {exponent})"


@pytest.mark.parametrize("base,exponent", [
    # Test that float_power works with integer inputs (promotes to float)
    (2, 3),
    (4, 2),
    (10, 5),
    (-2, 3),
])
def test_float_power_integer_promotion(base, exponent):
    """
    Test that float_power works with integer inputs and promotes them to QuadPrecDType.
    This is the key difference from power - float_power always returns float types.
    """
    # Create arrays with integer inputs
    base_arr = np.array([base], dtype=QuadPrecDType())
    exp_arr = np.array([exponent], dtype=QuadPrecDType())

    result = np.float_power(base_arr, exp_arr)

    # Result should be QuadPrecDType
    assert result.dtype.name == "QuadPrecDType128"

    # Check the value
    expected = float(base) ** float(exponent)
    np.testing.assert_allclose(float(result[0]), expected, rtol=1e-13)


def test_float_power_array():
    """Test float_power with arrays"""
    bases = np.array([2.0, 4.0, 9.0, 16.0], dtype=QuadPrecDType())
    exponents = np.array([3.0, 0.5, 2.0, 0.25], dtype=QuadPrecDType())

    result = np.float_power(bases, exponents)
    expected = np.array([8.0, 2.0, 81.0, 2.0], dtype=np.float64)

    assert result.dtype.name == "QuadPrecDType128"
    for i in range(len(result)):
        np.testing.assert_allclose(float(result[i]), expected[i], rtol=1e-13)


@pytest.mark.parametrize("val", [
    # Positive values
    "3.0", "12.5", "100.0", "1e100", "0.0",
    # Negative values
    "-3.0", "-12.5", "-100.0", "-1e100", "-0.0",
    # Special values
    "inf", "-inf", "nan", "-nan",
    # Small values
    "1e-100", "-1e-100"
])
def test_fabs(val):
    """
    Test np.fabs ufunc for QuadPrecision dtype.
    fabs computes absolute values (positive magnitude) for floating-point numbers.
    It should behave identically to np.absolute for real (non-complex) types.
    """
    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = np.fabs(quad_val)
    float_result = np.fabs(float_val)

    # Test with both scalar and array
    quad_arr = np.array([quad_val], dtype=QuadPrecDType())
    quad_arr_result = np.fabs(quad_arr)

    # Check scalar result
    np.testing.assert_array_equal(np.array(quad_result).astype(float), float_result)

    # Check array result
    np.testing.assert_array_equal(quad_arr_result.astype(float)[0], float_result)

    # For zero results, check sign (should always be positive after fabs)
    if float_result == 0.0:
        assert not np.signbit(quad_result), f"fabs({val}) should not have negative sign"
        assert not np.signbit(quad_arr_result[0]), f"fabs({val}) should not have negative sign"


@pytest.mark.parametrize("x1,x2", [
    # Basic cases: x1 < 0 -> 0
    ("-1.0", "0.5"), ("-5.0", "0.5"), ("-100.0", "0.5"),
    ("-1e10", "0.5"), ("-0.1", "0.5"),
    
    # Basic cases: x1 == 0 -> x2
    ("0.0", "0.5"), ("0.0", "0.0"), ("0.0", "1.0"),
    ("-0.0", "0.5"), ("-0.0", "0.0"), ("-0.0", "1.0"),
    
    # Basic cases: x1 > 0 -> 1
    ("1.0", "0.5"), ("5.0", "0.5"), ("100.0", "0.5"),
    ("1e10", "0.5"), ("0.1", "0.5"),
    
    # Edge cases with different x2 values
    ("0.0", "-1.0"), ("0.0", "2.0"), ("0.0", "100.0"),
    
    # Special values: infinity
    ("inf", "0.5"), ("-inf", "0.5"),
    ("inf", "0.0"), ("-inf", "0.0"),
    
    # Special values: NaN (should propagate)
    ("nan", "0.5"), ("0.5", "nan"), ("nan", "nan"),
    ("-nan", "0.5"), ("0.5", "-nan"),
    
    # Edge case: zero x1 with special x2
    ("0.0", "inf"), ("0.0", "-inf"), ("0.0", "nan"),
    ("-0.0", "inf"), ("-0.0", "-inf"), ("-0.0", "nan"),
])
def test_heaviside(x1, x2):
    """
    Test np.heaviside ufunc for QuadPrecision dtype.
    
    heaviside(x1, x2) = 0 if x1 < 0
                        x2 if x1 == 0
                        1 if x1 > 0
    
    This is the Heaviside step function where x2 determines the value at x1=0.
    """
    quad_x1 = QuadPrecision(x1)
    quad_x2 = QuadPrecision(x2)
    float_x1 = float(x1)
    float_x2 = float(x2)

    # Test scalar inputs
    quad_result = np.heaviside(quad_x1, quad_x2)
    float_result = np.heaviside(float_x1, float_x2)

    # Test array inputs
    quad_arr_x1 = np.array([quad_x1], dtype=QuadPrecDType())
    quad_arr_x2 = np.array([quad_x2], dtype=QuadPrecDType())
    quad_arr_result = np.heaviside(quad_arr_x1, quad_arr_x2)

    # Check results match
    np.testing.assert_array_equal(
        np.array(quad_result).astype(float), 
        float_result,
        err_msg=f"Scalar heaviside({x1}, {x2}) mismatch"
    )
    
    np.testing.assert_array_equal(
        quad_arr_result.astype(float)[0], 
        float_result,
        err_msg=f"Array heaviside({x1}, {x2}) mismatch"
    )

    # Additional checks for non-NaN results
    if not np.isnan(float_result):
        # Verify the expected value based on x1
        if float_x1 < 0:
            assert float(quad_result) == 0.0, f"Expected 0 for heaviside({x1}, {x2})"
        elif float_x1 == 0.0:
            np.testing.assert_array_equal(
                float(quad_result), float_x2,
                err_msg=f"Expected {x2} for heaviside(0, {x2})"
            )
        else:  # float_x1 > 0
            assert float(quad_result) == 1.0, f"Expected 1 for heaviside({x1}, {x2})"


def test_heaviside_broadcast():
    """Test that heaviside works with broadcasting"""
    x1 = np.array([-1.0, 0.0, 1.0], dtype=QuadPrecDType())
    x2 = QuadPrecision("0.5")
    
    result = np.heaviside(x1, x2)
    expected = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    
    assert result.dtype.name == "QuadPrecDType128"
    np.testing.assert_array_equal(result.astype(float), expected)

    # Test with array for both arguments
    x1_arr = np.array([-2.0, -0.0, 0.0, 5.0], dtype=QuadPrecDType())
    x2_arr = np.array([0.5, 0.5, 1.0, 0.5], dtype=QuadPrecDType())
    
    result = np.heaviside(x1_arr, x2_arr)
    expected = np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float64)
    
    assert result.dtype.name == "QuadPrecDType128"
    np.testing.assert_array_equal(result.astype(float), expected)


@pytest.mark.parametrize("func", [np.conj, np.conjugate])
@pytest.mark.parametrize("value", [
    0.0,
    -0.0,
    1.5,
    -1.5,
    np.inf,
    -np.inf,
    np.nan,
])
def test_conj_conjugate_identity(func, value):
    """Test that conj and conjugate are identity (no-op) for real quad precision numbers"""
    x = QuadPrecision(value)
    result = func(x)
    
    # For NaN, use special comparison
    if np.isnan(value):
        assert np.isnan(float(result))
    else:
        assert result == x


@pytest.mark.parametrize("x1,x2,expected", [
    # Basic Pythagorean triples
    (3.0, 4.0, 5.0),
    (5.0, 12.0, 13.0),
    # Zero cases
    (0.0, 0.0, 0.0),
    (0.0, 5.0, 5.0),
    (5.0, 0.0, 5.0),
    # Negative values (hypot uses absolute values)
    (-3.0, -4.0, 5.0),
    (-3.0, 4.0, 5.0),
    (3.0, -4.0, 5.0),
    # Symmetry
    (3.14159265358979323846, 2.71828182845904523536, None),  # Will test symmetry
    (2.71828182845904523536, 3.14159265358979323846, None),  # Will test symmetry
    # Infinity cases
    (np.inf, 0.0, np.inf),
    (0.0, np.inf, np.inf),
    (np.inf, np.inf, np.inf),
    (-np.inf, 0.0, np.inf),
    (np.inf, -np.inf, np.inf),
    # NaN cases
    (np.nan, 3.0, np.nan),
    (3.0, np.nan, np.nan),
    (np.nan, np.nan, np.nan),
])
def test_hypot(x1, x2, expected):
    """Test hypot ufunc with various edge cases"""
    q1 = QuadPrecision(x1)
    q2 = QuadPrecision(x2)
    result = np.hypot(q1, q2)
    
    assert isinstance(result, QuadPrecision)
    
    if expected is None:
        # Symmetry test - just check the values are equal
        result_reverse = np.hypot(q2, q1)
        assert result == result_reverse
    elif np.isnan(expected):
        assert np.isnan(float(result))
    elif np.isinf(expected):
        assert np.isinf(float(result))
    else:
        np.testing.assert_allclose(float(result), expected, rtol=1e-13)


@pytest.mark.parametrize("op", [np.degrees, np.rad2deg])
@pytest.mark.parametrize("radians,expected_degrees", [
    # Basic conversions
    (0.0, 0.0),
    (np.pi / 6, 30.0),
    (np.pi / 4, 45.0),
    (np.pi / 3, 60.0),
    (np.pi / 2, 90.0),
    (np.pi, 180.0),
    (3 * np.pi / 2, 270.0),
    (2 * np.pi, 360.0),
    # Negative values
    (-np.pi / 2, -90.0),
    (-np.pi, -180.0),
    # Special values
    (np.inf, np.inf),
    (-np.inf, -np.inf),
    (np.nan, np.nan),
    # Edge cases
    (-0.0, -0.0),
])
def test_degrees_rad2deg(op, radians, expected_degrees):
    """Test degrees and rad2deg ufuncs convert radians to degrees"""
    q_rad = QuadPrecision(radians)
    result = op(q_rad)

    assert isinstance(result, QuadPrecision)

    if np.isnan(expected_degrees):
        assert np.isnan(float(result))
    elif np.isinf(expected_degrees):
        assert np.isinf(float(result))
        if expected_degrees > 0:
            assert float(result) > 0
        else:
            assert float(result) < 0
    else:
        np.testing.assert_allclose(float(result), expected_degrees, rtol=1e-13)


@pytest.mark.parametrize("op", [np.radians, np.deg2rad])
@pytest.mark.parametrize("degrees,expected_radians", [
    # Basic conversions
    (0.0, 0.0),
    (30.0, np.pi / 6),
    (45.0, np.pi / 4),
    (60.0, np.pi / 3),
    (90.0, np.pi / 2),
    (180.0, np.pi),
    (270.0, 3 * np.pi / 2),
    (360.0, 2 * np.pi),
    # Negative values
    (-90.0, -np.pi / 2),
    (-180.0, -np.pi),
    # Special values
    (np.inf, np.inf),
    (-np.inf, -np.inf),
    (np.nan, np.nan),
    # Edge cases
    (0.0, 0.0),
    (-0.0, -0.0),
])
def test_radians(op, degrees, expected_radians):
    """Test radians and deg2rad ufuncs convert degrees to radians"""
    q_deg = QuadPrecision(degrees)
    result = op(q_deg)

    assert isinstance(result, QuadPrecision)

    if np.isnan(expected_radians):
        assert np.isnan(float(result))
    elif np.isinf(expected_radians):
        assert np.isinf(float(result))
        if expected_radians > 0:
            assert float(result) > 0
        else:
            assert float(result) < 0
    else:
        np.testing.assert_allclose(float(result), expected_radians, rtol=1e-13)


class TestNextAfter:
    """Test cases for np.nextafter function with QuadPrecision dtype"""
    
    @pytest.mark.parametrize("x1,x2", [
        # NaN tests
        (np.nan, 1.0),
        (1.0, np.nan),
        (np.nan, np.nan),
    ])
    def test_nan(self, x1, x2):
        """Test nextafter with NaN inputs returns NaN"""
        q_x1 = QuadPrecision(x1)
        q_x2 = QuadPrecision(x2)
        
        result = np.nextafter(q_x1, q_x2)
        
        assert isinstance(result, QuadPrecision)
        assert np.isnan(float(result))

    def test_precision(self):
        """Test that nextafter provides the exact next representable value"""
        # Start with 1.0 and move towards 2.0
        x1 = QuadPrecision(1.0)
        x2 = QuadPrecision(2.0)
        
        result = np.nextafter(x1, x2)
        
        # Get machine epsilon from finfo
        finfo = np.finfo(QuadPrecDType())
        expected = x1 + finfo.eps
        
        # result should be exactly 1.0 + eps
        assert result == expected
        
        # Moving the other direction should give us back 1.0
        result_back = np.nextafter(result, x1)
        assert result_back == x1

    def test_smallest_subnormal(self):
        """Test that nextafter(0.0, 1.0) returns the smallest positive subnormal (TRUE_MIN)"""
        zero = QuadPrecision(0.0)
        one = QuadPrecision(1.0)

        result = np.nextafter(zero, one)  # smallest_subnormal
        finfo = np.finfo(QuadPrecDType())
        
        assert result == finfo.smallest_subnormal, \
            f"nextafter(0.0, 1.0) should equal smallest_subnormal, got {result} vs {finfo.smallest_subnormal}"
        
        # Verify it's positive and very small
        assert result > zero, "nextafter(0.0, 1.0) should be positive"
        
        # Moving back towards zero should give us zero
        result_back = np.nextafter(result, zero)
        assert result_back == zero, f"nextafter(smallest_subnormal, 0.0) should be 0.0, got {result_back}"

    def test_negative_zero(self):
        """Test nextafter with negative zero"""
        neg_zero = QuadPrecision(-0.0)
        pos_zero = QuadPrecision(0.0)
        one = QuadPrecision(1.0)
        neg_one = QuadPrecision(-1.0)
        
        finfo = np.finfo(QuadPrecDType())
        
        # nextafter(-0.0, 1.0) should return smallest positive subnormal
        result = np.nextafter(neg_zero, one)
        assert result == finfo.smallest_subnormal, \
            f"nextafter(-0.0, 1.0) should be smallest_subnormal, got {result}"
        assert result > pos_zero, "Result should be positive"
        
        # nextafter(+0.0, -1.0) should return smallest negative subnormal
        result_neg = np.nextafter(pos_zero, neg_one)
        expected_neg_subnormal = -finfo.smallest_subnormal
        assert result_neg == expected_neg_subnormal, \
            f"nextafter(+0.0, -1.0) should be -smallest_subnormal, got {result_neg}"
        assert result_neg < pos_zero, "Result should be negative"

    def test_infinity_cases(self):
        """Test nextafter with infinity edge cases"""
        pos_inf = QuadPrecision(np.inf)
        neg_inf = QuadPrecision(-np.inf)
        one = QuadPrecision(1.0)
        neg_one = QuadPrecision(-1.0)
        zero = QuadPrecision(0.0)
        
        finfo = np.finfo(QuadPrecDType())
        
        # nextafter(+inf, finite) should return max finite value
        result = np.nextafter(pos_inf, zero)
        assert not np.isinf(result), "nextafter(+inf, 0) should be finite"
        assert result < pos_inf, "Result should be less than +inf"
        assert result == finfo.max, f"nextafter(+inf, 0) should be max, got {result} vs {finfo.max}"
        
        # nextafter(-inf, finite) should return -max (most negative finite)
        result_neg = np.nextafter(neg_inf, zero)
        assert not np.isinf(result_neg), "nextafter(-inf, 0) should be finite"
        assert result_neg > neg_inf, "Result should be greater than -inf"
        assert result_neg == -finfo.max, f"nextafter(-inf, 0) should be -max, got {result_neg}"
        
        # Verify symmetry: nextafter(result, +inf) should give us +inf back
        back_to_inf = np.nextafter(result, pos_inf)
        assert back_to_inf == pos_inf, "nextafter(max_finite, +inf) should be +inf"
        
        # nextafter(+inf, +inf) should return +inf
        result_inf = np.nextafter(pos_inf, pos_inf)
        assert result_inf == pos_inf, "nextafter(+inf, +inf) should be +inf"
        
        # nextafter(-inf, -inf) should return -inf
        result_neg_inf = np.nextafter(neg_inf, neg_inf)
        assert result_neg_inf == neg_inf, "nextafter(-inf, -inf) should be -inf"

    def test_max_to_infinity(self):
        """Test nextafter from max finite value to infinity"""
        finfo = np.finfo(QuadPrecDType())
        max_val = finfo.max
        pos_inf = QuadPrecision(np.inf)
        neg_inf = QuadPrecision(-np.inf)
        
        # nextafter(max_finite, +inf) should return +inf
        result = np.nextafter(max_val, pos_inf)
        assert np.isinf(result), f"nextafter(max, +inf) should be inf, got {result}"
        assert result > max_val, "Result should be greater than max"
        assert result == pos_inf, "Result should be +inf"
        
        # nextafter(-max_finite, -inf) should return -inf
        neg_max_val = -max_val
        result_neg = np.nextafter(neg_max_val, neg_inf)
        assert np.isinf(result_neg), f"nextafter(-max, -inf) should be -inf, got {result_neg}"
        assert result_neg < neg_max_val, "Result should be less than -max"
        assert result_neg == neg_inf, "Result should be -inf"

    def test_near_max(self):
        """Test nextafter near maximum finite value"""
        finfo = np.finfo(QuadPrecDType())
        max_val = finfo.max
        zero = QuadPrecision(0.0)
        pos_inf = QuadPrecision(np.inf)
        
        # nextafter(max, 0) should return a value less than max
        result = np.nextafter(max_val, zero)
        assert result < max_val, "nextafter(max, 0) should be less than max"
        assert not np.isinf(result), "Result should be finite"
        
        # The difference should be one ULP at that scale
        # Moving back should give us max again
        result_back = np.nextafter(result, pos_inf)
        assert result_back == max_val, f"Moving back should return max, got {result_back}"

    def test_symmetry(self):
        """Test symmetry properties of nextafter"""
        values = [0.0, 1.0, -1.0, 1e10, -1e10, 1e-10, -1e-10]
        
        for val in values:
            q_val = QuadPrecision(val)
            
            # nextafter(x, +direction) then nextafter(result, x) should return x
            if not np.isinf(val):
                result_up = np.nextafter(q_val, QuadPrecision(np.inf))
                result_back = np.nextafter(result_up, q_val)
                assert result_back == q_val, \
                    f"Symmetry failed for {val}: nextafter then back should return original"
                
                # Same for down direction
                result_down = np.nextafter(q_val, QuadPrecision(-np.inf))
                result_back_down = np.nextafter(result_down, q_val)
                assert result_back_down == q_val, \
                    f"Symmetry failed for {val}: nextafter down then back should return original"

    def test_direction(self):
        """Test that nextafter moves in the correct direction"""
        test_cases = [
            (1.0, 2.0, "greater"),    # towards larger
            (2.0, 1.0, "less"),       # towards smaller
            (-1.0, -2.0, "less"),     # towards more negative
            (-2.0, -1.0, "greater"),  # towards less negative
            (1.0, np.inf, "greater"), # towards +inf
            (1.0, -np.inf, "less"),   # towards -inf
        ]
        
        for x, y, expected_dir in test_cases:
            q_x = QuadPrecision(x)
            q_y = QuadPrecision(y)
            result = np.nextafter(q_x, q_y)
            
            if expected_dir == "greater":
                assert result > q_x, f"nextafter({x}, {y}) should be > {x}, got {result}"
            elif expected_dir == "less":
                assert result < q_x, f"nextafter({x}, {y}) should be < {x}, got {result}"

class TestSpacing:
    """Test cases for np.spacing function with QuadPrecision dtype"""
    
    @pytest.mark.parametrize("x", [
        np.nan, -np.nan,
        np.inf, -np.inf,
    ])
    def test_special_values_return_nan(self, x):
        """Test spacing with NaN and infinity inputs returns NaN"""
        q_x = QuadPrecision(x)
        result = np.spacing(q_x)
        
        assert isinstance(result, QuadPrecision)
        assert np.isnan(result), f"spacing({x}) should be NaN, got {result}"
    
    @pytest.mark.parametrize("x", [
        1.0, -1.0,
        10.0, -10.0,
        100.0, -100.0,
    ])
    def test_sign_preservation(self, x):
        """Test that spacing preserves the sign of the input"""
        q_x = QuadPrecision(x)
        result = np.spacing(q_x)
        
        q_zero = QuadPrecision(0)
        # spacing should have the same sign as x
        if x > 0:
            assert result > q_zero, f"spacing({x}) should be positive, got {result}"
        else:
            assert result < q_zero, f"spacing({x}) should be negative, got {result}"
        
        # Compare with numpy behavior
        np_result = np.spacing(np.float64(x))
        assert np.signbit(result) == np.signbit(np_result), \
            f"Sign mismatch for spacing({x}): quad signbit={np.signbit(result)}, numpy signbit={np.signbit(np_result)}"
    
    @pytest.mark.parametrize("x", [
        0.0,
        -0.0,
    ])
    def test_zero(self, x):
        """Test spacing of zero returns smallest_subnormal"""
        q_x = QuadPrecision(x)
        result = np.spacing(q_x)
        
        finfo = np.finfo(QuadPrecDType())
        q_zero = QuadPrecision(0)
        
        # spacing(±0.0) should return smallest_subnormal (positive)
        assert result == finfo.smallest_subnormal, \
            f"spacing({x}) should be smallest_subnormal, got {result}"
        assert result > q_zero, f"spacing({x}) should be positive, got {result}"
    
    @pytest.mark.parametrize("x", [
        1.0,
        -1.0,
    ])
    def test_one_and_negative_one(self, x):
        """Test spacing(±1.0) equals ±machine epsilon"""
        q_x = QuadPrecision(x)
        result = np.spacing(q_x)
        
        finfo = np.finfo(QuadPrecDType())
        q_zero = QuadPrecision(0)
        
        # For binary floating point, spacing(±1.0) = ±eps
        expected = finfo.eps if x > 0 else -finfo.eps
        assert result == expected, \
            f"spacing({x}) should equal {expected}, got {result}"
        
        if x > 0:
            assert result > q_zero, "spacing(1.0) should be positive"
        else:
            assert result < q_zero, "spacing(-1.0) should be negative"
    
    @pytest.mark.parametrize("x", [
        1.0, -1.0,
        2.0, -2.0,
        10.0, -10.0,
        100.0, -100.0,
        1e10, -1e10,
        1e-10, -1e-10,
        0.5, -0.5,
        0.25, -0.25,
    ])
    def test_spacing_is_non_zero(self, x):
        """Test that spacing always has positive magnitude"""
        q_x = QuadPrecision(x)
        result = np.spacing(q_x)
        
        q_zero = QuadPrecision(0)
        # The absolute value should be positive
        abs_result = np.abs(result)
        assert abs_result > q_zero, f"|spacing({x})| should be positive, got {abs_result}"
    
    def test_smallest_subnormal(self):
        """Test spacing at smallest subnormal value"""
        finfo = np.finfo(QuadPrecDType())
        smallest = finfo.smallest_subnormal
        
        result = np.spacing(smallest)
        
        q_zero = QuadPrecision(0)
        # spacing(smallest_subnormal) should be smallest_subnormal itself
        # (it's the minimum spacing in the subnormal range)
        assert result == smallest, \
            f"spacing(smallest_subnormal) should be smallest_subnormal, got {result}"
        assert result > q_zero, "Result should be positive"
    
    @pytest.mark.parametrize("x", [
        1.5, -1.5,
        3.7, -3.7,
        42.0, -42.0,
        1e8, -1e8,
    ])
    def test_finite_values(self, x):
        """Test spacing on various finite values"""
        q_x = QuadPrecision(x)
        result = np.spacing(q_x)
        
        q_zero = QuadPrecision(0)
        # Result should be finite
        assert np.isfinite(result), \
            f"spacing({x}) should be finite, got {result}"
        
        # Result should be non-zero
        assert result != q_zero, \
            f"spacing({x}) should be non-zero, got {result}"
        
        # Result should have same sign as input
        assert np.signbit(result) == np.signbit(q_x), \
            f"spacing({x}) should have same sign as {x}"
    
    def test_array_spacing(self):
        """Test spacing on an array of QuadPrecision values"""
        values = [1.0, -1.0, 2.0, -2.0, 0.0, 10.0, -10.0]
        q_array = np.array([QuadPrecision(v) for v in values])
        
        result = np.spacing(q_array)
        
        q_zero = QuadPrecision(0)
        # Check each result
        for i, val in enumerate(values):
            q_val = QuadPrecision(val)
            if val != 0:
                assert np.signbit(result[i]) == np.signbit(q_val), \
                    f"Sign mismatch for spacing({val})"
            else:
                assert result[i] > q_zero, \
                    f"spacing(0) should be positive, got {result[i]}"
    
    @pytest.mark.parametrize("x", [
        1e-100, -1e-100,
        1e-200, -1e-200,
    ])
    def test_subnormal_range(self, x):
        """Test spacing in subnormal range"""
        q_x = QuadPrecision(x)
        result = np.spacing(q_x)
        
        finfo = np.finfo(QuadPrecDType())
        
        # In subnormal range, spacing should be smallest_subnormal
        # or at least very small
        assert np.abs(result) >= finfo.smallest_subnormal, \
            f"spacing({x}) should be >= smallest_subnormal"
        
        q_zero = QuadPrecision(0)
        # Sign should match (but for very small subnormals, spacing might underflow to zero)
        if result != q_zero:
            assert np.signbit(result) == np.signbit(q_x), \
                f"spacing({x}) should have same sign as {x}"
            
    def test_smallest_normal_spacing(self):
        """Test spacing for smallest normal value and 2*smallest normal"""
        finfo = np.finfo(QuadPrecDType())
        smallest_normal = finfo.smallest_normal
        
        # Test spacing at smallest normal value
        result1 = np.spacing(smallest_normal)
        
        # Test spacing at 2 * smallest normal value
        two_smallest_normal = 2 * smallest_normal
        result2 = np.spacing(two_smallest_normal)
        
        q_zero = QuadPrecision("0")
        
        # spacing(smallest_normal) should be smallest_subnormal
        # (the spacing at the boundary between subnormal and normal range)
        expected1 = finfo.smallest_subnormal
        assert result1 == expected1, \
            f"spacing(smallest_normal) should be {expected1}, got {result1}"
        assert result1 > q_zero, "Result should be positive"
        
        # The scaling relationship: spacing(2*x) = 2*spacing(x) for normal numbers
        expected2 = 2 * finfo.smallest_subnormal
        assert result2 == expected2, \
            f"spacing(2*smallest_normal) should be {expected2}, got {result2}"
        assert result2 > q_zero, "Result should be positive"


class TestModf:
    """Test cases for np.modf function with QuadPrecision dtype"""
    
    @pytest.mark.parametrize("x", [
        # Basic positive/negative numbers
        "3.14159", "-3.14159", "2.71828", "-2.71828", "1.5", "-1.5", "0.75", "-0.75",
        # Integers (fractional part should be zero)
        "0.0", "-0.0", "1.0", "-1.0", "5.0", "-5.0", "42.0", "-42.0",
        # Small numbers
        "0.001", "-0.001", "0.000123", "-0.000123",
        # Large numbers  
        "1e10", "-1e10", "1e15", "-1e15",
        # Numbers close to integers
        "0.999999999999", "-0.999999999999", "1.000000000001", "-1.000000000001",
        # Special values
        "inf", "-inf", "nan",
        # Edge cases for sign consistency
        "5.7", "-5.7", "0.3", "-0.3"
    ])
    def test_modf(self, x):
        """Test modf against NumPy's behavior"""
        quad_x = QuadPrecision(x)
        
        # Compute modf for both QuadPrecision and float64
        quad_frac, quad_int = np.modf(quad_x)
        
        # Create numpy float64 for reference
        try:
            float_x = np.float64(x)
            np_frac, np_int = np.modf(float_x)
        except (ValueError, OverflowError):
            # Handle cases where string can't convert to float64 (like "nan")
            float_x = np.float64(float(x))
            np_frac, np_int = np.modf(float_x)
        
        # Check return types
        assert isinstance(quad_frac, QuadPrecision), f"Fractional part should be QuadPrecision for {x}"
        assert isinstance(quad_int, QuadPrecision), f"Integral part should be QuadPrecision for {x}"
        
        # Direct comparison with NumPy results
        if np.isnan(np_frac):
            assert np.isnan(quad_frac), f"Expected NaN fractional part for modf({x})"
        else:
            np.testing.assert_allclose(
                quad_frac, np_frac, rtol=1e-12, atol=1e-15,
                err_msg=f"Fractional part mismatch for modf({x})"
            )
            
        if np.isnan(np_int):
            assert np.isnan(quad_int), f"Expected NaN integral part for modf({x})"  
        elif np.isinf(np_int):
            assert np.isinf(quad_int), f"Expected inf integral part for modf({x})"
            assert np.signbit(quad_int) == np.signbit(np_int), f"Sign mismatch for inf integral part modf({x})"
        else:
            np.testing.assert_allclose(
                quad_int, np_int, rtol=1e-12, atol=0,
                err_msg=f"Integral part mismatch for modf({x})"
            )
        
        # Check sign preservation for zero values
        if np_frac == 0.0:
            assert np.signbit(quad_frac) == np.signbit(np_frac), f"Zero fractional sign mismatch for modf({x})"
        if np_int == 0.0:
            assert np.signbit(quad_int) == np.signbit(np_int), f"Zero integral sign mismatch for modf({x})"
        
        # Verify reconstruction property for finite values
        if np.isfinite(float_x) and not np.isnan(float_x):
            reconstructed = quad_int + quad_frac
            np.testing.assert_allclose(
                reconstructed, quad_x, rtol=1e-12, atol=1e-15,
                err_msg=f"Reconstruction failed for modf({x}): {quad_int} + {quad_frac} != {quad_x}"
            )


class TestLdexp:
    """Tests for ldexp function (x * 2**exp)"""
    
    @pytest.mark.parametrize("x_val,exp_val", [
        ("1.0", 0),
        ("1.0", 1),
        ("1.0", 2),
        ("1.0", 10),
        ("1.0", -1),
        ("1.0", -2),
        ("1.0", -10),
        ("2.5", 3),
        ("0.5", 5),
        ("-3.0", 4),
        ("-0.75", -2),
        ("1.5", 100),
        ("1.5", -100),
    ])
    def test_ldexp_basic(self, x_val, exp_val):
        """Test ldexp with basic values"""
        quad_x = QuadPrecision(x_val)
        float_x = np.float64(x_val)
        
        quad_result = np.ldexp(quad_x, exp_val)
        float_result = np.ldexp(float_x, exp_val)
        
        assert isinstance(quad_result, QuadPrecision), f"Result should be QuadPrecision for ldexp({x_val}, {exp_val})"
        
        np.testing.assert_allclose(
            quad_result, float_result, rtol=1e-12, atol=1e-15,
            err_msg=f"ldexp({x_val}, {exp_val}) mismatch"
        )
        
        # Verify against direct calculation for finite results
        if np.isfinite(float_result):
            expected = float(x_val) * (2 ** exp_val)
            np.testing.assert_allclose(
                quad_result, expected, rtol=1e-10,
                err_msg=f"ldexp({x_val}, {exp_val}) doesn't match x * 2^exp"
            )
    
    @pytest.mark.parametrize("x_val,exp_val", [
        ("0.0", 0),
        ("0.0", 1),
        ("0.0", -1),
        ("0.0", 100),
        ("0.0", -100),
        ("-0.0", 0),
        ("-0.0", 1),
        ("-0.0", -1),
        ("-0.0", 100),
        ("-0.0", -100),
    ])
    def test_ldexp_zero(self, x_val, exp_val):
        """Test ldexp with zero values (should preserve sign)"""
        quad_x = QuadPrecision(x_val)
        float_x = np.float64(x_val)
        
        quad_result = np.ldexp(quad_x, exp_val)
        float_result = np.ldexp(float_x, exp_val)
        
        # Zero * 2^exp = zero (with sign preserved)
        assert quad_result == 0.0, f"ldexp({x_val}, {exp_val}) should be zero"
        assert np.signbit(quad_result) == np.signbit(float_result), \
            f"Sign mismatch for ldexp({x_val}, {exp_val})"
    
    @pytest.mark.parametrize("x_val,exp_val", [
        ("inf", 0),
        ("inf", 1),
        ("inf", -1),
        ("inf", 100),
        ("-inf", 0),
        ("-inf", 1),
        ("-inf", -1),
        ("-inf", 100),
    ])
    def test_ldexp_inf(self, x_val, exp_val):
        """Test ldexp with infinity (should preserve infinity and sign)"""
        quad_x = QuadPrecision(x_val)
        float_x = np.float64(x_val)
        
        quad_result = np.ldexp(quad_x, exp_val)
        float_result = np.ldexp(float_x, exp_val)
        
        assert np.isinf(quad_result), f"ldexp({x_val}, {exp_val}) should be infinity"
        assert np.signbit(quad_result) == np.signbit(float_result), \
            f"Sign mismatch for ldexp({x_val}, {exp_val})"
    
    @pytest.mark.parametrize("x_val,exp_val", [
        ("nan", 0),
        ("nan", 1),
        ("nan", -1),
        ("nan", 100),
        ("-nan", 0),
    ])
    def test_ldexp_nan(self, x_val, exp_val):
        """Test ldexp with NaN (should return NaN)"""
        quad_x = QuadPrecision(x_val)
        
        quad_result = np.ldexp(quad_x, exp_val)
        
        assert np.isnan(quad_result), f"ldexp({x_val}, {exp_val}) should be NaN"
    
    @pytest.mark.parametrize("x_val,exp_val", [
        ("1.5", 16384),  # Large positive exponent (likely overflow)
        ("2.0", 20000),
    ])
    def test_ldexp_overflow(self, x_val, exp_val):
        """Test ldexp with overflow to infinity"""
        quad_x = QuadPrecision(x_val)
        float_x = np.float64(x_val)
        
        quad_result = np.ldexp(quad_x, exp_val)
        float_result = np.ldexp(float_x, exp_val)
        
        # Both should overflow to infinity
        assert np.isinf(quad_result), f"ldexp({x_val}, {exp_val}) should overflow to infinity"
        assert np.isinf(float_result), f"numpy ldexp({x_val}, {exp_val}) should overflow to infinity"
        assert np.signbit(quad_result) == np.signbit(float_result), \
            f"Sign mismatch for overflow ldexp({x_val}, {exp_val})"
    
    @pytest.mark.parametrize("x_val,exp_val", [
        ("1.5", -16500),  # Large negative exponent (likely underflow)
        ("2.0", -20000),
    ])
    def test_ldexp_underflow(self, x_val, exp_val):
        """Test ldexp with underflow to zero"""
        quad_x = QuadPrecision(x_val)
        float_x = np.float64(x_val)
        
        quad_result = np.ldexp(quad_x, exp_val)
        float_result = np.ldexp(float_x, exp_val)
        
        # Both should underflow to zero
        assert quad_result == 0.0, f"ldexp({x_val}, {exp_val}) should underflow to zero"
        assert float_result == 0.0, f"numpy ldexp({x_val}, {exp_val}) should underflow to zero"
        # Sign should be preserved
        assert np.signbit(quad_result) == np.signbit(float(x_val)), \
            f"Sign should be preserved for underflow ldexp({x_val}, {exp_val})"


class TestFrexp:
    """Tests for frexp function (decompose x into mantissa and exponent)"""
    
    @pytest.mark.parametrize("x_val", [
        "1.0",
        "2.0",
        "3.0",
        "4.0",
        "0.5",
        "0.25",
        "8.0",
        "16.0",
        "-1.0",
        "-2.0",
        "-3.0",
        "-4.0",
        "-0.5",
        "-0.25",
        "-8.0",
        "-16.0",
        "1.5",
        "2.5",
        "3.5",
        "-1.5",
        "-2.5",
        "0.1",
        "0.9",
        "1000.0",
        "-1000.0",
    ])
    def test_frexp_basic(self, x_val):
        """Test frexp with basic values - work directly with QuadPrecision"""
        quad_x = QuadPrecision(x_val)
        
        # Get results
        quad_m, quad_e = np.frexp(quad_x)
        
        # Check types
        assert isinstance(quad_m, QuadPrecision), f"Mantissa should be QuadPrecision for frexp({x_val})"
        assert isinstance(quad_e, (int, np.integer)), f"Exponent should be integer for frexp({x_val})"
        
        # Verify mantissa is in correct range: 0.5 <= |mantissa| < 1.0
        abs_m = abs(quad_m)
        half = QuadPrecision("0.5")
        one = QuadPrecision("1.0")
        assert abs_m >= half and abs_m < one, \
            f"Mantissa {quad_m} for frexp({x_val}) not in [0.5, 1.0) range"
        
        # Verify reconstruction: x = mantissa * 2^exponent using ldexp
        reconstructed = np.ldexp(quad_m, int(quad_e))
        # Compare directly without float conversion
        assert reconstructed == quad_x, \
            f"Reconstruction failed for frexp({x_val}): {reconstructed} != {quad_x}"
        
        # Compare with NumPy float64 to ensure results are close
        float_x = np.float64(x_val)
        float_m, float_e = np.frexp(float_x)
        
        # Mantissa should be close to float64 result (within float64 precision)
        np.testing.assert_allclose(
            quad_m, float_m, rtol=1e-15, atol=1e-15,
            err_msg=f"Mantissa differs from NumPy float64 for frexp({x_val})"
        )
        
        # Exponent should match exactly for values in float64 range
        assert quad_e == float_e, \
            f"Exponent mismatch with NumPy for frexp({x_val}): {quad_e} != {float_e}"
    
    @pytest.mark.parametrize("x_val", [
        "0.0",
        "-0.0",
    ])
    def test_frexp_zero(self, x_val):
        """Test frexp with zero values (should return ±0 mantissa, exponent 0)"""
        quad_x = QuadPrecision(x_val)
        
        quad_m, quad_e = np.frexp(quad_x)
        
        # Mantissa should be zero with same sign
        zero = QuadPrecision("0.0")
        assert quad_m == zero, f"Mantissa should be zero for frexp({x_val})"
        assert np.signbit(quad_m) == np.signbit(quad_x), \
            f"Sign mismatch for frexp({x_val}) mantissa"
        
        # Exponent should be 0
        assert quad_e == 0, f"Exponent should be 0 for frexp({x_val})"
        
        # Compare with NumPy float64
        float_x = np.float64(x_val)
        float_m, float_e = np.frexp(float_x)
        
        # Mantissa should match (both zero with same sign)
        np.testing.assert_allclose(
            quad_m, float_m, rtol=0, atol=0,
            err_msg=f"Mantissa differs from NumPy float64 for frexp({x_val})"
        )
        assert np.signbit(quad_m) == np.signbit(float_m), \
            f"Sign mismatch with NumPy for frexp({x_val})"
        
        # Exponent should match
        assert quad_e == float_e, \
            f"Exponent mismatch with NumPy for frexp({x_val}): {quad_e} != {float_e}"
    
    @pytest.mark.parametrize("x_val", [
        "inf",
        "-inf",
        "nan",
        "-nan",
    ])
    def test_frexp_special_values(self, x_val):
        """Test frexp with special values (inf, nan)
        
        For these edge cases, the C standard specifies that the exponent value
        is unspecified/implementation-defined. We only verify that:
        1. The mantissa matches the expected value (±inf or NaN)
        2. The mantissa behavior matches NumPy's float64
        3. The exponent is an integer type
        
        We do NOT compare exponent values as they can differ across platforms
        (e.g., Linux returns 0, Windows returns -1).
        """
        quad_x = QuadPrecision(x_val)
        quad_m, quad_e = np.frexp(quad_x)
        
        # Compare with NumPy float64
        float_x = np.float64(x_val)
        float_m, float_e = np.frexp(float_x)
        
        # Exponent should be an integer type (but we don't check the value)
        assert isinstance(quad_e, (int, np.integer)), \
            f"Exponent should be integer type for frexp({x_val})"
        
        # Check mantissa behavior
        if "inf" in x_val:
            # Mantissa should be infinity with same sign
            assert np.isinf(quad_m), f"Mantissa should be infinity for frexp({x_val})"
            assert np.isinf(float_m), f"NumPy mantissa should also be infinity for frexp({x_val})"
            assert np.signbit(quad_m) == np.signbit(quad_x), \
                f"Sign mismatch for frexp({x_val}) mantissa"
            assert np.signbit(quad_m) == np.signbit(float_m), \
                f"Sign mismatch with NumPy for frexp({x_val})"
        else:  # nan
            # Mantissa should be NaN
            assert np.isnan(quad_m), f"Mantissa should be NaN for frexp({x_val})"
            assert np.isnan(float_m), f"NumPy mantissa should also be NaN for frexp({x_val})"
    
    def test_frexp_very_large(self):
        """Test frexp with very large values"""
        # Large value that's still finite
        quad_x = QuadPrecision("1e100")
        
        quad_m, quad_e = np.frexp(quad_x)
        
        # Verify mantissa range
        abs_m = abs(quad_m)
        half = QuadPrecision("0.5")
        one = QuadPrecision("1.0")
        assert abs_m >= half and abs_m < one, \
            f"Mantissa {quad_m} for large value not in [0.5, 1.0) range"
        
        # Verify reconstruction using ldexp (preserves full quad precision)
        reconstructed = np.ldexp(quad_m, int(quad_e))
        assert reconstructed == quad_x, \
            f"Reconstruction failed for large value: {reconstructed} != {quad_x}"
    
    def test_frexp_very_small(self):
        """Test frexp with very small positive values"""
        # Small positive value
        quad_x = QuadPrecision("1e-100")
        
        quad_m, quad_e = np.frexp(quad_x)
        
        # Verify mantissa range
        abs_m = abs(quad_m)
        half = QuadPrecision("0.5")
        one = QuadPrecision("1.0")
        assert abs_m >= half and abs_m < one, \
            f"Mantissa {quad_m} for small value not in [0.5, 1.0) range"
        
        # Verify reconstruction using ldexp (preserves full quad precision)
        reconstructed = np.ldexp(quad_m, int(quad_e))
        assert reconstructed == quad_x, \
            f"Reconstruction failed for small value: {reconstructed} != {quad_x}"
# testng buffer
def test_buffer():
    a = QuadPrecision(1.0)
    buff = a.data

    reconstructed = np.frombuffer(buff, dtype=QuadPrecDType())[0]
    assert reconstructed == a, "Buffer reconstruction failed"

@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, -1.0, 3.14, -2.71, "inf", "-inf", "nan"])
def test_imag_real(value):
    a = QuadPrecision(value)
    if np.isnan(a):
        assert np.isnan(a.real), "Real part of NaN should be NaN"
        assert a.imag == QuadPrecision(0.0), "Imaginary part should be zero"
        return
    assert a.real == a, "Real part mismatch"
    assert a.imag == QuadPrecision(0.0), "Imaginary part should be zero"


class TestStringParsing:
    """Test suite for string parsing functionality (fromstr and scanfunc)."""
    
    def test_fromstring_simple(self):
        """Test np.fromstring with simple values."""
        result = np.fromstring("1.5 2.5 3.5", sep=" ", dtype=QuadPrecDType(backend='sleef'))
        expected = np.array([1.5, 2.5, 3.5], dtype=QuadPrecDType(backend='sleef'))
        np.testing.assert_array_equal(result, expected)
    
    def test_fromstring_high_precision(self):
        """Test np.fromstring preserves high precision values."""
        # Create a high-precision value
        finfo = np.finfo(QuadPrecision)
        val = 1.0 + finfo.eps
        val_str = str(val)
        
        # Parse it back
        result = np.fromstring(val_str, sep=" ", dtype=QuadPrecDType(backend='sleef'))
        expected = np.array([val], dtype=QuadPrecDType(backend='sleef'))
        
        # Should maintain precision
        assert result[0] == expected[0], "High precision value not preserved"
    
    def test_fromstring_multiple_values(self):
        """Test np.fromstring with multiple values."""
        s = " 1.0 2.0 3.0 4.0 5.0"
        result = np.fromstring(s, sep=" ", dtype=QuadPrecDType(backend='sleef'))
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=QuadPrecDType(backend='sleef'))
        np.testing.assert_array_equal(result, expected)
    
    def test_fromstring_newline_separator(self):
        """Test np.fromstring with newline separator."""
        s = "1.5\n2.5\n3.5"
        result = np.fromstring(s, sep="\n", dtype=QuadPrecDType(backend='sleef'))
        expected = np.array([1.5, 2.5, 3.5], dtype=QuadPrecDType(backend='sleef'))
        np.testing.assert_array_equal(result, expected)
    
    def test_fromstring_scientific_notation(self):
        """Test np.fromstring with scientific notation."""
        s = "1.23e-10 4.56e20"
        result = np.fromstring(s, sep=" ", dtype=QuadPrecDType(backend='sleef'))
        expected = np.array([1.23e-10, 4.56e20], dtype=QuadPrecDType(backend='sleef'))
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fromstring_negative_values(self):
        """Test np.fromstring with negative values."""
        s = "-1.5 -2.5 -3.5"
        result = np.fromstring(s, sep=" ", dtype=QuadPrecDType(backend='sleef'))
        expected = np.array([-1.5, -2.5, -3.5], dtype=QuadPrecDType(backend='sleef'))
        np.testing.assert_array_equal(result, expected)


class TestFileIO:
    """Test suite for file I/O functionality (scanfunc)."""
    
    def test_fromfile_simple(self):
        """Test np.fromfile with simple values."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("1.5\n2.5\n3.5")
            fname = f.name
        
        try:
            result = np.fromfile(fname, sep="\n", dtype=QuadPrecDType(backend='sleef'))
            expected = np.array([1.5, 2.5, 3.5], dtype=QuadPrecDType(backend='sleef'))
            np.testing.assert_array_equal(result, expected)
        finally:
            os.unlink(fname)
    
    def test_fromfile_space_separator(self):
        """Test np.fromfile with space separator."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("1.0 2.0 3.0 4.0 5.0")
            fname = f.name
        
        try:
            result = np.fromfile(fname, sep=" ", dtype=QuadPrecDType(backend='sleef'))
            expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=QuadPrecDType(backend='sleef'))
            np.testing.assert_array_equal(result, expected)
        finally:
            os.unlink(fname)
    
    def test_tofile_fromfile_roundtrip(self):
        """Test that tofile/fromfile roundtrips correctly."""
        import tempfile
        import os
        
        original = np.array([1.5, 2.5, 3.5, 4.5], dtype=QuadPrecDType(backend='sleef'))
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            fname = f.name
        
        try:
            # Write to file
            original.tofile(fname, sep=" ")
            
            # Read back
            result = np.fromfile(fname, sep=" ", dtype=QuadPrecDType(backend='sleef'))
            
            np.testing.assert_array_equal(result, original)
        finally:
            os.unlink(fname)
    
    def test_fromfile_high_precision(self):
        """Test np.fromfile preserves high precision values."""
        import tempfile
        import os
        
        # Create a high-precision value
        finfo = np.finfo(QuadPrecision)
        val = 1.0 + finfo.eps
        expected = np.array([val, val, val], dtype=QuadPrecDType(backend='sleef'))
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for v in expected:
                f.write(str(v) + '\n')
            fname = f.name
        
        try:
            result = np.fromfile(fname, sep="\n", dtype=QuadPrecDType(backend='sleef'))
            
            # Check each value maintains precision
            for i in range(len(expected)):
                assert result[i] == expected[i], f"High precision value {i} not preserved"
        finally:
            os.unlink(fname)
    
    def test_fromfile_no_trailing_newline(self):
        """Test np.fromfile handles files without trailing newline."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            # Write without trailing newline
            f.write("1.5\n2.5\n3.5")
            fname = f.name
        
        try:
            result = np.fromfile(fname, sep="\n", dtype=QuadPrecDType(backend='sleef'))
            expected = np.array([1.5, 2.5, 3.5], dtype=QuadPrecDType(backend='sleef'))
            np.testing.assert_array_equal(result, expected)
        finally:
            os.unlink(fname)
    
    def test_fromfile_empty_file(self):
        """Test np.fromfile with empty file."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            fname = f.name
        
        try:
            result = np.fromfile(fname, sep="\n", dtype=QuadPrecDType(backend='sleef'))
            assert len(result) == 0, "Empty file should produce empty array"
        finally:
            os.unlink(fname)
    
    def test_fromfile_single_value(self):
        """Test np.fromfile with single value."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("42.0")
            fname = f.name
        
        try:
            result = np.fromfile(fname, sep=" ", dtype=QuadPrecDType(backend='sleef'))
            expected = np.array([42.0], dtype=QuadPrecDType(backend='sleef'))
            np.testing.assert_array_equal(result, expected)
        finally:
            os.unlink(fname)


class Test_Is_Integer_Methods:
    """Test suite for float compatibility methods: is_integer() and as_integer_ratio()."""
    
    @pytest.mark.parametrize("value,expected", [
        # Positive integers
        ("1.0", True),
        ("42.0", True),
        ("1000.0", True),
        # Negative integers
        ("-1.0", True),
        ("-42.0", True),
        # Zero
        ("0.0", True),
        ("-0.0", True),
        # Large integers
        ("1e20", True),
        ("123456789012345678901234567890", True),
        # Fractional values
        ("1.5", False),
        ("3.14", False),
        ("-2.5", False),
        ("0.1", False),
        ("0.0001", False),
        # Values close to integers but not exact
        ("1.0000000000001", False),
        ("0.9999999999999", False),
        # Special values
        ("inf", False),
        ("-inf", False),
        ("nan", False),
    ])
    def test_is_integer(self, value, expected):
        """Test is_integer() returns correct result for various values."""
        assert QuadPrecision(value).is_integer() == expected
    
    @pytest.mark.parametrize("value", ["0.0", "1.0", "1.5", "-3.0", "-3.7", "42.0"])
    def test_is_integer_compatibility_with_float(self, value):
        """Test is_integer() matches behavior of Python's float."""
        quad_val = QuadPrecision(value)
        float_val = float(value)
        assert quad_val.is_integer() == float_val.is_integer()
    
    @pytest.mark.parametrize("value,expected_num,expected_denom", [
        ("1.0", 1, 1),
        ("42.0", 42, 1),
        ("-5.0", -5, 1),
        ("0.0", 0, 1),
        ("-0.0", 0, 1),
    ])
    def test_as_integer_ratio_integers(self, value, expected_num, expected_denom):
        """Test as_integer_ratio() for integer values."""
        num, denom = QuadPrecision(value).as_integer_ratio()
        assert num == expected_num and denom == expected_denom
    
    @pytest.mark.parametrize("value,expected_ratio", [
        ("0.5", 0.5),
        ("0.25", 0.25),
        ("1.5", 1.5),
        ("-2.5", -2.5),
    ])
    def test_as_integer_ratio_fractional(self, value, expected_ratio):
        """Test as_integer_ratio() for fractional values."""
        num, denom = QuadPrecision(value).as_integer_ratio()
        assert QuadPrecision(str(num)) / QuadPrecision(str(denom)) == QuadPrecision(str(expected_ratio))
        assert denom > 0  # Denominator should always be positive
    
    @pytest.mark.parametrize("value", [
        "3.14", "0.1", "1.414213562373095", "2.718281828459045",
        "-1.23456789", "1000.001", "0.0001", "1e20", "1.23e15", "1e-30", quad_pi
    ])
    def test_as_integer_ratio_reconstruction(self, value):
        """Test that as_integer_ratio() can reconstruct the original value."""
        quad_val = QuadPrecision(value)
        num, denom = quad_val.as_integer_ratio()
        # todo: can remove str converstion after merging PR #213
        reconstructed = QuadPrecision(str(num)) / QuadPrecision(str(denom))
        assert reconstructed == quad_val
    
    def test_as_integer_ratio_return_types(self):
        """Test that as_integer_ratio() returns Python ints."""
        num, denom = QuadPrecision("3.14").as_integer_ratio()
        assert isinstance(num, int)
        assert isinstance(denom, int)
    
    @pytest.mark.parametrize("value", ["-1.0", "-3.14", "-0.5", "1.0", "3.14", "0.5"])
    def test_as_integer_ratio_denominator_positive(self, value):
        """Test that denominator is always positive."""
        num, denom = QuadPrecision(value).as_integer_ratio()
        assert denom > 0
    
    @pytest.mark.parametrize("value,exception,match", [
        ("inf", OverflowError, "Cannot convert infinite value to integer ratio"),
        ("-inf", OverflowError, "Cannot convert infinite value to integer ratio"),
        ("nan", ValueError, "Cannot convert NaN to integer ratio"),
    ])
    def test_as_integer_ratio_special_values_raise(self, value, exception, match):
        """Test that as_integer_ratio() raises appropriate errors for special values."""
        with pytest.raises(exception, match=match):
            QuadPrecision(value).as_integer_ratio()
    
    @pytest.mark.parametrize("value", ["1.0", "0.5", "3.14", "-2.5", "0.0"])
    def test_as_integer_ratio_compatibility_with_float(self, value):
        """Test as_integer_ratio() matches behavior of Python's float where possible."""
        quad_val = QuadPrecision(value)
        float_val = float(value)
        
        quad_num, quad_denom = quad_val.as_integer_ratio()
        float_num, float_denom = float_val.as_integer_ratio()
        
        # The ratios should be equal
        quad_ratio = quad_num / quad_denom
        float_ratio = float_num / float_denom
        assert abs(quad_ratio - float_ratio) < 1e-15

def test_quadprecision_scalar_dtype_expose():
    quad_ld = QuadPrecision("1e100", backend="longdouble")
    quad_sleef = QuadPrecision("1e100", backend="sleef")
    assert quad_ld.dtype == QuadPrecDType(backend='longdouble')
    assert np.dtype(quad_ld) == QuadPrecDType(backend='longdouble')
    
    assert quad_ld.dtype.backend == 1
    assert np.dtype(quad_ld).backend == 1

    assert quad_sleef.dtype == QuadPrecDType(backend='sleef')
    assert np.dtype(quad_sleef) == QuadPrecDType(backend='sleef')
    assert quad_sleef.dtype.backend == 0
    assert np.dtype(quad_sleef).backend == 0


class TestPickle:
    """Comprehensive test suite for pickle support in QuadPrecDType."""
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_dtype_basic(self, backend):
        """Test basic pickle/unpickle of QuadPrecDType instances."""
        import pickle
        
        # Create original dtype
        original = QuadPrecDType(backend=backend)
        
        # Pickle and unpickle
        pickled = pickle.dumps(original)
        unpickled = pickle.loads(pickled)
        
        # Verify dtype is preserved
        assert isinstance(unpickled, type(original))
        assert unpickled.backend == original.backend
        assert str(unpickled) == str(original)
    
    @pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_dtype_all_protocols(self, protocol, backend):
        """Test pickle with all pickle protocol versions."""
        import pickle
        
        original = QuadPrecDType(backend=backend)
        pickled = pickle.dumps(original, protocol=protocol)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.backend == original.backend
        assert str(unpickled) == str(original)
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    @pytest.mark.parametrize("value", [
        "0.0", "-0.0", "1.0", "-1.0", 
        "3.141592653589793238462643383279502884197",
        "2.718281828459045235360287471352662497757",
        "1e100", "1e-100", "-1e100", "-1e-100",
        "inf", "-inf", "nan"
    ])
    def test_pickle_scalar(self, backend, value):
        """Test pickle/unpickle of QuadPrecision scalars in arrays."""
        import pickle
        
        # Create scalar as 0-d array (scalars pickle differently)
        original = np.array(QuadPrecision(value, backend=backend))
        
        # Pickle and unpickle
        pickled = pickle.dumps(original)
        unpickled = pickle.loads(pickled)
        
        # Verify value is preserved
        if np.isnan(float(original[()])):
            assert np.isnan(float(unpickled[()]))
        else:
            assert unpickled[()] == original[()]
            assert float(unpickled[()]) == float(original[()])
        
        # Verify dtype and backend
        assert unpickled.dtype == original.dtype
        assert unpickled.dtype.backend == original.dtype.backend
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_array_1d(self, backend):
        """Test pickle/unpickle of 1D arrays."""
        import pickle
        
        # Create array
        original = np.array([1.5, 2.5, 3.5, 4.5], dtype=QuadPrecDType(backend=backend))
        
        # Pickle and unpickle
        pickled = pickle.dumps(original)
        unpickled = pickle.loads(pickled)
        
        # Verify array is preserved
        np.testing.assert_array_equal(unpickled, original)
        assert unpickled.dtype == original.dtype
        assert unpickled.dtype.backend == original.dtype.backend
        assert unpickled.shape == original.shape
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_array_2d(self, backend):
        """Test pickle/unpickle of 2D arrays."""
        import pickle
        
        # Create 2D array
        original = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]], dtype=QuadPrecDType(backend=backend))
        
        # Pickle and unpickle
        pickled = pickle.dumps(original)
        unpickled = pickle.loads(pickled)
        
        # Verify array is preserved
        np.testing.assert_array_equal(unpickled, original)
        assert unpickled.dtype == original.dtype
        assert unpickled.dtype.backend == original.dtype.backend
        assert unpickled.shape == original.shape
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_array_special_values(self, backend):
        """Test pickle/unpickle of arrays with special values."""
        import pickle
        
        # Create array with special values
        original = np.array([
            QuadPrecision("0.0", backend=backend),
            QuadPrecision("-0.0", backend=backend),
            QuadPrecision("inf", backend=backend),
            QuadPrecision("-inf", backend=backend),
            QuadPrecision("nan", backend=backend),
            QuadPrecision("1e100", backend=backend),
            QuadPrecision("1e-100", backend=backend),
        ], dtype=QuadPrecDType(backend=backend))
        
        # Pickle and unpickle
        pickled = pickle.dumps(original)
        unpickled = pickle.loads(pickled)
        
        # Verify each element (handling NaN specially)
        for i in range(len(original)):
            if np.isnan(float(original[i])):
                assert np.isnan(float(unpickled[i]))
            else:
                assert float(unpickled[i]) == float(original[i])
                # Check sign for zeros
                if float(original[i]) == 0.0:
                    assert np.signbit(unpickled[i]) == np.signbit(original[i])
        
        assert unpickled.dtype == original.dtype
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_empty_array(self, backend):
        """Test pickle/unpickle of empty arrays."""
        import pickle
        
        # Create empty array
        original = np.array([], dtype=QuadPrecDType(backend=backend))
        
        # Pickle and unpickle
        pickled = pickle.dumps(original)
        unpickled = pickle.loads(pickled)
        
        # Verify empty array is preserved
        assert len(unpickled) == 0
        assert unpickled.dtype == original.dtype
        assert unpickled.shape == original.shape
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_high_precision_values(self, backend):
        """Test that high precision is preserved through pickle."""
        import pickle
        
        # Create high-precision values
        pi_str = "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067"
        e_str = "2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427"
        
        original = np.array([
            QuadPrecision(pi_str, backend=backend),
            QuadPrecision(e_str, backend=backend),
        ], dtype=QuadPrecDType(backend=backend))
        
        # Pickle and unpickle
        pickled = pickle.dumps(original)
        unpickled = pickle.loads(pickled)
        
        # Verify high precision is maintained
        for i in range(len(original)):
            assert unpickled[i] == original[i]
            assert str(unpickled[i]) == str(original[i])
        
        assert unpickled.dtype == original.dtype
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_with_npz(self, backend):
        """Test saving and loading arrays with np.savez."""
        import tempfile
        import os
        
        # Create arrays
        arr1 = np.array([1.5, 2.5, 3.5], dtype=QuadPrecDType(backend=backend))
        arr2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=QuadPrecDType(backend=backend))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            fname = f.name
        
        try:
            # Save arrays
            np.savez(fname, array1=arr1, array2=arr2)
            
            # Load arrays (custom dtypes require allow_pickle=True)
            loaded = np.load(fname, allow_pickle=True)
            loaded_arr1 = loaded['array1']
            loaded_arr2 = loaded['array2']
            
            # Verify arrays are preserved
            np.testing.assert_array_equal(loaded_arr1, arr1)
            np.testing.assert_array_equal(loaded_arr2, arr2)
            assert loaded_arr1.dtype == arr1.dtype
            assert loaded_arr2.dtype == arr2.dtype
            expected_backend = 0 if backend == 'sleef' else 1
            assert loaded_arr1.dtype.backend == expected_backend
            assert loaded_arr2.dtype.backend == expected_backend
            
            # Close the file before cleanup (required on Windows)
            loaded.close()
        finally:
            os.unlink(fname)
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_with_savez_compressed(self, backend):
        """Test saving and loading arrays with np.savez_compressed."""
        import tempfile
        import os
        
        # Create array with many values
        original = np.linspace(0, 100, 1000, dtype=np.float64).astype(QuadPrecDType(backend=backend))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            fname = f.name
        
        try:
            # Save compressed
            np.savez_compressed(fname, data=original)
            
            # Load (custom dtypes require allow_pickle=True)
            loaded = np.load(fname, allow_pickle=True)
            loaded_arr = loaded['data']
            
            # Verify array is preserved
            np.testing.assert_array_equal(loaded_arr, original)
            assert loaded_arr.dtype == original.dtype
            expected_backend = 0 if backend == 'sleef' else 1
            assert loaded_arr.dtype.backend == expected_backend
            
            # Close the file before cleanup (required on Windows)
            loaded.close()
        finally:
            os.unlink(fname)
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_npz_special_values(self, backend):
        """Test np.savez with arrays containing special values."""
        import tempfile
        import os
        
        # Create array with special values
        original = np.array([
            QuadPrecision("0.0", backend=backend),
            QuadPrecision("-0.0", backend=backend),
            QuadPrecision("inf", backend=backend),
            QuadPrecision("-inf", backend=backend),
            QuadPrecision("nan", backend=backend),
        ], dtype=QuadPrecDType(backend=backend))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            fname = f.name
        
        try:
            # Save
            np.savez(fname, special=original)
            
            # Load (custom dtypes require allow_pickle=True)
            loaded = np.load(fname, allow_pickle=True)
            loaded_arr = loaded['special']
            
            # Verify each element
            for i in range(len(original)):
                if np.isnan(float(original[i])):
                    assert np.isnan(float(loaded_arr[i]))
                else:
                    assert float(loaded_arr[i]) == float(original[i])
                    if float(original[i]) == 0.0:
                        assert np.signbit(loaded_arr[i]) == np.signbit(original[i])
            
            assert loaded_arr.dtype == original.dtype
            
            # Close the file before cleanup (required on Windows)
            loaded.close()
        finally:
            os.unlink(fname)
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_npz_multiple_arrays(self, backend):
        """Test np.savez with multiple arrays of different shapes."""
        import tempfile
        import os
        
        # Create arrays of different shapes
        arr_scalar = np.array(QuadPrecision("42.0", backend=backend))
        arr_1d = np.array([1.0, 2.0, 3.0], dtype=QuadPrecDType(backend=backend))
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=QuadPrecDType(backend=backend))
        arr_3d = np.arange(24, dtype=np.float64).reshape((2, 3, 4)).astype(QuadPrecDType(backend=backend))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            fname = f.name
        
        try:
            # Save all arrays
            np.savez(fname, 
                    scalar=arr_scalar,
                    one_d=arr_1d,
                    two_d=arr_2d,
                    three_d=arr_3d)
            
            # Load (custom dtypes require allow_pickle=True)
            loaded = np.load(fname, allow_pickle=True)
            
            # Verify all arrays
            np.testing.assert_array_equal(loaded['scalar'], arr_scalar)
            np.testing.assert_array_equal(loaded['one_d'], arr_1d)
            np.testing.assert_array_equal(loaded['two_d'], arr_2d)
            np.testing.assert_array_equal(loaded['three_d'], arr_3d)
            
            # Verify dtypes and backends
            expected_backend = 0 if backend == 'sleef' else 1
            for key in ['scalar', 'one_d', 'two_d', 'three_d']:
                assert loaded[key].dtype.backend == expected_backend
            
            # Close the file before cleanup (required on Windows)
            loaded.close()
        finally:
            os.unlink(fname)
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_npz_exact_values(self, backend):
        """Test that np.savez preserves exact values using np.testing.assert_allclose."""
        import tempfile
        import os
        
        # Create array with precise values
        original = np.array([
            1.0 / 3.0,  # Repeating decimal
            np.sqrt(2.0),  # Irrational
            np.pi,  # Pi
            np.e,  # Euler's number
            1.23456789012345678901234567890,  # Many digits
        ], dtype=np.float64).astype(QuadPrecDType(backend=backend))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            fname = f.name
        
        try:
            # Save
            np.savez(fname, data=original)
            
            # Load (custom dtypes require allow_pickle=True)
            loaded = np.load(fname, allow_pickle=True)
            loaded_arr = loaded['data']
            
            # Verify exact values using assert_allclose
            np.testing.assert_allclose(
                loaded_arr.astype(np.float64),
                original.astype(np.float64),
                rtol=0, atol=0,  # Exact comparison
                err_msg="Values changed after save/load"
            )
            
            # Also check element-wise equality
            for i in range(len(original)):
                assert loaded_arr[i] == original[i]
            
            # Close the file before cleanup (required on Windows)
            loaded.close()
        finally:
            os.unlink(fname)
    
    def test_pickle_backend_preservation_sleef_to_longdouble(self):
        """Test that different backends maintain their identity through pickle."""
        import pickle
        
        # Create arrays with different backends
        sleef_arr = np.array([1.5, 2.5], dtype=QuadPrecDType(backend='sleef'))
        longdouble_arr = np.array([1.5, 2.5], dtype=QuadPrecDType(backend='longdouble'))
        
        # Pickle both
        sleef_pickled = pickle.dumps(sleef_arr)
        longdouble_pickled = pickle.dumps(longdouble_arr)
        
        # Unpickle
        sleef_unpickled = pickle.loads(sleef_pickled)
        longdouble_unpickled = pickle.loads(longdouble_pickled)
        
        # Verify backends are preserved
        assert sleef_unpickled.dtype.backend == 0  # BACKEND_SLEEF
        assert longdouble_unpickled.dtype.backend == 1  # BACKEND_LONGDOUBLE
        
        # Verify they are different
        assert sleef_unpickled.dtype.backend != longdouble_unpickled.dtype.backend
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_array_view(self, backend):
        """Test pickle/unpickle of array views."""
        import pickle
        
        # Create array and view
        base_array = np.arange(10, dtype=np.float64).astype(QuadPrecDType(backend=backend))
        view = base_array[2:8:2]  # Slice with stride
        
        # Pickle and unpickle the view
        pickled = pickle.dumps(view)
        unpickled = pickle.loads(pickled)
        
        # Verify view is preserved
        np.testing.assert_array_equal(unpickled, view)
        assert unpickled.dtype == view.dtype
        assert unpickled.shape == view.shape
    
    @pytest.mark.parametrize("backend", ["sleef", "longdouble"])
    def test_pickle_fortran_order(self, backend):
        """Test pickle/unpickle of Fortran-ordered arrays."""
        import pickle
        
        # Create Fortran-ordered array
        original = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]], 
                           dtype=QuadPrecDType(backend=backend),
                           order='F')
        
        # Pickle and unpickle
        pickled = pickle.dumps(original)
        unpickled = pickle.loads(pickled)
        
        # Verify array is preserved
        np.testing.assert_array_equal(unpickled, original)
        assert unpickled.dtype == original.dtype
        assert unpickled.flags.f_contiguous == original.flags.f_contiguous