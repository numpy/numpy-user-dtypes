import pytest
import numpy as np
from numpy_quaddtype import QuadPrecision, QuadPrecDType


# ================================================================================
# UTILITIES
# ================================================================================

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


# ================================================================================
# VECTOR-VECTOR DOT PRODUCT TESTS
# ================================================================================

class TestVectorVectorDot:
    """Test vector-vector np.matmul products"""
    
    def test_simple_dot_product(self):
        """Test basic vector np.matmul product"""
        x = create_quad_array([1, 2, 3])
        y = create_quad_array([4, 5, 6])
        
        result = np.matmul(x, y)
        expected = 1*4 + 2*5 + 3*6  # = 32
        
        assert isinstance(result, QuadPrecision)
        assert_quad_equal(result, expected)
    
    def test_orthogonal_vectors(self):
        """Test orthogonal vectors (should give zero)"""
        x = create_quad_array([1, 0, 0])
        y = create_quad_array([0, 1, 0])
        
        result = np.matmul(x, y)
        assert_quad_equal(result, 0.0)
    
    def test_same_vector(self):
        """Test np.matmul product of vector with itself"""
        x = create_quad_array([2, 3, 4])
        
        result = np.matmul(x, x)
        expected = 2*2 + 3*3 + 4*4  # = 29
        
        assert_quad_equal(result, expected)
    
    @pytest.mark.parametrize("size", [1, 2, 5, 10, 50, 100])
    def test_various_vector_sizes(self, size):
        """Test different vector sizes from small to large"""
        # Create vectors with known pattern
        x_vals = [i + 1 for i in range(size)]  # [1, 2, 3, ...]
        y_vals = [2 * (i + 1) for i in range(size)]  # [2, 4, 6, ...]
        
        x = create_quad_array(x_vals)
        y = create_quad_array(y_vals)
        
        result = np.matmul(x, y)
        expected = sum(x_vals[i] * y_vals[i] for i in range(size))
        
        assert_quad_equal(result, expected)
    
    def test_negative_and_fractional_values(self):
        """Test vectors with negative and fractional values"""
        x = create_quad_array([1.5, -2.5, 3.25])
        y = create_quad_array([-1.25, 2.75, -3.5])
        
        result = np.matmul(x, y)
        expected = 1.5*(-1.25) + (-2.5)*2.75 + 3.25*(-3.5)
        
        assert_quad_equal(result, expected)


# ================================================================================
# MATRIX-VECTOR MULTIPLICATION TESTS  
# ================================================================================

class TestMatrixVectorDot:
    """Test matrix-vector multiplication"""
    
    def test_simple_matrix_vector(self):
        """Test basic matrix-vector multiplication"""
        # 2x3 matrix
        A = create_quad_array([1, 2, 3, 4, 5, 6], shape=(2, 3))
        # 3x1 vector  
        x = create_quad_array([1, 1, 1])
        
        result = np.matmul(A, x)
        expected = [1+2+3, 4+5+6]  # [6, 15]
        
        assert result.shape == (2,)
        for i in range(2):
            assert_quad_equal(result[i], expected[i])
    
    def test_identity_matrix_vector(self):
        """Test multiplication with identity matrix"""
        # 3x3 identity matrix
        I = create_quad_array([1, 0, 0, 0, 1, 0, 0, 0, 1], shape=(3, 3))
        x = create_quad_array([2, 3, 4])
        
        result = np.matmul(I, x)
        
        assert result.shape == (3,)
        for i in range(3):
            assert_quad_equal(result[i], float(x[i]))
    
    @pytest.mark.parametrize("m,n", [(2,3), (3,2), (5,4), (10,8), (20,15)])
    def test_various_matrix_vector_sizes(self, m, n):
        """Test various matrix-vector sizes from small to large"""
        # Create m×n matrix with sequential values
        A_vals = [(i*n + j + 1) for i in range(m) for j in range(n)]
        A = create_quad_array(A_vals, shape=(m, n))
        
        # Create n×1 vector with simple values
        x_vals = [i + 1 for i in range(n)]
        x = create_quad_array(x_vals)
        
        result = np.matmul(A, x)
        
        assert result.shape == (m,)
        
        # Verify manually for small matrices
        if m <= 5 and n <= 5:
            for i in range(m):
                expected = sum(A_vals[i*n + j] * x_vals[j] for j in range(n))
                assert_quad_equal(result[i], expected)


# ================================================================================
# MATRIX-MATRIX MULTIPLICATION TESTS
# ================================================================================

class TestMatrixMatrixDot:
    """Test matrix-matrix multiplication"""
    
    def test_simple_matrix_matrix(self):
        """Test basic matrix-matrix multiplication"""
        # 2x2 matrices
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        B = create_quad_array([5, 6, 7, 8], shape=(2, 2))
        
        result = np.matmul(A, B)
        
        # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        expected = [[19, 22], [43, 50]]
        
        assert result.shape == (2, 2)
        for i in range(2):
            for j in range(2):
                assert_quad_equal(result[i, j], expected[i][j])
    
    def test_identity_matrix_multiplication(self):
        """Test multiplication with identity matrix"""
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        I = create_quad_array([1, 0, 0, 1], shape=(2, 2))
        
        # A * I should equal A
        result1 = np.matmul(A, I)
        assert_quad_array_equal(result1, A)
        
        # I * A should equal A  
        result2 = np.matmul(I, A)
        assert_quad_array_equal(result2, A)
    
    @pytest.mark.parametrize("m,n,k", [(2,2,2), (2,3,4), (3,2,5), (4,4,4), (5,6,7)])
    def test_various_matrix_sizes(self, m, n, k):
        """Test various matrix sizes: (m×k) × (k×n) = (m×n)"""
        # Create A: m×k matrix
        A_vals = [(i*k + j + 1) for i in range(m) for j in range(k)]
        A = create_quad_array(A_vals, shape=(m, k))
        
        # Create B: k×n matrix  
        B_vals = [(i*n + j + 1) for i in range(k) for j in range(n)]
        B = create_quad_array(B_vals, shape=(k, n))
        
        result = np.matmul(A, B)
        
        assert result.shape == (m, n)
        
        # Verify manually for small matrices
        if m <= 3 and n <= 3 and k <= 3:
            for i in range(m):
                for j in range(n):
                    expected = sum(A_vals[i*k + l] * B_vals[l*n + j] for l in range(k))
                    assert_quad_equal(result[i, j], expected)
    
    def test_associativity(self):
        """Test matrix multiplication associativity: (A*B)*C = A*(B*C)"""
        # Use small 2x2 matrices for simplicity
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        B = create_quad_array([2, 1, 1, 2], shape=(2, 2))
        C = create_quad_array([1, 1, 2, 1], shape=(2, 2))
        
        # Compute (A*B)*C
        AB = np.matmul(A, B)
        result1 = np.matmul(AB, C)
        
        # Compute A*(B*C)
        BC = np.matmul(B, C)
        result2 = np.matmul(A, BC)
        
        assert_quad_array_equal(result1, result2, rtol=1e-25)


# ================================================================================
# LARGE MATRIX TESTS
# ================================================================================

class TestLargeMatrices:
    """Test performance and correctness with larger matrices"""
    
    @pytest.mark.parametrize("size", [50, 100, 200])
    def test_large_square_matrices(self, size):
        """Test large square matrix multiplication"""
        # Create matrices with simple pattern for verification
        A_vals = [1.0 if i == j else 0.1 for i in range(size) for j in range(size)]  # Near-diagonal
        B_vals = [1.0] * (size * size)  # All ones
        
        A = create_quad_array(A_vals, shape=(size, size))
        B = create_quad_array(B_vals, shape=(size, size))
        
        result = np.matmul(A, B)
        
        assert result.shape == (size, size)
        
        # Each element = sum of a row in A = 1.0 + 0.1*(size-1)
        expected_value = 1.0 + 0.1 * (size - 1)
        
        # Check diagonal and off-diagonal elements
        assert_quad_equal(result[0, 0], expected_value, rtol=1e-15, atol=1e-15)
        if size > 1:
            assert_quad_equal(result[0, 1], expected_value, rtol=1e-15, atol=1e-15)
        
        # Additional verification: check a few more elements
        if size > 2:
            assert_quad_equal(result[1, 0], expected_value, rtol=1e-15, atol=1e-15)
            assert_quad_equal(result[size//2, size//2], expected_value, rtol=1e-15, atol=1e-15)
    
    def test_large_vector_operations(self):
        """Test large vector np.matmul products"""
        size = 1000
        
        # Create vectors with known sum
        x_vals = [1.0] * size
        y_vals = [2.0] * size
        
        x = create_quad_array(x_vals)
        y = create_quad_array(y_vals)
        
        result = np.matmul(x, y)
        expected = size * 1.0 * 2.0  # = 2000.0
        
        assert_quad_equal(result, expected)
    
    def test_rectangular_large_matrices(self):
        """Test large rectangular matrix operations"""
        m, n, k = 100, 80, 120
        
        # Create simple patterns
        A_vals = [(i + j + 1) % 10 for i in range(m) for j in range(k)]
        B_vals = [(i + j + 1) % 10 for i in range(k) for j in range(n)]
        
        A = create_quad_array(A_vals, shape=(m, k))
        B = create_quad_array(B_vals, shape=(k, n))
        
        result = np.matmul(A, B)
        
        assert result.shape == (m, n)
        
        # Verify that result doesn't contain NaN or inf
        result_flat = result.flatten()
        for i in range(min(10, len(result_flat))):  # Check first few elements
            val = float(result_flat[i])
            assert not np.isnan(val), f"NaN found at position {i}"
            assert not np.isinf(val), f"Inf found at position {i}"


# ================================================================================
# BASIC ERROR HANDLING
# ================================================================================

class TestBasicErrorHandling:
    """Test basic error conditions"""
    
    def test_dimension_mismatch_vectors(self):
        """Test dimension mismatch in vectors"""
        x = create_quad_array([1, 2])
        y = create_quad_array([1, 2, 3])
        
        with pytest.raises(ValueError, match=r"matmul: Input operand 1 has a mismatch in its core dimension 0"):
            np.matmul(x, y)
    
    def test_dimension_mismatch_matrix_vector(self):
        """Test dimension mismatch in matrix-vector"""
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        x = create_quad_array([1, 2, 3])  # Wrong size
        
        with pytest.raises(ValueError, match=r"matmul: Input operand 1 has a mismatch in its core dimension 0"):
            np.matmul(A, x)
    
    def test_dimension_mismatch_matrices(self):
        """Test dimension mismatch in matrix-matrix"""
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        B = create_quad_array([1, 2, 3, 4, 5, 6], shape=(3, 2))  # Wrong size
        
        with pytest.raises(ValueError, match=r"matmul: Input operand 1 has a mismatch in its core dimension 0"):
            np.matmul(A, B)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])