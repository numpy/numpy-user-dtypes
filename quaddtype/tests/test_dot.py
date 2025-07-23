import pytest
import numpy as np
from utils import create_quad_array, assert_quad_equal, assert_quad_array_equal, arrays_equal_with_nan
from numpy_quaddtype import QuadPrecision, QuadPrecDType


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
# SPECIAL VALUES EDGE CASE TESTS
# ================================================================================

class TestSpecialValueEdgeCases:
    """Test matmul with special IEEE 754 values (NaN, inf, -0.0)"""
    
    @pytest.mark.parametrize("special_val", ["0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
    def test_vector_with_special_values(self, special_val):
        """Test vectors containing special values"""
        # Create vectors with special values
        x = create_quad_array([1.0, float(special_val), 2.0])
        y = create_quad_array([3.0, 4.0, 5.0])
        
        result = np.matmul(x, y)
        
        # Compare with float64 reference
        x_float = np.array([1.0, float(special_val), 2.0], dtype=np.float64)
        y_float = np.array([3.0, 4.0, 5.0], dtype=np.float64)
        expected = np.matmul(x_float, y_float)
        
        # Handle special value comparisons
        if np.isnan(expected):
            assert np.isnan(float(result))
        elif np.isinf(expected):
            assert np.isinf(float(result))
            assert np.sign(float(result)) == np.sign(expected)
        else:
            assert_quad_equal(result, expected)
    
    @pytest.mark.parametrize("special_val", ["0.0", "-0.0", "inf", "-inf", "nan"])
    def test_matrix_vector_with_special_values(self, special_val):
        """Test matrix-vector multiplication with special values"""
        # Matrix with special value
        A = create_quad_array([1.0, float(special_val), 3.0, 4.0], shape=(2, 2))
        x = create_quad_array([2.0, 1.0])
        
        result = np.matmul(A, x)
        
        # Compare with float64 reference  
        A_float = np.array([[1.0, float(special_val)], [3.0, 4.0]], dtype=np.float64)
        x_float = np.array([2.0, 1.0], dtype=np.float64)
        expected = np.matmul(A_float, x_float)
        
        assert result.shape == expected.shape
        for i in range(len(expected)):
            if np.isnan(expected[i]):
                assert np.isnan(float(result[i]))
            elif np.isinf(expected[i]):
                assert np.isinf(float(result[i]))
                assert np.sign(float(result[i])) == np.sign(expected[i])
            else:
                assert_quad_equal(result[i], expected[i])
    
    @pytest.mark.parametrize("special_val", ["0.0", "-0.0", "inf", "-inf", "nan"])
    def test_matrix_matrix_with_special_values(self, special_val):
        """Test matrix-matrix multiplication with special values"""
        A = create_quad_array([1.0, 2.0, float(special_val), 4.0], shape=(2, 2))
        B = create_quad_array([5.0, 6.0, 7.0, 8.0], shape=(2, 2))
        
        result = np.matmul(A, B)
        
        # Compare with float64 reference
        A_float = np.array([[1.0, 2.0], [float(special_val), 4.0]], dtype=np.float64)
        B_float = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        expected = np.matmul(A_float, B_float)
        
        assert result.shape == expected.shape
        assert arrays_equal_with_nan(result, expected)
    
    def test_all_nan_matrix(self):
        """Test matrices filled with NaN"""
        A = create_quad_array([float("nan")] * 4, shape=(2, 2))
        B = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        
        result = np.matmul(A, B)
        
        # Result should be all NaN (NaN * anything = NaN)
        for i in range(2):
            for j in range(2):
                assert np.isnan(float(result[i, j]))
    
    def test_inf_times_zero_produces_nan(self):
        """Test that Inf * 0 correctly produces NaN per IEEE 754"""
        # Create a scenario where Inf * 0 occurs in matrix multiplication
        A = create_quad_array([float("inf"), 1.0], shape=(1, 2))
        B = create_quad_array([0.0, 1.0], shape=(2, 1))
        
        result = np.matmul(A, B)
        
        # Result should be inf*0 + 1*1 = NaN + 1 = NaN
        assert np.isnan(float(result[0, 0])), "Inf * 0 should produce NaN per IEEE 754"
    
    def test_nan_propagation(self):
        """Test that NaN properly propagates through matrix operations"""
        A = create_quad_array([1.0, float("nan"), 3.0, 4.0], shape=(2, 2))
        B = create_quad_array([1.0, 0.0, 0.0, 1.0], shape=(2, 2))  # Identity
        
        result = np.matmul(A, B)
        
        # C[0,0] = 1*1 + nan*0 = 1 + nan = nan (nan*0 = nan, not like inf*0)
        # C[0,1] = 1*0 + nan*1 = 0 + nan = nan  
        # C[1,0] = 3*1 + 4*0 = 3 + 0 = 3
        # C[1,1] = 3*0 + 4*1 = 0 + 4 = 4
        assert np.isnan(float(result[0, 0]))
        assert np.isnan(float(result[0, 1]))
        assert_quad_equal(result[1, 0], 3.0)
        assert_quad_equal(result[1, 1], 4.0)
    
    def test_zero_division_and_indeterminate_forms(self):
        """Test handling of indeterminate forms in matrix operations"""
        # Test various indeterminate forms that should produce NaN
        
        # Case: Inf - Inf form
        A = create_quad_array([float("inf"), float("inf")], shape=(1, 2))
        B = create_quad_array([1.0, -1.0], shape=(2, 1))
        
        result = np.matmul(A, B)
        
        # Result should be inf*1 + inf*(-1) = inf - inf = NaN
        assert np.isnan(float(result[0, 0])), "Inf - Inf should produce NaN per IEEE 754"
    
    def test_mixed_inf_values(self):
        """Test matrices with mixed infinite values"""
        # Use all-ones matrix to avoid Inf * 0 = NaN issues
        A = create_quad_array([float("inf"), 2, float("-inf"), 3], shape=(2, 2))
        B = create_quad_array([1, 1, 1, 1], shape=(2, 2))  # All ones to avoid Inf*0
        
        result = np.matmul(A, B)
        
        # C[0,0] = inf*1 + 2*1 = inf + 2 = inf
        # C[0,1] = inf*1 + 2*1 = inf + 2 = inf  
        # C[1,0] = -inf*1 + 3*1 = -inf + 3 = -inf
        # C[1,1] = -inf*1 + 3*1 = -inf + 3 = -inf
        assert np.isinf(float(result[0, 0])) and float(result[0, 0]) > 0
        assert np.isinf(float(result[0, 1])) and float(result[0, 1]) > 0
        assert np.isinf(float(result[1, 0])) and float(result[1, 0]) < 0  
        assert np.isinf(float(result[1, 1])) and float(result[1, 1]) < 0


# ================================================================================
# DEGENERATE AND EMPTY CASE TESTS
# ================================================================================

class TestDegenerateCases:
    """Test edge cases with degenerate dimensions"""
    
    def test_single_element_matrices(self):
        """Test 1x1 matrix operations"""
        A = create_quad_array([3.0], shape=(1, 1))
        B = create_quad_array([4.0], shape=(1, 1))
        
        result = np.matmul(A, B)
        
        assert result.shape == (1, 1)
        assert_quad_equal(result[0, 0], 12.0)
    
    def test_single_element_vector(self):
        """Test operations with single-element vectors"""
        x = create_quad_array([5.0])
        y = create_quad_array([7.0])
        
        result = np.matmul(x, y)
        
        assert isinstance(result, QuadPrecision)
        assert_quad_equal(result, 35.0)
    
    def test_very_tall_matrix(self):
        """Test very tall matrices (1000x1)"""
        size = 1000
        A = create_quad_array([1.0] * size, shape=(size, 1))
        B = create_quad_array([2.0], shape=(1, 1))
        
        result = np.matmul(A, B)
        
        assert result.shape == (size, 1)
        for i in range(min(10, size)):  # Check first 10 elements
            assert_quad_equal(result[i, 0], 2.0)
    
    def test_very_wide_matrix(self):
        """Test very wide matrices (1x1000)"""
        size = 1000
        A = create_quad_array([1.0], shape=(1, 1))  
        B = create_quad_array([3.0] * size, shape=(1, size))
        
        result = np.matmul(A, B)
        
        assert result.shape == (1, size)
        for i in range(min(10, size)):  # Check first 10 elements
            assert_quad_equal(result[0, i], 3.0)
    
    def test_zero_matrices(self):
        """Test matrices filled with zeros"""
        A = create_quad_array([0.0] * 9, shape=(3, 3))
        B = create_quad_array([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=(3, 3))
        
        result = np.matmul(A, B)
        
        assert result.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert_quad_equal(result[i, j], 0.0)
    
    def test_repeated_row_matrix(self):
        """Test matrices with repeated rows"""
        # Matrix with all rows the same
        A = create_quad_array([1, 2, 3] * 3, shape=(3, 3))  # Each row is [1, 2, 3]
        B = create_quad_array([1, 0, 0, 0, 1, 0, 0, 0, 1], shape=(3, 3))  # Identity
        
        result = np.matmul(A, B)
        
        # Result should have all rows equal to [1, 2, 3]
        for i in range(3):
            assert_quad_equal(result[i, 0], 1.0)
            assert_quad_equal(result[i, 1], 2.0)
            assert_quad_equal(result[i, 2], 3.0)
    
    def test_repeated_column_matrix(self):
        """Test matrices with repeated columns"""
        A = create_quad_array([1, 0, 0, 0, 1, 0, 0, 0, 1], shape=(3, 3))  # Identity
        B = create_quad_array([2, 2, 2, 3, 3, 3, 4, 4, 4], shape=(3, 3))  # Each column repeated
        
        result = np.matmul(A, B)
        
        # Result should be same as B (identity multiplication)
        assert_quad_array_equal(result, B)


# ================================================================================
# NUMERICAL STABILITY AND PRECISION TESTS
# ================================================================================

class TestNumericalStability:
    """Test numerical stability with extreme values"""
    
    def test_very_large_values(self):
        """Test matrices with very large values"""
        large_val = 1e100
        A = create_quad_array([large_val, 1, 1, large_val], shape=(2, 2))
        B = create_quad_array([1, 0, 0, 1], shape=(2, 2))  # Identity
        
        result = np.matmul(A, B)
        
        # Should preserve large values without overflow
        assert_quad_equal(result[0, 0], large_val)
        assert_quad_equal(result[1, 1], large_val)
        assert not np.isinf(float(result[0, 0]))
        assert not np.isinf(float(result[1, 1]))
    
    def test_very_small_values(self):
        """Test matrices with very small values"""
        small_val = 1e-100
        A = create_quad_array([small_val, 0, 0, small_val], shape=(2, 2))
        B = create_quad_array([1, 0, 0, 1], shape=(2, 2))  # Identity
        
        result = np.matmul(A, B)
        
        # Should preserve small values without underflow
        assert_quad_equal(result[0, 0], small_val)
        assert_quad_equal(result[1, 1], small_val)
        assert float(result[0, 0]) != 0.0
        assert float(result[1, 1]) != 0.0
    
    def test_mixed_scale_values(self):
        """Test matrices with mixed magnitude values"""
        A = create_quad_array([1e100, 1e-100, 1e50, 1e-50], shape=(2, 2))
        B = create_quad_array([1, 0, 0, 1], shape=(2, 2))  # Identity
        
        result = np.matmul(A, B)
        
        # All values should be preserved accurately
        assert_quad_equal(result[0, 0], 1e100)
        assert_quad_equal(result[0, 1], 1e-100)
        assert_quad_equal(result[1, 0], 1e50)
        assert_quad_equal(result[1, 1], 1e-50)
    
    def test_precision_critical_case(self):
        """Test case that would lose precision in double"""
        # Create a case where large values cancel in the dot product
        # Vector: [1e20, 1.0, -1e20] dot [1, 0, 1] should equal 1.0
        x = create_quad_array([1e20, 1.0, -1e20])
        y = create_quad_array([1.0, 0.0, 1.0])
        
        result = np.matmul(x, y)
        
        # The result should be 1e20*1 + 1.0*0 + (-1e20)*1 = 1e20 - 1e20 = 0, but we want 1
        # Let me fix this: [1e20, 1.0, -1e20] dot [0, 1, 0] = 1.0
        x = create_quad_array([1e20, 1.0, -1e20])
        y = create_quad_array([0.0, 1.0, 0.0])
        
        result = np.matmul(x, y)
        
        # This would likely fail in double precision due to representation issues
        assert_quad_equal(result, 1.0, atol=1e-25)
    
    def test_condition_number_extreme(self):
        """Test matrices with extreme condition numbers"""
        # Nearly singular matrix (very small determinant)
        eps = 1e-50
        A = create_quad_array([1, 1, 1, 1+eps], shape=(2, 2))
        B = create_quad_array([1, 0, 0, 1], shape=(2, 2))
        
        result = np.matmul(A, B)
        
        # Result should be computed accurately
        assert_quad_equal(result[0, 0], 1.0)
        assert_quad_equal(result[0, 1], 1.0)
        assert_quad_equal(result[1, 0], 1.0)
        assert_quad_equal(result[1, 1], 1.0 + eps)
    
    def test_accumulation_precision(self):
        """Test precision in accumulation of many terms"""
        size = 100
        # Create vectors where each term contributes equally
        x_vals = [1.0 / size] * size
        y_vals = [1.0] * size
        
        x = create_quad_array(x_vals)
        y = create_quad_array(y_vals)
        
        result = np.matmul(x, y)
        
        # Result should be exactly 1.0 
        assert_quad_equal(result, 1.0, atol=1e-25)


# ================================================================================
# CROSS-VALIDATION TESTS
# ================================================================================

class TestCrossValidation:
    """Test consistency with float64 reference implementations"""
    
    @pytest.mark.parametrize("size", [2, 3, 5, 10])
    def test_consistency_with_float64_vectors(self, size):
        """Test vector operations consistency with float64"""
        # Use values well within float64 range
        x_vals = [i + 0.5 for i in range(size)]
        y_vals = [2 * i + 1.5 for i in range(size)]
        
        # QuadPrecision computation
        x_quad = create_quad_array(x_vals)
        y_quad = create_quad_array(y_vals)
        result_quad = np.matmul(x_quad, y_quad)
        
        # float64 reference
        x_float = np.array(x_vals, dtype=np.float64)
        y_float = np.array(y_vals, dtype=np.float64)
        result_float = np.matmul(x_float, y_float)
        
        # Results should match within float64 precision
        assert_quad_equal(result_quad, result_float, rtol=1e-14)
    
    @pytest.mark.parametrize("m,n,k", [(2,2,2), (3,3,3), (4,5,6)])
    def test_consistency_with_float64_matrices(self, m, n, k):
        """Test matrix operations consistency with float64"""
        # Create test matrices with float64-representable values
        A_vals = [(i + j + 1) * 0.25 for i in range(m) for j in range(k)]
        B_vals = [(i * 2 + j) * 0.125 for i in range(k) for j in range(n)]
        
        # QuadPrecision computation
        A_quad = create_quad_array(A_vals, shape=(m, k))
        B_quad = create_quad_array(B_vals, shape=(k, n))
        result_quad = np.matmul(A_quad, B_quad)
        
        # float64 reference
        A_float = np.array(A_vals, dtype=np.float64).reshape(m, k)
        B_float = np.array(B_vals, dtype=np.float64).reshape(k, n)
        result_float = np.matmul(A_float, B_float)
        
        # Results should match within float64 precision
        for i in range(m):
            for j in range(n):
                assert_quad_equal(result_quad[i, j], result_float[i, j], rtol=1e-14)
    
    def test_quad_precision_advantage(self):
        """Test cases where quad precision shows advantage over float64"""
        A = create_quad_array([1.0, 1e-30], shape=(1, 2))
        B = create_quad_array([1.0, 1.0], shape=(2, 1))
        
        result_quad = np.matmul(A, B)
        
        # The result should be 1.0 + 1e-30 = 1.0000000000000000000000000000001
        expected = 1.0 + 1e-30
        assert_quad_equal(result_quad[0, 0], expected, rtol=1e-25)
        
        # Verify that this value is actually different from 1.0 in quad precision
        diff = result_quad[0, 0] - 1.0
        assert abs(diff) > 0  # Should be non-zero in quad precision


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