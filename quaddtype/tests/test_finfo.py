import pytest
import numpy as np
from utils import assert_quad_equal as assert_quad_close
from numpy_quaddtype import QuadPrecDType, QuadPrecision


class TestFinfoConstants:
    """Test suite for verifying all finfo constants are correctly implemented."""
    
    def test_basic_integer_properties(self):
        """Test basic integer properties of finfo."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # Test basic properties
        assert finfo.bits == 128, f"Expected bits=128, got {finfo.bits}"
        assert finfo.nmant == 112, f"Expected nmant=112 (mantissa bits), got {finfo.nmant}"
        assert finfo.nexp == 15, f"Expected nexp=15 (exponent bits including sign and bias), got {finfo.nexp}"
        assert finfo.iexp == 15, f"Expected iexp=15 (exponent bits), got {finfo.iexp}"
        assert finfo.minexp == -16382, f"Expected minexp=-16382, got {finfo.minexp}"
        assert finfo.maxexp == 16384, f"Expected maxexp=16384, got {finfo.maxexp}"
        assert finfo.precision == 33, f"Expected precision=33 (decimal digits), got {finfo.precision}"
        assert finfo.machep == -112, f"Expected machep=-112, got {finfo.machep}"
        assert finfo.negep == -113, f"Expected negep=-113, got {finfo.negep}"
    
    def test_epsilon_values(self):
        """Test epsilon values (eps and epsneg)."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # eps = 2^-112 (difference between 1.0 and next larger representable float)
        assert isinstance(finfo.eps, QuadPrecision), f"eps should be QuadPrecision, got {type(finfo.eps)}"
        assert_quad_close(finfo.eps, "1.925929944387235853055977942584927e-034")
        
        # epsneg = 2^-113 (difference between 1.0 and next smaller representable float)
        assert isinstance(finfo.epsneg, QuadPrecision), f"epsneg should be QuadPrecision, got {type(finfo.epsneg)}"
        assert_quad_close(finfo.epsneg, "9.629649721936179265279889712924637e-035")
    
    def test_max_and_min_values(self):
        """Test maximum and minimum finite values."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # max = SLEEF_QUAD_MAX
        assert isinstance(finfo.max, QuadPrecision), f"max should be QuadPrecision, got {type(finfo.max)}"
        assert_quad_close(finfo.max, "1.189731495357231765085759326628007e+4932")
        
        # min = -SLEEF_QUAD_MAX (most negative finite value)
        assert isinstance(finfo.min, QuadPrecision), f"min should be QuadPrecision, got {type(finfo.min)}"
        assert_quad_close(finfo.min, "-1.189731495357231765085759326628007e+4932")
        
        # Verify min is negative
        zero = QuadPrecision("0")
        assert finfo.min < zero, f"min should be negative, got {finfo.min}"
    
    def test_tiny_and_smallest_values(self):
        """Test tiny (smallest_normal) and smallest_subnormal values."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # tiny = smallest_normal = SLEEF_QUAD_MIN = 2^-16382
        assert isinstance(finfo.tiny, QuadPrecision), f"tiny should be QuadPrecision, got {type(finfo.tiny)}"
        assert isinstance(finfo.smallest_normal, QuadPrecision), \
            f"smallest_normal should be QuadPrecision, got {type(finfo.smallest_normal)}"
        
        # tiny and smallest_normal should be the same
        assert finfo.tiny == finfo.smallest_normal, \
            f"tiny and smallest_normal should be equal, got {finfo.tiny} != {finfo.smallest_normal}"
        assert_quad_close(finfo.tiny, "3.362103143112093506262677817321753e-4932")
        
        # smallest_subnormal = 2^-16494 (smallest positive representable number)
        assert isinstance(finfo.smallest_subnormal, QuadPrecision), \
            f"smallest_subnormal should be QuadPrecision, got {type(finfo.smallest_subnormal)}"
        assert_quad_close(finfo.smallest_subnormal, "6.0e-4966")
    
    def test_resolution(self):
        """Test resolution property (10^-precision)."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # Resolution should be approximately 10^-33 for quad precision
        assert isinstance(finfo.resolution, QuadPrecision), \
            f"resolution should be QuadPrecision, got {type(finfo.resolution)}"
        assert_quad_close(finfo.resolution, "1.0e-33")
    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
