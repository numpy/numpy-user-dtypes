import pytest
import numpy as np
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
    
    def test_epsilon_values(self):
        """Test epsilon values (eps and epsneg)."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # eps = 2^-112 (difference between 1.0 and next larger representable float)
        expected_eps = QuadPrecision("1.92592994438723585305597794258492732e-34")
        assert isinstance(finfo.eps, QuadPrecision), f"eps should be QuadPrecision, got {type(finfo.eps)}"
        assert str(finfo.eps) == str(expected_eps), f"eps: expected {expected_eps}, got {finfo.eps}"
        
        # epsneg = 2^-113 (difference between 1.0 and next smaller representable float)
        expected_epsneg = QuadPrecision("9.62964972193617926527988971292463660e-35")
        assert isinstance(finfo.epsneg, QuadPrecision), f"epsneg should be QuadPrecision, got {type(finfo.epsneg)}"
        assert str(finfo.epsneg) == str(expected_epsneg), f"epsneg: expected {expected_epsneg}, got {finfo.epsneg}"
    
    def test_max_and_min_values(self):
        """Test maximum and minimum finite values."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # max = SLEEF_QUAD_MAX
        expected_max = QuadPrecision("1.18973149535723176508575932662800702e+4932")
        assert isinstance(finfo.max, QuadPrecision), f"max should be QuadPrecision, got {type(finfo.max)}"
        assert str(finfo.max) == str(expected_max), f"max: expected {expected_max}, got {finfo.max}"
        
        # min = -SLEEF_QUAD_MAX (most negative finite value)
        expected_min = QuadPrecision("-1.18973149535723176508575932662800702e+4932")
        assert isinstance(finfo.min, QuadPrecision), f"min should be QuadPrecision, got {type(finfo.min)}"
        assert str(finfo.min) == str(expected_min), f"min: expected {expected_min}, got {finfo.min}"
        
        # Verify min is negative
        assert str(finfo.min).startswith('-'), f"min should be negative, got {finfo.min}"
    
    def test_tiny_and_smallest_values(self):
        """Test tiny (smallest_normal) and smallest_subnormal values."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # tiny = smallest_normal = SLEEF_QUAD_MIN = 2^-16382
        expected_tiny = QuadPrecision("3.36210314311209350626267781732175260e-4932")
        assert isinstance(finfo.tiny, QuadPrecision), f"tiny should be QuadPrecision, got {type(finfo.tiny)}"
        assert isinstance(finfo.smallest_normal, QuadPrecision), \
            f"smallest_normal should be QuadPrecision, got {type(finfo.smallest_normal)}"
        
        # tiny and smallest_normal should be the same
        assert str(finfo.tiny) == str(finfo.smallest_normal), \
            f"tiny and smallest_normal should be equal, got {finfo.tiny} != {finfo.smallest_normal}"
        assert str(finfo.tiny) == str(expected_tiny), f"tiny: expected {expected_tiny}, got {finfo.tiny}"
        
        # smallest_subnormal = 2^-16494 (smallest positive representable number)
        expected_smallest_subnormal = QuadPrecision("6.47517511943802511092443895822764655e-4966")
        assert isinstance(finfo.smallest_subnormal, QuadPrecision), \
            f"smallest_subnormal should be QuadPrecision, got {type(finfo.smallest_subnormal)}"
        assert str(finfo.smallest_subnormal) == str(expected_smallest_subnormal), \
            f"smallest_subnormal: expected {expected_smallest_subnormal}, got {finfo.smallest_subnormal}"
    
    def test_resolution(self):
        """Test resolution property (10^-precision)."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # Resolution should be approximately 10^-33 for quad precision
        expected_resolution = QuadPrecision("1e-33")
        assert isinstance(finfo.resolution, QuadPrecision), \
            f"resolution should be QuadPrecision, got {type(finfo.resolution)}"
        assert str(finfo.resolution) == str(expected_resolution), \
            f"resolution: expected {expected_resolution}, got {finfo.resolution}"
    
    def test_dtype_property(self):
        """Test that finfo.dtype returns the correct dtype."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # finfo.dtype should return a QuadPrecDType
        assert isinstance(finfo.dtype, type(dtype)), \
            f"finfo.dtype should be QuadPrecDType, got {type(finfo.dtype)}"
    
    def test_machep_and_negep(self):
        """Test machep and negep exponent properties."""
        dtype = QuadPrecDType()
        finfo = np.finfo(dtype)
        
        # machep: exponent that yields eps (should be -112 for quad precision: 2^-112)
        # negep: exponent that yields epsneg (should be -113 for quad precision: 2^-113)
        # These are calculated by NumPy from eps and epsneg values
        
        # Just verify they exist and are integers
        assert isinstance(finfo.machep, (int, np.integer)), \
            f"machep should be integer, got {type(finfo.machep)}"
        assert isinstance(finfo.negep, (int, np.integer)), \
            f"negep should be integer, got {type(finfo.negep)}"
    
    def test_finfo_comparison_with_float64(self):
        """Verify quad precision has better precision than float64."""
        quad_finfo = np.finfo(QuadPrecDType())
        float64_finfo = np.finfo(np.float64)
        
        # Quad precision should have more precision
        assert quad_finfo.bits > 64, \
            f"Quad bits ({quad_finfo.bits}) should be > 64"
        assert quad_finfo.nmant > float64_finfo.nmant, \
            f"Quad nmant ({quad_finfo.nmant}) should be > float64 nmant ({float64_finfo.nmant})"
        assert quad_finfo.precision > float64_finfo.precision, \
            f"Quad precision ({quad_finfo.precision}) should be > float64 precision ({float64_finfo.precision})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
