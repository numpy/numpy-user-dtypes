# Plan for `numpy-quaddtype` v1.0.0
- [ ] High-Endian System support
- [ ] Complete Documentation

| ufunc name    | Added | Edge Cases Tested\*                                                     |
| ------------- | ----- | ----------------------------------------------------------------------- |
| add           | ✅    | ✅                                                                      |
| subtract      | ✅    | ✅                                                                      |
| multiply      | ✅    | ✅                                                                      |
| matmul        | ✅    | ✅                                                                      |
| divide        | ✅    | ✅                                                                      |
| logaddexp     |       |                                                                         |
| logaddexp2    |       |                                                                         |
| true_divide   |       |                                                                         |
| floor_divide  |       |                                                                         |
| negative      | ✅    | ✅                                                                      |
| positive      | ✅    | ✅                                                                      |
| power         | ✅    | ✅                                                                      |
| float_power   |       |                                                                         |
| remainder     |       |                                                                         |
| mod           | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/-0.0/large values)_         |
| fmod          |       |                                                                         |
| divmod        |       |                                                                         |
| absolute      | ✅    | ✅                                                                      |
| fabs          |       |                                                                         |
| rint          | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/±0.0/halfway cases)_        |
| sign          | ✅    | ✅                                                                      |
| heaviside     |       |                                                                         |
| conj          |       |                                                                         |
| conjugate     |       |                                                                         |
| exp           | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/large +/- values/overflow)_ |
| exp2          | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/large +/- values/overflow)_ |
| log           | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/0/-values/1)_               |
| log2          | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/0/-values/1)_               |
| log10         | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/0/-values/1)_               |
| expm1         |       |                                                                         |
| log1p         | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/-1/small values)_           |
| sqrt          | ✅    | ✅                                                                      |
| square        | ✅    | ✅                                                                      |
| cbrt          |       |                                                                         |
| reciprocal    | ✅    | ✅                                                                      |
| gcd           |       |                                                                         |
| lcm           |       |                                                                         |
| sin           | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/0/π multiples/2π range)_    |
| cos           | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/0/π multiples/2π range)_    |
| tan           | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/0/π/2 asymptotes)_          |
| arcsin        | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/±1/out-of-domain)_          |
| arccos        | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/±1/out-of-domain)_          |
| arctan        | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/0/asymptotes)_              |
| arctan2       | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/0/quadrant coverage)_       |
| hypot         |       |                                                                         |
| sinh          |       |                                                                         |
| cosh          |       |                                                                         |
| tanh          |       |                                                                         |
| arcsinh       |       |                                                                         |
| arccosh       |       |                                                                         |
| arctanh       |       |                                                                         |
| degrees       |       |                                                                         |
| radians       |       |                                                                         |
| deg2rad       |       |                                                                         |
| rad2deg       |       |                                                                         |
| bitwise_and   |       |                                                                         |
| bitwise_or    |       |                                                                         |
| bitwise_xor   |       |                                                                         |
| invert        |       |                                                                         |
| left_shift    |       |                                                                         |
| right_shift   |       |                                                                         |
| greater       | ✅    | ✅                                                                      |
| greater_equal | ✅    | ✅                                                                      |
| less          | ✅    | ✅                                                                      |
| less_equal    | ✅    | ✅                                                                      |
| not_equal     | ✅    | ✅                                                                      |
| equal         | ✅    | ✅                                                                      |
| logical_and   |       |                                                                         |
| logical_or    |       |                                                                         |
| logical_xor   |       |                                                                         |
| logical_not   |       |                                                                         |
| maximum       | ✅    | ✅                                                                      |
| minimum       | ✅    | ✅                                                                      |
| fmax          | ✅    | ✅                                                                      |
| fmin          | ✅    | ✅                                                                      |
| isfinite      | ✅    | ✅                                                                      |
| isinf         | ✅    | ✅                                                                      |
| isnan         | ✅    | ✅                                                                      |
| isnat         |       |                                                                         |
| signbit       | ✅    | ✅                                                                      |
| copysign      | ✅    | ✅                                                                      |
| nextafter     |       |                                                                         |
| spacing       |       |                                                                         |
| modf          |       |                                                                         |
| ldexp         |       |                                                                         |
| frexp         |       |                                                                         |
| floor         | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/±0.0/halfway values)_       |
| ceil          | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/±0.0/halfway values)_       |
| trunc         | ✅    | ❌ _Need: basic tests + edge cases (NaN/inf/±0.0/fractional values)_    |

\* **Edge Cases Tested**: Indicates whether the ufunc has parametrized tests that compare QuadPrecision results against `float` and `np.float64` for edge cases including:

- Special values: `0.0`, `-0.0`, `inf`, `-inf`, `nan`, `-nan`
- For trigonometric functions: Critical points like `0`, `π/2`, `π`, `3π/2`, `2π`, values in `[0, 2π]`
- For logarithmic functions: Values near `0`, `1`, large values
- For exponential functions: Large positive/negative values, values near `0`

**Testing Status:**

- ✅ = Comprehensive edge case tests exist in `test_quaddtype.py` with parametrized tests against float64
- 🟡 = Good basic testing exists but missing some edge cases (specific missing tests noted in italics)
- ❌ = Ufunc is implemented but lacks systematic testing (required tests noted in italics)
- (blank) = Ufunc not yet implemented (implementation needed first)
