# Plan for `numpy-quaddtype` v1.0.0

| ufunc name    | Added |
| ------------- | ----- |
| add           | ✅    |
| subtract      | ✅    |
| multiply      | ✅    |
| matmul        | #116  |
| divide        | ✅    |
| logaddexp     |       |
| logaddexp2    |       |
| true_divide   |       |
| floor_divide  |       |
| negative      | ✅    |
| positive      | ✅    |
| power         | ✅    |
| float_power   |       |
| remainder     |       |
| mod           | ✅    |
| fmod          |       |
| divmod        |       |
| absolute      | ✅    |
| fabs          |       |
| rint          | ✅    |
| sign          |       |
| heaviside     |       |
| conj          |       |
| conjugate     |       |
| exp           | ✅    |
| exp2          | ✅    |
| log           | ✅    |
| log2          | ✅    |
| log10         | ✅    |
| expm1         |       |
| log1p         | ✅    |
| sqrt          | ✅    |
| square        | ✅    |
| cbrt          |       |
| reciprocal    |       |
| gcd           |       |
| lcm           |       |
| sin           | ✅    |
| cos           | ✅    |
| tan           | ✅    |
| arcsin        | ✅    |
| arccos        | ✅    |
| arctan        | ✅    |
| arctan2       | ✅    |
| hypot         |       |
| sinh          |       |
| cosh          |       |
| tanh          |       |
| arcsinh       |       |
| arccosh       |       |
| arctanh       |       |
| degrees       |       |
| radians       |       |
| deg2rad       |       |
| rad2deg       |       |
| bitwise_and   |       |
| bitwise_or    |       |
| bitwise_xor   |       |
| invert        |       |
| left_shift    |       |
| right_shift   |       |
| greater       | ✅    |
| greater_equal | ✅    |
| less          | ✅    |
| less_equal    | ✅    |
| not_equal     | ✅    |
| equal         | ✅    |
| logical_and   |       |
| logical_or    |       |
| logical_xor   |       |
| logical_not   |       |
| maximum       | ✅    |
| minimum       | ✅    |
| fmax          |       |
| fmin          |       |
| isfinite      |       |
| isinf         |       |
| isnan         |       |
| isnat         |       |
| signbit       |       |
| copysign      |       |
| nextafter     |       |
| spacing       |       |
| modf          |       |
| ldexp         |       |
| frexp         |       |
| floor         | ✅    |
| ceil          | ✅    |
| trunc         | ✅    |

- Fixing QBLAS integration to work unaligned arrays without or recovering from bad allocation fallback
