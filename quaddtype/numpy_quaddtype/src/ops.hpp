#include <sleef.h>
#include <sleefquad.h>
#include <cmath>
#include "constants.hpp"

// Unary Quad Operations
typedef Sleef_quad (*unary_op_quad_def)(const Sleef_quad *);
// Unary Quad operations with 2 outputs (for modf, frexp)
typedef void (*unary_op_2out_quad_def)(const Sleef_quad *, Sleef_quad *, Sleef_quad *);

static inline Sleef_quad
quad_negative(const Sleef_quad *op)
{
    return Sleef_negq1(*op);
}

static inline Sleef_quad
quad_positive(const Sleef_quad *op)
{
    return *op;
}

static inline Sleef_quad
quad_sign(const Sleef_quad *op)
{
    int sign = Sleef_icmpq1(*op, QUAD_PRECISION_ZERO);
    // sign(x=NaN) = x; otherwise sign(x) in { -1.0; 0.0; +1.0 }
    return Sleef_iunordq1(*op, *op) ? *op : Sleef_cast_from_int64q1(sign);
}

static inline Sleef_quad
quad_absolute(const Sleef_quad *op)
{
    return Sleef_fabsq1(*op);
}

static inline Sleef_quad
quad_conjugate(const Sleef_quad *op)
{
    // For real numbers, conjugate is the identity function (no-op)
    return *op;
}

static inline Sleef_quad
quad_rint(const Sleef_quad *op)
{
    Sleef_quad halfway = Sleef_addq1_u05(
        Sleef_truncq1(*op),
        Sleef_copysignq1(Sleef_cast_from_doubleq1(0.5), *op)
    );

    // Sleef_rintq1 does not handle some near-halfway cases correctly, so we
    // manually round up or down when x is not exactly halfway
    return Sleef_icmpeqq1(*op, halfway) ? Sleef_rintq1(*op) : (
        Sleef_icmpleq1(*op, halfway) ? Sleef_floorq1(*op) : Sleef_ceilq1(*op)
    );
}

static inline Sleef_quad
quad_trunc(const Sleef_quad *op)
{
    return Sleef_truncq1(*op);
}

static inline Sleef_quad
quad_floor(const Sleef_quad *op)
{
    return Sleef_floorq1(*op);
}

static inline Sleef_quad
quad_ceil(const Sleef_quad *op)
{
    return Sleef_ceilq1(*op);
}

static inline Sleef_quad
quad_sqrt(const Sleef_quad *op)
{
    return Sleef_sqrtq1_u05(*op);
}

static inline Sleef_quad
quad_cbrt(const Sleef_quad *op)
{
    // SLEEF doesn't provide cbrt, so we implement it using pow
    // cbrt(x) = x^(1/3)
    // For negative values: cbrt(-x) = -cbrt(x)
    
    // Handle special cases
    if (Sleef_iunordq1(*op, *op)) {
        return *op;  // NaN
    }
    if (Sleef_icmpeqq1(*op, QUAD_PRECISION_ZERO)) {
        return *op;  // ±0
    }
    // Check if op is ±inf: isinf(x) = abs(x) == inf
    if (Sleef_icmpeqq1(Sleef_fabsq1(*op), QUAD_PRECISION_INF)) {
        return *op;  // ±inf
    }
    
    // Compute 1/3 as a quad precision constant
    Sleef_quad three = Sleef_cast_from_int64q1(3);
    Sleef_quad one_third = Sleef_divq1_u05(QUAD_PRECISION_ONE, three);
    
    // Handle negative values: cbrt(-x) = -cbrt(x)
    if (Sleef_icmpltq1(*op, QUAD_PRECISION_ZERO)) {
        Sleef_quad abs_val = Sleef_fabsq1(*op);
        Sleef_quad result = Sleef_powq1_u10(abs_val, one_third);
        return Sleef_negq1(result);
    }
    
    // Positive values
    return Sleef_powq1_u10(*op, one_third);
}

static inline Sleef_quad
quad_square(const Sleef_quad *op)
{
    return Sleef_mulq1_u05(*op, *op);
}

static inline Sleef_quad
quad_reciprocal(const Sleef_quad *op)
{
    return Sleef_divq1_u05(QUAD_PRECISION_ONE, *op);
}

static inline Sleef_quad
quad_log(const Sleef_quad *op)
{
    return Sleef_logq1_u10(*op);
}

static inline Sleef_quad
quad_log2(const Sleef_quad *op)
{
    return Sleef_log2q1_u10(*op);
}

static inline Sleef_quad
quad_log10(const Sleef_quad *op)
{
    return Sleef_log10q1_u10(*op);
}

static inline Sleef_quad
quad_log1p(const Sleef_quad *op)
{
    return Sleef_log1pq1_u10(*op);
}

static inline Sleef_quad
quad_exp(const Sleef_quad *op)
{
    return Sleef_expq1_u10(*op);
}

static inline Sleef_quad
quad_exp2(const Sleef_quad *op)
{
    return Sleef_exp2q1_u10(*op);
}

static inline Sleef_quad
quad_expm1(const Sleef_quad *op)
{
    return Sleef_expm1q1_u10(*op);
}

static inline Sleef_quad
quad_sin(const Sleef_quad *op)
{
    return Sleef_sinq1_u10(*op);
}

static inline Sleef_quad
quad_cos(const Sleef_quad *op)
{
    return Sleef_cosq1_u10(*op);
}

static inline Sleef_quad
quad_tan(const Sleef_quad *op)
{
    return Sleef_tanq1_u10(*op);
}

static inline Sleef_quad
quad_asin(const Sleef_quad *op)
{
    return Sleef_asinq1_u10(*op);
}

static inline Sleef_quad
quad_acos(const Sleef_quad *op)
{
    return Sleef_acosq1_u10(*op);
}

static inline Sleef_quad
quad_atan(const Sleef_quad *op)
{
    return Sleef_atanq1_u10(*op);
}

static inline Sleef_quad
quad_sinh(const Sleef_quad *op)
{
    return Sleef_sinhq1_u10(*op);
}

static inline Sleef_quad
quad_cosh(const Sleef_quad *op)
{
    return Sleef_coshq1_u10(*op);
}

static inline Sleef_quad
quad_tanh(const Sleef_quad *op)
{
    return Sleef_tanhq1_u10(*op);
}

static inline Sleef_quad
quad_asinh(const Sleef_quad *op)
{
    return Sleef_asinhq1_u10(*op);
}

static inline Sleef_quad
quad_acosh(const Sleef_quad *op)
{
    return Sleef_acoshq1_u10(*op);
}

static inline Sleef_quad
quad_atanh(const Sleef_quad *op)
{
    return Sleef_atanhq1_u10(*op);
}

static inline Sleef_quad
quad_degrees(const Sleef_quad *op)
{
    // degrees = radians * 180 / π
    static const Sleef_quad one_eighty = sleef_q(+0x1680000000000LL, 0x0000000000000000ULL, 7); // 180.0 in quad
    Sleef_quad ratio = Sleef_divq1_u05(one_eighty, SLEEF_M_PIq);
    return Sleef_mulq1_u05(*op, ratio);
}

static inline Sleef_quad
quad_radians(const Sleef_quad *op)
{
    // radians = degrees * π / 180
    static const Sleef_quad one_eighty = sleef_q(+0x1680000000000LL, 0x0000000000000000ULL, 7);
    Sleef_quad ratio = Sleef_divq1_u05(SLEEF_M_PIq, one_eighty);
    return Sleef_mulq1_u05(*op, ratio);
}

// Unary long double operations
typedef long double (*unary_op_longdouble_def)(const long double *);

static inline long double
ld_negative(const long double *op)
{
    return -(*op);
}

static inline long double
ld_positive(const long double *op)
{
    return *op;
}

static inline long double
ld_absolute(const long double *op)
{
    return fabsl(*op);
}

static inline long double
ld_conjugate(const long double *op)
{
    // For real numbers, conjugate is the identity function (no-op)
    return *op;
}

static inline long double
ld_sign(const long double *op)
{
    if (*op < 0.0)
        return -1.0;
    if (*op == 0.0)
        return 0.0;
    if (*op > 0.0)
        return 1.0;
    // sign(x=NaN) = x
    return *op;
}

static inline long double
ld_rint(const long double *op)
{
    return rintl(*op);
}

static inline long double
ld_trunc(const long double *op)
{
    return truncl(*op);
}

static inline long double
ld_floor(const long double *op)
{
    return floorl(*op);
}

static inline long double
ld_ceil(const long double *op)
{
    return ceill(*op);
}

static inline long double
ld_sqrt(const long double *op)
{
    return sqrtl(*op);
}

static inline long double
ld_cbrt(const long double *op)
{
    return cbrtl(*op);
}

static inline long double
ld_square(const long double *op)
{
    return (*op) * (*op);
}

static inline long double
ld_reciprocal(const long double *op)
{
    return 1.0L / (*op);
}

static inline long double
ld_log(const long double *op)
{
    return logl(*op);
}

static inline long double
ld_log2(const long double *op)
{
    return log2l(*op);
}

static inline long double
ld_log10(const long double *op)
{
    return log10l(*op);
}

static inline long double
ld_log1p(const long double *op)
{
    return log1pl(*op);
}

static inline long double
ld_exp(const long double *op)
{
    return expl(*op);
}

static inline long double
ld_exp2(const long double *op)
{
    return exp2l(*op);
}

static inline long double
ld_expm1(const long double *op)
{
    return expm1l(*op);
}

static inline long double
ld_sin(const long double *op)
{
    return sinl(*op);
}

static inline long double
ld_cos(const long double *op)
{
    return cosl(*op);
}

static inline long double
ld_tan(const long double *op)
{
    return tanl(*op);
}

static inline long double
ld_asin(const long double *op)
{
    return asinl(*op);
}

static inline long double
ld_acos(const long double *op)
{
    return acosl(*op);
}

static inline long double
ld_atan(const long double *op)
{
    return atanl(*op);
}

static inline long double
ld_sinh(const long double *op)
{
    return sinhl(*op);
}

static inline long double
ld_cosh(const long double *op)
{
    return coshl(*op);
}

static inline long double
ld_tanh(const long double *op)
{
    return tanhl(*op);
}

static inline long double
ld_asinh(const long double *op)
{
    return asinhl(*op);
}

static inline long double
ld_acosh(const long double *op)
{
    return acoshl(*op);
}

static inline long double
ld_atanh(const long double *op)
{
    return atanhl(*op);
}

static inline long double
ld_degrees(const long double *op)
{
    // degrees = radians * 180 / π
    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif
    return (*op) * (180.0L / static_cast<long double>(M_PI));
}

static inline long double
ld_radians(const long double *op)
{
    // radians = degrees * π / 180
    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif
    return (*op) * (static_cast<long double>(M_PI) / 180.0L);
}

// Unary Quad properties
typedef npy_bool (*unary_prop_quad_def)(const Sleef_quad *);

static inline npy_bool
quad_signbit(const Sleef_quad *op)
{
    // FIXME @juntyr or @SwayamInSync: replace with binary implementation
    //  once we test big and little endian in CI
    Sleef_quad one_signed = Sleef_copysignq1(QUAD_PRECISION_ONE, *op);
    // signbit(x) = 1 iff copysign(1, x) == -1
    return Sleef_icmpltq1(one_signed, QUAD_PRECISION_ZERO);
}

static inline npy_bool
quad_isfinite(const Sleef_quad *op)
{
    // isfinite(x) = abs(x) < inf
    return Sleef_icmpltq1(Sleef_fabsq1(*op), QUAD_PRECISION_INF);
}

static inline npy_bool
quad_isinf(const Sleef_quad *op)
{
    // isinf(x) = abs(x) == inf
    return Sleef_icmpeqq1(Sleef_fabsq1(*op), QUAD_PRECISION_INF);
}

static inline npy_bool
quad_isnan(const Sleef_quad *op)
{
    return Sleef_iunordq1(*op, *op);
}

// Unary long double properties
typedef npy_bool (*unary_prop_longdouble_def)(const long double *);

static inline npy_bool
ld_signbit(const long double *op)
{
    return signbit(*op);
}

static inline npy_bool
ld_isfinite(const long double *op)
{
    return isfinite(*op);
}

static inline npy_bool
ld_isinf(const long double *op)
{
    return isinf(*op);
}

static inline npy_bool
ld_isnan(const long double *op)
{
    return isnan(*op);
}

// Binary Quad operations
typedef Sleef_quad (*binary_op_quad_def)(const Sleef_quad *, const Sleef_quad *);
// Binary Quad operations with 2 outputs (for divmod, modf, frexp)
typedef void (*binary_op_2out_quad_def)(const Sleef_quad *, const Sleef_quad *, Sleef_quad *, Sleef_quad *);

static inline Sleef_quad
quad_add(const Sleef_quad *in1, const Sleef_quad *in2)
{
    return Sleef_addq1_u05(*in1, *in2);
}

static inline Sleef_quad
quad_sub(const Sleef_quad *in1, const Sleef_quad *in2)
{
    return Sleef_subq1_u05(*in1, *in2);
}

static inline Sleef_quad
quad_mul(const Sleef_quad *a, const Sleef_quad *b)
{
    return Sleef_mulq1_u05(*a, *b);
}

static inline Sleef_quad
quad_div(const Sleef_quad *a, const Sleef_quad *b)
{
    return Sleef_divq1_u05(*a, *b);
}

static inline Sleef_quad
quad_floor_divide(const Sleef_quad *a, const Sleef_quad *b)
{
    // Handle NaN inputs
    if (Sleef_iunordq1(*a, *b)) {
        return Sleef_iunordq1(*a, *a) ? *a : *b;
    }
    
    // inf / finite_nonzero or -inf / finite_nonzero -> NaN
    // But inf / 0 -> inf
    if (quad_isinf(a) && quad_isfinite(b) && !Sleef_icmpeqq1(*b, QUAD_PRECISION_ZERO)) {
        return QUAD_PRECISION_NAN;
    }
    
    // 0 / 0 (including -0.0 / 0.0, 0.0 / -0.0, -0.0 / -0.0) -> NaN
    if (Sleef_icmpeqq1(*a, QUAD_PRECISION_ZERO) && Sleef_icmpeqq1(*b, QUAD_PRECISION_ZERO)) {
        return QUAD_PRECISION_NAN;
    }
    
    Sleef_quad quotient = Sleef_divq1_u05(*a, *b);
    Sleef_quad result = Sleef_floorq1(quotient);
    
    // floor_divide semantics: when result is -0.0 from non-zero numerator, convert to -1.0
    // This happens when: (negative & non-zero)/+inf, (positive & non-zero)/-inf
    // But NOT when numerator is ±0.0 (then result stays as ±0.0)
    if (Sleef_icmpeqq1(result, QUAD_PRECISION_ZERO) && quad_signbit(&result) && 
        !Sleef_icmpeqq1(*a, QUAD_PRECISION_ZERO)) {
        return Sleef_negq1(QUAD_PRECISION_ONE);  // -1.0
    }
    
    return result;
}

static inline Sleef_quad
quad_pow(const Sleef_quad *a, const Sleef_quad *b)
{
    return Sleef_powq1_u10(*a, *b);
}

static inline Sleef_quad
quad_mod(const Sleef_quad *a, const Sleef_quad *b)
{
    // division by zero
    if (Sleef_icmpeqq1(*b, QUAD_PRECISION_ZERO)) {
        return QUAD_PRECISION_NAN;
    }

    // NaN inputs
    if (Sleef_iunordq1(*a, *b)) {
        return Sleef_iunordq1(*a, *a) ? *a : *b;  // Return the NaN
    }

    // infinity dividend -> NaN
    if (quad_isinf(a)) {
        return QUAD_PRECISION_NAN;
    }

    // finite % inf
    if (quad_isfinite(a) && quad_isinf(b)) {
        int sign_a = quad_signbit(a);
        int sign_b = quad_signbit(b);

        // return a if sign_a == sign_b
        return (sign_a == sign_b) ? *a : *b;
    }

    // NumPy mod formula: a % b = a - floor(a/b) * b
    Sleef_quad quotient = Sleef_divq1_u05(*a, *b);
    Sleef_quad floored = Sleef_floorq1(quotient);
    Sleef_quad product = Sleef_mulq1_u05(floored, *b);
    Sleef_quad result = Sleef_subq1_u05(*a, product);

    // Handle zero result sign: when result is exactly zero,
    // it should have the same sign as the divisor (NumPy convention)
    if (Sleef_icmpeqq1(result, QUAD_PRECISION_ZERO)) {
        if (Sleef_icmpltq1(*b, QUAD_PRECISION_ZERO)) {
            return Sleef_negq1(QUAD_PRECISION_ZERO);  // -0.0
        }
        else {
            return QUAD_PRECISION_ZERO;  // +0.0
        }
    }

    return result;
}

static inline Sleef_quad
quad_fmod(const Sleef_quad *a, const Sleef_quad *b)
{
    // Handle NaN inputs
    if (Sleef_iunordq1(*a, *b)) {
        return Sleef_iunordq1(*a, *a) ? *a : *b;
    }
    
    // Division by zero -> NaN
    if (Sleef_icmpeqq1(*b, QUAD_PRECISION_ZERO)) {
        return QUAD_PRECISION_NAN;
    }
    
    // Infinity dividend -> NaN
    if (quad_isinf(a)) {
        return QUAD_PRECISION_NAN;
    }
    
    // Finite % infinity -> return dividend (same as a)
    if (quad_isfinite(a) && quad_isinf(b)) {
        return *a;
    }
    
    // x - trunc(x/y) * y
    Sleef_quad result = Sleef_fmodq1(*a, *b);
    
    if (Sleef_icmpeqq1(result, QUAD_PRECISION_ZERO)) {
        // Preserve sign of dividend (first argument)
        Sleef_quad sign_test = Sleef_copysignq1(QUAD_PRECISION_ONE, *a);
        if (Sleef_icmpltq1(sign_test, QUAD_PRECISION_ZERO)) {
            return Sleef_negq1(QUAD_PRECISION_ZERO);  // -0.0
        }
        else {
            return QUAD_PRECISION_ZERO;  // +0.0
        }
    }
    
    return result;
}

static inline void
quad_divmod(const Sleef_quad *a, const Sleef_quad *b, 
            Sleef_quad *out_quotient, Sleef_quad *out_remainder)
{
    *out_quotient = quad_floor_divide(a, b);
    *out_remainder = quad_mod(a, b);
}

static inline Sleef_quad
quad_minimum(const Sleef_quad *in1, const Sleef_quad *in2)
{
    if (Sleef_iunordq1(*in1, *in2)) {
        return Sleef_iunordq1(*in1, *in1) ? *in1 : *in2;
    }
    // minimum(-0.0, +0.0) = -0.0
    if (Sleef_icmpeqq1(*in1, QUAD_PRECISION_ZERO) && Sleef_icmpeqq1(*in2, QUAD_PRECISION_ZERO)) {
        return Sleef_icmpleq1(Sleef_copysignq1(QUAD_PRECISION_ONE, *in1), Sleef_copysignq1(QUAD_PRECISION_ONE, *in2)) ? *in1 : *in2;
    }
    return Sleef_fminq1(*in1, *in2);
}

static inline Sleef_quad
quad_maximum(const Sleef_quad *in1, const Sleef_quad *in2)
{
    if (Sleef_iunordq1(*in1, *in2)) {
        return Sleef_iunordq1(*in1, *in1) ? *in1 : *in2;
    }
    // maximum(-0.0, +0.0) = +0.0
    if (Sleef_icmpeqq1(*in1, QUAD_PRECISION_ZERO) && Sleef_icmpeqq1(*in2, QUAD_PRECISION_ZERO)) {
        return Sleef_icmpgeq1(Sleef_copysignq1(QUAD_PRECISION_ONE, *in1), Sleef_copysignq1(QUAD_PRECISION_ONE, *in2)) ? *in1 : *in2;
    }
    return Sleef_fmaxq1(*in1, *in2);
}

static inline Sleef_quad
quad_fmin(const Sleef_quad *in1, const Sleef_quad *in2)
{
    if (Sleef_iunordq1(*in1, *in2)) {
        return Sleef_iunordq1(*in2, *in2) ? *in1 : *in2;
    }
    // fmin(-0.0, +0.0) = -0.0
    if (Sleef_icmpeqq1(*in1, QUAD_PRECISION_ZERO) && Sleef_icmpeqq1(*in2, QUAD_PRECISION_ZERO)) {
        return Sleef_icmpleq1(Sleef_copysignq1(QUAD_PRECISION_ONE, *in1), Sleef_copysignq1(QUAD_PRECISION_ONE, *in2)) ? *in1 : *in2;
    }
    return Sleef_fminq1(*in1, *in2);
}

static inline Sleef_quad
quad_fmax(const Sleef_quad *in1, const Sleef_quad *in2)
{
    if (Sleef_iunordq1(*in1, *in2)) {
        return Sleef_iunordq1(*in2, *in2) ? *in1 : *in2;
    }
    // maximum(-0.0, +0.0) = +0.0
    if (Sleef_icmpeqq1(*in1, QUAD_PRECISION_ZERO) && Sleef_icmpeqq1(*in2, QUAD_PRECISION_ZERO)) {
        return Sleef_icmpgeq1(Sleef_copysignq1(QUAD_PRECISION_ONE, *in1), Sleef_copysignq1(QUAD_PRECISION_ONE, *in2)) ? *in1 : *in2;
    }
    return Sleef_fmaxq1(*in1, *in2);
}

static inline Sleef_quad
quad_atan2(const Sleef_quad *in1, const Sleef_quad *in2)
{
    return Sleef_atan2q1_u10(*in1, *in2);
}

static inline Sleef_quad
quad_copysign(const Sleef_quad *in1, const Sleef_quad *in2)
{
    return Sleef_copysignq1(*in1, *in2);
}

static inline Sleef_quad
quad_logaddexp(const Sleef_quad *x, const Sleef_quad *y)
{
    // logaddexp(x, y) = log(exp(x) + exp(y))
    // Numerically stable implementation: max(x, y) + log1p(exp(-abs(x - y)))
    
    // Handle NaN
    if (Sleef_iunordq1(*x, *y)) {
        return Sleef_iunordq1(*x, *x) ? *x : *y;
    }
    
    // Handle infinities
    // If both are -inf, result is -inf
    Sleef_quad neg_inf = Sleef_negq1(QUAD_PRECISION_INF);
    if (Sleef_icmpeqq1(*x, neg_inf) && Sleef_icmpeqq1(*y, neg_inf)) {
        return neg_inf;
    }
    
    // If either is +inf, result is +inf
    if (Sleef_icmpeqq1(*x, QUAD_PRECISION_INF) || Sleef_icmpeqq1(*y, QUAD_PRECISION_INF)) {
        return QUAD_PRECISION_INF;
    }
    
    // If one is -inf, result is the other value
    if (Sleef_icmpeqq1(*x, neg_inf)) {
        return *y;
    }
    if (Sleef_icmpeqq1(*y, neg_inf)) {
        return *x;
    }
    
    // Numerically stable computation
    Sleef_quad diff = Sleef_subq1_u05(*x, *y);
    Sleef_quad abs_diff = Sleef_fabsq1(diff);
    Sleef_quad neg_abs_diff = Sleef_negq1(abs_diff);
    Sleef_quad exp_term = Sleef_expq1_u10(neg_abs_diff);
    Sleef_quad log1p_term = Sleef_log1pq1_u10(exp_term);
    
    Sleef_quad max_val = Sleef_icmpgtq1(*x, *y) ? *x : *y;
    return Sleef_addq1_u05(max_val, log1p_term);
}

static inline Sleef_quad
quad_logaddexp2(const Sleef_quad *x, const Sleef_quad *y)
{
    // logaddexp2(x, y) = log2(2^x + 2^y)
    // Numerically stable implementation: max(x, y) + log2(1 + 2^(-abs(x - y)))
    
    // Handle NaN
    if (Sleef_iunordq1(*x, *y)) {
        return Sleef_iunordq1(*x, *x) ? *x : *y;
    }
    
    // Handle infinities
    // If both are -inf, result is -inf
    Sleef_quad neg_inf = Sleef_negq1(QUAD_PRECISION_INF);
    if (Sleef_icmpeqq1(*x, neg_inf) && Sleef_icmpeqq1(*y, neg_inf)) {
        return neg_inf;
    }
    
    // If either is +inf, result is +inf
    if (Sleef_icmpeqq1(*x, QUAD_PRECISION_INF) || Sleef_icmpeqq1(*y, QUAD_PRECISION_INF)) {
        return QUAD_PRECISION_INF;
    }
    
    // If one is -inf, result is the other value
    if (Sleef_icmpeqq1(*x, neg_inf)) {
        return *y;
    }
    if (Sleef_icmpeqq1(*y, neg_inf)) {
        return *x;
    }
    
    // log2(2^x + 2^y) = max(x, y) + log2(1 + 2^(-abs(x - y)))
    Sleef_quad diff = Sleef_subq1_u05(*x, *y);
    Sleef_quad abs_diff = Sleef_fabsq1(diff);
    Sleef_quad neg_abs_diff = Sleef_negq1(abs_diff);
    Sleef_quad exp2_term = Sleef_exp2q1_u10(neg_abs_diff);
    Sleef_quad one_plus_exp2 = Sleef_addq1_u05(QUAD_PRECISION_ONE, exp2_term);
    Sleef_quad log2_term = Sleef_log2q1_u10(one_plus_exp2);
    
    Sleef_quad max_val = Sleef_icmpgtq1(*x, *y) ? *x : *y;
    return Sleef_addq1_u05(max_val, log2_term);
}

static inline Sleef_quad
quad_heaviside(const Sleef_quad *x1, const Sleef_quad *x2)
{
    // heaviside(x1, x2) = 0 if x1 < 0, x2 if x1 == 0, 1 if x1 > 0
    // NaN propagation: only propagate NaN from x1, not from x2 (unless x1 == 0)
    if (Sleef_iunordq1(*x1, *x1)) {
        return *x1;  // x1 is NaN, return NaN
    }
    
    if (Sleef_icmpltq1(*x1, QUAD_PRECISION_ZERO)) {
        return QUAD_PRECISION_ZERO;
    }
    else if (Sleef_icmpeqq1(*x1, QUAD_PRECISION_ZERO)) {
        return *x2;  // When x1 == 0, return x2 (even if x2 is NaN)
    }
    else {
        return QUAD_PRECISION_ONE;
    }
}

static inline Sleef_quad
quad_hypot(const Sleef_quad *x1, const Sleef_quad *x2)
{
    // hypot(x1, x2) = sqrt(x1^2 + x2^2)
    return Sleef_hypotq1_u05(*x1, *x2);
}

// todo: we definitely need to refactor this file, getting too clumsy everything here

static inline void quad_get_words64(int64_t *hx, uint64_t *lx, Sleef_quad x)
{
    union {
        Sleef_quad q;
        struct {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
            uint64_t hi;
            uint64_t lo;
#else
            uint64_t lo;
            uint64_t hi;
#endif
        } i;
    } u;
    u.q = x;
    *hx = (int64_t)u.i.hi;
    *lx = u.i.lo;
}

static inline Sleef_quad quad_set_words64(int64_t hx, uint64_t lx)
{
    union {
        Sleef_quad q;
        struct {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
            uint64_t hi;
            uint64_t lo;
#else
            uint64_t lo;
            uint64_t hi;
#endif
        } i;
    } u;
    u.i.hi = (uint64_t)hx;
    u.i.lo = lx;
    return u.q;
}


static inline Sleef_quad
quad_nextafter(const Sleef_quad *x, const Sleef_quad *y)
{
    int64_t hx, hy, ix;
    uint64_t lx, ly;

    quad_get_words64(&hx, &lx, *x);
    quad_get_words64(&hy, &ly, *y);
    
    // extracting absolute value
    ix = hx & 0x7fffffffffffffffLL;
    (void)ly;  // unused but needed for quad_get_words64

    // NaN if either is NaN
    if (Sleef_iunordq1(*x, *y)) {
        return Sleef_addq1_u05(*x, *y); // still NaN
    }

    // x == y then return y
    if (Sleef_icmpeqq1(*x, *y)) {
        return *y;
    }

    // both input 0 then extract sign from y and return correspondingly signed smallest subnormal
    if ((ix | lx) == 0) {
        Sleef_quad result = quad_set_words64(hy & 0x8000000000000000LL, 1); // quad_set_words64(sign_y, 1)
        return result;
    }

    if (hx >= 0) 
    {
        if (hx > hy || ((hx == hy) && (lx > ly))) 
        {
            //  Moving toward smaller y (x > y)
            // low word is 0 then decrement high word first (borrowing)
            if (lx == 0) 
                hx--;
            lx--;
        } 
        else 
        {
            lx++;
            if (lx == 0)
                // carry to high words
                hx++;
        }
    } 
    else 
    {
        // Moving toward larger y
        // similar to above case just direction will be swapped
        if (hy >= 0 || hx > hy || ((hx == hy) && (lx > ly))) 
        {
            if (lx == 0) 
                hx--;
            lx--;
        } 
        else 
        {
            lx++;
            if (lx == 0) 
                hx++;
        }
    }

    // check if reached infinity
    // this can be NaN XOR inf but NaN are already checked at start
    hy = hx & 0x7fff000000000000LL;    
    if (hy == 0x7fff000000000000LL) {
        Sleef_quad result = quad_set_words64(hx, lx);
        return result;
    }
    // check whether entered into subnormal range
    // 0 exponent i.e. either (0 or subnormal)
    if (hy == 0) {
        Sleef_quad result = quad_set_words64(hx, lx);
        return result;
    }
    
    // well I did not need to have those above checks
    // but they can be important when setting FPE flag manually
    return quad_set_words64(hx, lx);
}

static inline Sleef_quad
quad_spacing(const Sleef_quad *x)
{   
    // spacing(x) returns the distance between x and the next representable value
    // The result has the SAME SIGN as x (NumPy convention)
    // For x >= 0: spacing(x) = nextafter(x, +inf) - x
    // For x < 0:  spacing(x) = nextafter(x, -inf) - x (negative result)
    
    // Handle NaN
    if (Sleef_iunordq1(*x, *x)) {
        return *x;  // NaN
    }
    
    // Handle infinity -> NaN (numpy convention)
    if (quad_isinf(x)) {
        return QUAD_PRECISION_NAN;
    }
    
    // Determine direction based on sign of x
    Sleef_quad direction;
    if (Sleef_icmpltq1(*x, QUAD_PRECISION_ZERO)) {
        // Negative: move toward -inf
        direction = Sleef_negq1(QUAD_PRECISION_INF);
    } else {
        // Positive or zero: move toward +inf
        direction = QUAD_PRECISION_INF;
    }
    
    // Compute nextafter(x, direction)
    Sleef_quad next = quad_nextafter(x, &direction);
    
    // spacing = next - x (preserves sign)
    Sleef_quad result = Sleef_subq1_u05(next, *x);
    
    return result;
}

// Mixed-type operations (quad, int) -> quad
typedef Sleef_quad (*ldexp_op_quad_def)(const Sleef_quad *, const int *);
typedef long double (*ldexp_op_longdouble_def)(const long double *, const int *);

// Frexp operations: quad -> (quad mantissa, int exponent)
typedef Sleef_quad (*frexp_op_quad_def)(const Sleef_quad *, int *);
typedef long double (*frexp_op_longdouble_def)(const long double *, int *);

static inline Sleef_quad
quad_ldexp(const Sleef_quad *x, const int *exp)
{
    // ldexp(x, exp) returns x * 2^exp
    // SLEEF expects: Sleef_quad, int
    
    // NaN input -> NaN output (with sign preserved)
    if (Sleef_iunordq1(*x, *x)) {
        return *x;
    }
    
    // ±0 * 2^exp = ±0 (preserves sign of zero)
    if (Sleef_icmpeqq1(*x, QUAD_PRECISION_ZERO)) {
        return *x;
    }
    
    // ±inf * 2^exp = ±inf (preserves sign of infinity)
    if (quad_isinf(x)) {
        return *x;
    }
    
    Sleef_quad result = Sleef_ldexpq1(*x, *exp);
    
    return result;
}

static inline long double
ld_ldexp(const long double *x, const int *exp)
{
    // ldexp(x, exp) returns x * 2^exp
    // stdlib ldexpl expects: long double, int
    
    // NaN input -> NaN output
    if (isnan(*x)) {
        return *x;
    }
    
    // ±0 * 2^exp = ±0 (preserves sign of zero)
    if (*x == 0.0L) {
        return *x;
    }
    
    // ±inf * 2^exp = ±inf (preserves sign of infinity)
    if (isinf(*x)) {
        return *x;
    }
    
    long double result = ldexpl(*x, *exp);
    
    return result;
}

static inline Sleef_quad
quad_frexp(const Sleef_quad *x, int *exp)
{
    // frexp(x) returns mantissa m and exponent e such that x = m * 2^e
    // where 0.5 <= |m| < 1.0
    // NumPy's documentation says "between -1 and 1" but actual behavior is:
    // - Positive x: mantissa in [0.5, 1.0)
    // - Negative x: mantissa in (-1.0, -0.5]
    // This matches SLEEF's Sleef_frexpq1 behavior exactly.
    
    // NaN input -> NaN output with exponent 0
    if (Sleef_iunordq1(*x, *x)) {
        *exp = 0;
        return *x;
    }
    
    // ±0 -> mantissa ±0 with exponent 0 (preserves sign of zero)
    if (Sleef_icmpeqq1(*x, QUAD_PRECISION_ZERO)) {
        *exp = 0;
        return *x;
    }
    
    // ±inf -> mantissa ±inf with exponent 0 (preserves sign of infinity)
    if (quad_isinf(x)) {
        *exp = 0;
        return *x;
    }
    
    Sleef_quad mantissa = Sleef_frexpq1(*x, exp);
    
    return mantissa;
}

static inline long double
ld_frexp(const long double *x, int *exp)
{
    // frexp(x) returns mantissa m and exponent e such that x = m * 2^e
    
    // NaN input -> NaN output with exponent 0
    if (isnan(*x)) {
        *exp = 0;
        return *x;
    }
    
    // ±0 -> mantissa ±0 with exponent 0 (preserves sign of zero)
    if (*x == 0.0L) {
        *exp = 0;
        return *x;
    }
    
    // ±inf -> mantissa ±inf with exponent 0 (preserves sign of infinity)
    if (isinf(*x)) {
        *exp = 0;
        return *x;
    }
    
    long double mantissa = frexpl(*x, exp);
    
    return mantissa;
}

// Binary long double operations
typedef long double (*binary_op_longdouble_def)(const long double *, const long double *);
// Binary long double operations with 2 outputs (for divmod, modf, frexp)
typedef void (*binary_op_2out_longdouble_def)(const long double *, const long double *, long double *, long double *);

static inline long double
ld_add(const long double *in1, const long double *in2)
{
    return (*in1) + (*in2);
}

static inline long double
ld_sub(const long double *in1, const long double *in2)
{
    return (*in1) - (*in2);
}

static inline long double
ld_mul(const long double *a, const long double *b)
{
    return (*a) * (*b);
}

static inline long double
ld_div(const long double *a, const long double *b)
{
    return (*a) / (*b);
}

static inline long double
ld_floor_divide(const long double *a, const long double *b)
{
    // Handle NaN inputs
    if (isnan(*a) || isnan(*b)) {
        return isnan(*a) ? *a : *b;
    }
    
    // inf / finite_nonzero or -inf / finite_nonzero -> NaN
    // But inf / 0 -> inf
    if (isinf(*a) && isfinite(*b) && *b != 0.0L) {
        return NAN;
    }
    
    // 0 / 0 (including -0.0 / 0.0, 0.0 / -0.0, -0.0 / -0.0) -> NaN
    if (*a == 0.0L && *b == 0.0L) {
        return NAN;
    }
    
    // Compute a / b and apply floor
    long double result = floorl((*a) / (*b));
    
    // floor_divide semantics: when result is -0.0 from non-zero numerator, convert to -1.0
    // This happens when: (negative & non-zero)/+inf, (positive & non-zero)/-inf
    // But NOT when numerator is ±0.0 (then result stays as ±0.0)
    if (result == 0.0L && signbit(result) && *a != 0.0L) {
        return -1.0L;
    }
    
    return result;
}

static inline long double
ld_pow(const long double *a, const long double *b)
{
    return powl(*a, *b);
}

static inline long double
ld_mod(const long double *a, const long double *b)
{
    if (*b == 0.0L)
        return NAN;
    if (isnan(*a) || isnan(*b))
        return isnan(*a) ? *a : *b;
    if (isinf(*a))
        return NAN;

    if (isfinite(*a) && isinf(*b)) {
        int sign_a = signbit(*a);
        int sign_b = signbit(*b);
        return (sign_a == sign_b) ? *a : *b;
    }

    long double quotient = (*a) / (*b);
    long double floored = floorl(quotient);
    long double result = (*a) - floored * (*b);

    if (result == 0.0L) {
        return (*b < 0.0L) ? -0.0L : 0.0L;
    }

    return result;
}

static inline long double
ld_fmod(const long double *a, const long double *b)
{
    // Handle NaN inputs
    if (isnan(*a) || isnan(*b)) {
        return isnan(*a) ? *a : *b;
    }
    
    // Division by zero -> NaN
    if (*b == 0.0L) {
        return NAN;
    }
    
    // Infinity dividend -> NaN
    if (isinf(*a)) {
        return NAN;
    }
    
    // Finite % infinity -> return dividend
    if (isfinite(*a) && isinf(*b)) {
        return *a;
    }

    long double result = fmodl(*a, *b);
    
    if (result == 0.0L) {
        // Preserve sign of dividend
        if (signbit(*a)) {
            return -0.0L;
        } else {
            return 0.0L;
        }
    }
    
    return result;
}

static inline void
ld_divmod(const long double *a, const long double *b,
          long double *out_quotient, long double *out_remainder)
{
    *out_quotient = ld_floor_divide(a, b);
    *out_remainder = ld_mod(a, b);
}

static inline long double
ld_minimum(const long double *in1, const long double *in2)
{
    return isnan(*in1) ? *in1 : (*in1 < *in2) ? *in1 : *in2;
}

static inline long double
ld_maximum(const long double *in1, const long double *in2)
{
    return isnan(*in1) ? *in1 : (*in1 > *in2) ? *in1 : *in2;
}

static inline long double
ld_fmin(const long double *in1, const long double *in2)
{
    return fmin(*in1, *in2);
}

static inline long double
ld_fmax(const long double *in1, const long double *in2)
{
    return fmax(*in1, *in2);
}

static inline long double
ld_atan2(const long double *in1, const long double *in2)
{
    return atan2l(*in1, *in2);
}

static inline long double
ld_copysign(const long double *in1, const long double *in2)
{
    return copysignl(*in1, *in2);
}

static inline long double
ld_logaddexp(const long double *x, const long double *y)
{
    // logaddexp(x, y) = log(exp(x) + exp(y))
    // Numerically stable implementation: max(x, y) + log1p(exp(-abs(x - y)))
    
    // Handle NaN
    if (isnan(*x) || isnan(*y)) {
        return isnan(*x) ? *x : *y;
    }
    
    // Handle infinities
    // If both are -inf, result is -inf
    if (isinf(*x) && *x < 0 && isinf(*y) && *y < 0) {
        return -INFINITY;
    }
    
    // If either is +inf, result is +inf
    if ((isinf(*x) && *x > 0) || (isinf(*y) && *y > 0)) {
        return INFINITY;
    }
    
    // If one is -inf, result is the other value
    if (isinf(*x) && *x < 0) {
        return *y;
    }
    if (isinf(*y) && *y < 0) {
        return *x;
    }
    
    // Numerically stable computation
    long double diff = *x - *y;
    long double abs_diff = fabsl(diff);
    long double max_val = (*x > *y) ? *x : *y;
    return max_val + log1pl(expl(-abs_diff));
}

static inline long double
ld_logaddexp2(const long double *x, const long double *y)
{
    // logaddexp2(x, y) = log2(2^x + 2^y)
    // Numerically stable implementation: max(x, y) + log2(1 + 2^(-abs(x - y)))
    
    // Handle NaN
    if (isnan(*x) || isnan(*y)) {
        return isnan(*x) ? *x : *y;
    }
    
    // Handle infinities
    // If both are -inf, result is -inf
    if (isinf(*x) && *x < 0 && isinf(*y) && *y < 0) {
        return -INFINITY;
    }
    
    // If either is +inf, result is +inf
    if ((isinf(*x) && *x > 0) || (isinf(*y) && *y > 0)) {
        return INFINITY;
    }
    
    // If one is -inf, result is the other value
    if (isinf(*x) && *x < 0) {
        return *y;
    }
    if (isinf(*y) && *y < 0) {
        return *x;
    }
    
    // log2(2^x + 2^y) = max(x, y) + log2(1 + 2^(-abs(x - y)))
    long double diff = *x - *y;
    long double abs_diff = fabsl(diff);
    long double max_val = (*x > *y) ? *x : *y;
    // Use native log2l function for base-2 logarithm
    return max_val + log2l(1.0L + exp2l(-abs_diff));
}

static inline long double
ld_heaviside(const long double *x1, const long double *x2)
{
    // heaviside(x1, x2) = 0 if x1 < 0, x2 if x1 == 0, 1 if x1 > 0
    // NaN propagation: only propagate NaN from x1, not from x2 (unless x1 == 0)
    if (isnan(*x1)) {
        return *x1;  // x1 is NaN, return NaN
    }
    
    if (*x1 < 0.0L) {
        return 0.0L;
    }
    else if (*x1 == 0.0L) {
        return *x2;  // When x1 == 0, return x2 (even if x2 is NaN)
    }
    else {
        return 1.0L;
    }
}

static inline long double
ld_hypot(const long double *x1, const long double *x2)
{
    // hypot(x1, x2) = sqrt(x1^2 + x2^2)
    // Use the standard library hypotl function
    return hypotl(*x1, *x2);
}

static inline long double
ld_nextafter(const long double *x1, const long double *x2)
{
    return nextafterl(*x1, *x2);
}

static inline long double
ld_spacing(const long double *x)
{    
    // Handle NaN
    if (isnan(*x)) {
        return *x;  // NaN
    }
    
    // Handle infinity -> NaN (numpy convention)
    if (isinf(*x)) {
        return NAN;
    }
    
    // Determine direction based on sign of x
    long double direction;
    if (*x < 0.0L) {
        // Negative: move toward -inf
        direction = -INFINITY;
    } else {
        // Positive or zero: move toward +inf
        direction = INFINITY;
    }
    
    // Compute nextafter(x, direction)
    long double next = nextafterl(*x, direction);
    
    // spacing = next - x (preserves sign)
    long double result = next - (*x);
    
    return result;
}

// Unary operations with 2 outputs
static inline void
quad_modf(const Sleef_quad *a, Sleef_quad *out_fractional, Sleef_quad *out_integral)
{
    // int part stored in out_integral
    *out_fractional = Sleef_modfq1(*a, out_integral);
}

// Unary long double operations with 2 outputs  
typedef void (*unary_op_2out_longdouble_def)(const long double *, long double *, long double *);

static inline void
ld_modf(const long double *a, long double *out_fractional, long double *out_integral)
{
    *out_fractional = modfl(*a, out_integral);
}

// comparison quad functions
typedef npy_bool (*cmp_quad_def)(const Sleef_quad *, const Sleef_quad *);

static inline npy_bool
quad_equal(const Sleef_quad *a, const Sleef_quad *b)
{
    return Sleef_icmpeqq1(*a, *b);
}

static inline npy_bool
quad_notequal(const Sleef_quad *a, const Sleef_quad *b)
{
    return Sleef_icmpneq1(*a, *b) || Sleef_iunordq1(*a, *b);
}

static inline npy_bool
quad_less(const Sleef_quad *a, const Sleef_quad *b)
{
    return Sleef_icmpltq1(*a, *b);
}

static inline npy_bool
quad_lessequal(const Sleef_quad *a, const Sleef_quad *b)
{
    return Sleef_icmpleq1(*a, *b);
}

static inline npy_bool
quad_greater(const Sleef_quad *a, const Sleef_quad *b)
{
    return Sleef_icmpgtq1(*a, *b);
}

static inline npy_bool
quad_greaterequal(const Sleef_quad *a, const Sleef_quad *b)
{
    return Sleef_icmpgeq1(*a, *b);
}

// comparison quad functions
typedef npy_bool (*cmp_londouble_def)(const long double *, const long double *);

static inline npy_bool
ld_equal(const long double *a, const long double *b)
{
    return *a == *b;
}

static inline npy_bool
ld_notequal(const long double *a, const long double *b)
{
    return *a != *b;
}

static inline npy_bool
ld_less(const long double *a, const long double *b)
{
    return *a < *b;
}

static inline npy_bool
ld_lessequal(const long double *a, const long double *b)
{
    return *a <= *b;
}

static inline npy_bool
ld_greater(const long double *a, const long double *b)
{
    return *a > *b;
}

static inline npy_bool
ld_greaterequal(const long double *a, const long double *b)
{
    return *a >= *b;
}

// Logical operations

// Helper function to check if a Sleef_quad value is non-zero (truthy)
static inline npy_bool
quad_is_nonzero(const Sleef_quad *a)
{
    // A value is falsy if it's exactly zero (positive or negative)
    // NaN and inf are truthy
    npy_bool is_zero = Sleef_icmpeqq1(*a, QUAD_PRECISION_ZERO);
    return !is_zero;
}

// Helper function to check if a long double value is non-zero (truthy)
static inline npy_bool
ld_is_nonzero(const long double *a)
{
    // A value is falsy if it's exactly zero (positive or negative)
    // NaN and inf are truthy
    return *a != 0.0L;
}


static inline npy_bool
quad_logical_and(const Sleef_quad *a, const Sleef_quad *b)
{
    return quad_is_nonzero(a) && quad_is_nonzero(b);
}

static inline npy_bool
ld_logical_and(const long double *a, const long double *b)
{
    return ld_is_nonzero(a) && ld_is_nonzero(b);
}


static inline npy_bool
quad_logical_or(const Sleef_quad *a, const Sleef_quad *b)
{
    return quad_is_nonzero(a) || quad_is_nonzero(b);
}

static inline npy_bool
ld_logical_or(const long double *a, const long double *b)
{
    return ld_is_nonzero(a) || ld_is_nonzero(b);
}

static inline npy_bool
quad_logical_xor(const Sleef_quad *a, const Sleef_quad *b)
{
    npy_bool a_truthy = quad_is_nonzero(a);
    npy_bool b_truthy = quad_is_nonzero(b);
    return (a_truthy && !b_truthy) || (!a_truthy && b_truthy);
}

static inline npy_bool
ld_logical_xor(const long double *a, const long double *b)
{
    npy_bool a_truthy = ld_is_nonzero(a);
    npy_bool b_truthy = ld_is_nonzero(b);
    return (a_truthy && !b_truthy) || (!a_truthy && b_truthy);
}


// logical not
typedef npy_bool (*unary_logical_quad_def)(const Sleef_quad *);
typedef npy_bool (*unary_logical_longdouble_def)(const long double *);

static inline npy_bool
quad_logical_not(const Sleef_quad *a)
{
    return !quad_is_nonzero(a);
}

static inline npy_bool
ld_logical_not(const long double *a)
{
    return !ld_is_nonzero(a);
}

// Casting operations
static inline double
cast_sleef_to_double(const Sleef_quad in)
{
    if (quad_isnan(&in)) {
        return quad_signbit(&in) ? -NAN : NAN;
    }
    if (quad_isinf(&in)) {
        return quad_signbit(&in) ? -INFINITY : INFINITY;
    }
    if (Sleef_icmpeqq1(in, QUAD_PRECISION_ZERO))
    {
        return quad_signbit(&in) ? -0.0 : 0.0;
    }
    return Sleef_cast_to_doubleq1(in);
}