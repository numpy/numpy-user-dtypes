#include <sleef.h>
#include <sleefquad.h>
#include <cmath>

// Quad Constants, generated with qutil
#define QUAD_ZERO sleef_q(+0x0000000000000LL, 0x0000000000000000ULL, -16383)
#define QUAD_ONE sleef_q(+0x1000000000000LL, 0x0000000000000000ULL, 0)
#define QUAD_POS_INF sleef_q(+0x1000000000000LL, 0x0000000000000000ULL, 16384)
#define QUAD_NAN sleef_q(+0x1ffffffffffffLL, 0xffffffffffffffffULL, 16384)

// Unary Quad Operations
typedef Sleef_quad (*unary_op_quad_def)(const Sleef_quad *);

static Sleef_quad
quad_negative(const Sleef_quad *op)
{
    return Sleef_negq1(*op);
}

static Sleef_quad
quad_positive(const Sleef_quad *op)
{
    return *op;
}

static inline Sleef_quad
quad_sign(const Sleef_quad *op)
{
    int32_t sign = Sleef_icmpq1(*op, QUAD_ZERO);
    // sign(x=NaN) = x; otherwise sign(x) in { -1.0; 0.0; +1.0 }
    return Sleef_iunordq1(*op, *op) ? *op : Sleef_cast_from_int64q1(sign);
}

static inline Sleef_quad
quad_absolute(const Sleef_quad *op)
{
    return Sleef_fabsq1(*op);
}

static inline Sleef_quad
quad_fabs(const Sleef_quad *op)
{
    return Sleef_fabsq1(*op);
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
quad_square(const Sleef_quad *op)
{
    return Sleef_mulq1_u05(*op, *op);
}

static inline Sleef_quad
quad_reciprocal(const Sleef_quad *op)
{
    return Sleef_divq1_u05(QUAD_ONE, *op);
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
ld_fabs(const long double *op)
{
    return fabsl(*op);
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

// Unary Quad properties
typedef npy_bool (*unary_prop_quad_def)(const Sleef_quad *);

static inline npy_bool
quad_signbit(const Sleef_quad *op)
{
    // FIXME @juntyr or @SwayamInSync: replace with binary implementation
    //  once we test big and little endian in CI
    Sleef_quad one_signed = Sleef_copysignq1(QUAD_ONE, *op);
    // signbit(x) = 1 iff copysign(1, x) == -1
    return Sleef_icmpltq1(one_signed, QUAD_ZERO);
}

static inline npy_bool
quad_isfinite(const Sleef_quad *op)
{
    // isfinite(x) = abs(x) < inf
    return Sleef_icmpltq1(Sleef_fabsq1(*op), QUAD_POS_INF);
}

static inline npy_bool
quad_isinf(const Sleef_quad *op)
{
    // isinf(x) = abs(x) == inf
    return Sleef_icmpeqq1(Sleef_fabsq1(*op), QUAD_POS_INF);
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
    if (quad_isinf(a) && quad_isfinite(b) && !Sleef_icmpeqq1(*b, QUAD_ZERO)) {
        return QUAD_NAN;
    }
    
    // 0 / 0 (including -0.0 / 0.0, 0.0 / -0.0, -0.0 / -0.0) -> NaN
    if (Sleef_icmpeqq1(*a, QUAD_ZERO) && Sleef_icmpeqq1(*b, QUAD_ZERO)) {
        return QUAD_NAN;
    }
    
    Sleef_quad quotient = Sleef_divq1_u05(*a, *b);
    Sleef_quad result = Sleef_floorq1(quotient);
    
    // floor_divide semantics: when result is -0.0 from non-zero numerator, convert to -1.0
    // This happens when: (negative & non-zero)/+inf, (positive & non-zero)/-inf
    // But NOT when numerator is ±0.0 (then result stays as ±0.0)
    if (Sleef_icmpeqq1(result, QUAD_ZERO) && quad_signbit(&result) && 
        !Sleef_icmpeqq1(*a, QUAD_ZERO)) {
        return Sleef_negq1(QUAD_ONE);  // -1.0
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
    if (Sleef_icmpeqq1(*b, QUAD_ZERO)) {
        return QUAD_NAN;
    }

    // NaN inputs
    if (Sleef_iunordq1(*a, *b)) {
        return Sleef_iunordq1(*a, *a) ? *a : *b;  // Return the NaN
    }

    // infinity dividend -> NaN
    if (quad_isinf(a)) {
        return QUAD_NAN;
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
    if (Sleef_icmpeqq1(result, QUAD_ZERO)) {
        if (Sleef_icmpltq1(*b, QUAD_ZERO)) {
            return Sleef_negq1(QUAD_ZERO);  // -0.0
        }
        else {
            return QUAD_ZERO;  // +0.0
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
    if (Sleef_icmpeqq1(*b, QUAD_ZERO)) {
        return QUAD_NAN;
    }
    
    // Infinity dividend -> NaN
    if (quad_isinf(a)) {
        return QUAD_NAN;
    }
    
    // Finite % infinity -> return dividend (same as a)
    if (quad_isfinite(a) && quad_isinf(b)) {
        return *a;
    }
    
    // x - trunc(x/y) * y
    Sleef_quad result = Sleef_fmodq1(*a, *b);
    
    if (Sleef_icmpeqq1(result, QUAD_ZERO)) {
        // Preserve sign of dividend (first argument)
        Sleef_quad sign_test = Sleef_copysignq1(QUAD_ONE, *a);
        if (Sleef_icmpltq1(sign_test, QUAD_ZERO)) {
            return Sleef_negq1(QUAD_ZERO);  // -0.0
        }
        else {
            return QUAD_ZERO;  // +0.0
        }
    }
    
    return result;
}

static inline Sleef_quad
quad_minimum(const Sleef_quad *in1, const Sleef_quad *in2)
{
    if (Sleef_iunordq1(*in1, *in2)) {
        return Sleef_iunordq1(*in1, *in1) ? *in1 : *in2;
    }
    // minimum(-0.0, +0.0) = -0.0
    if (Sleef_icmpeqq1(*in1, QUAD_ZERO) && Sleef_icmpeqq1(*in2, QUAD_ZERO)) {
        return Sleef_icmpleq1(Sleef_copysignq1(QUAD_ONE, *in1), Sleef_copysignq1(QUAD_ONE, *in2)) ? *in1 : *in2;
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
    if (Sleef_icmpeqq1(*in1, QUAD_ZERO) && Sleef_icmpeqq1(*in2, QUAD_ZERO)) {
        return Sleef_icmpgeq1(Sleef_copysignq1(QUAD_ONE, *in1), Sleef_copysignq1(QUAD_ONE, *in2)) ? *in1 : *in2;
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
    if (Sleef_icmpeqq1(*in1, QUAD_ZERO) && Sleef_icmpeqq1(*in2, QUAD_ZERO)) {
        return Sleef_icmpleq1(Sleef_copysignq1(QUAD_ONE, *in1), Sleef_copysignq1(QUAD_ONE, *in2)) ? *in1 : *in2;
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
    if (Sleef_icmpeqq1(*in1, QUAD_ZERO) && Sleef_icmpeqq1(*in2, QUAD_ZERO)) {
        return Sleef_icmpgeq1(Sleef_copysignq1(QUAD_ONE, *in1), Sleef_copysignq1(QUAD_ONE, *in2)) ? *in1 : *in2;
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
    Sleef_quad neg_inf = Sleef_negq1(QUAD_POS_INF);
    if (Sleef_icmpeqq1(*x, neg_inf) && Sleef_icmpeqq1(*y, neg_inf)) {
        return neg_inf;
    }
    
    // If either is +inf, result is +inf
    if (Sleef_icmpeqq1(*x, QUAD_POS_INF) || Sleef_icmpeqq1(*y, QUAD_POS_INF)) {
        return QUAD_POS_INF;
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
    Sleef_quad neg_inf = Sleef_negq1(QUAD_POS_INF);
    if (Sleef_icmpeqq1(*x, neg_inf) && Sleef_icmpeqq1(*y, neg_inf)) {
        return neg_inf;
    }
    
    // If either is +inf, result is +inf
    if (Sleef_icmpeqq1(*x, QUAD_POS_INF) || Sleef_icmpeqq1(*y, QUAD_POS_INF)) {
        return QUAD_POS_INF;
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
    Sleef_quad one_plus_exp2 = Sleef_addq1_u05(QUAD_ONE, exp2_term);
    Sleef_quad log2_term = Sleef_log2q1_u10(one_plus_exp2);
    
    Sleef_quad max_val = Sleef_icmpgtq1(*x, *y) ? *x : *y;
    return Sleef_addq1_u05(max_val, log2_term);
}

// Binary long double operations
typedef long double (*binary_op_longdouble_def)(const long double *, const long double *);

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
