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
quad_rint(const Sleef_quad *op)
{
    return Sleef_rintq1(*op);
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
