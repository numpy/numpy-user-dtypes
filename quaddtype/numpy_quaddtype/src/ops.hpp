#include <sleef.h>
#include <sleefquad.h>
#include <cmath>

// Unary Quad Operations
typedef Sleef_quad (*unary_op_quad_def)(Sleef_quad *);

static Sleef_quad
quad_negative(Sleef_quad *op)
{
    return Sleef_negq1(*op);
}

static Sleef_quad
quad_positive(Sleef_quad *op)
{
    return *op;
}

static inline Sleef_quad
quad_absolute(Sleef_quad *op)
{
    return Sleef_fabsq1(*op);
}

static inline Sleef_quad
quad_rint(Sleef_quad *op)
{
    return Sleef_rintq1(*op);
}

static inline Sleef_quad
quad_trunc(Sleef_quad *op)
{
    return Sleef_truncq1(*op);
}

static inline Sleef_quad
quad_floor(Sleef_quad *op)
{
    return Sleef_floorq1(*op);
}

static inline Sleef_quad
quad_ceil(Sleef_quad *op)
{
    return Sleef_ceilq1(*op);
}

static inline Sleef_quad
quad_sqrt(Sleef_quad *op)
{
    return Sleef_sqrtq1_u05(*op);
}

static inline Sleef_quad
quad_square(Sleef_quad *op)
{
    return Sleef_mulq1_u05(*op, *op);
}

static inline Sleef_quad
quad_log(Sleef_quad *op)
{
    return Sleef_logq1_u10(*op);
}

static inline Sleef_quad
quad_log2(Sleef_quad *op)
{
    return Sleef_log2q1_u10(*op);
}

static inline Sleef_quad
quad_log10(Sleef_quad *op)
{
    return Sleef_log10q1_u10(*op);
}

static inline Sleef_quad
quad_log1p(Sleef_quad *op)
{
    return Sleef_log1pq1_u10(*op);
}

static inline Sleef_quad
quad_exp(Sleef_quad *op)
{
    return Sleef_expq1_u10(*op);
}

static inline Sleef_quad
quad_exp2(Sleef_quad *op)
{
    return Sleef_exp2q1_u10(*op);
}

static inline Sleef_quad
quad_sin(Sleef_quad *op)
{
    return Sleef_sinq1_u10(*op);
}

static inline Sleef_quad
quad_cos(Sleef_quad *op)
{
    return Sleef_cosq1_u10(*op);
}

static inline Sleef_quad
quad_tan(Sleef_quad *op)
{
    return Sleef_tanq1_u10(*op);
}

static inline Sleef_quad
quad_asin(Sleef_quad *op)
{
    return Sleef_asinq1_u10(*op);
}

static inline Sleef_quad
quad_acos(Sleef_quad *op)
{
    return Sleef_acosq1_u10(*op);
}

static inline Sleef_quad
quad_atan(Sleef_quad *op)
{
    return Sleef_atanq1_u10(*op);
}

// Unary long double operations
typedef long double (*unary_op_longdouble_def)(long double *);

static inline long double
ld_negative(long double *op)
{
    return -(*op);
}

static inline long double
ld_positive(long double *op)
{
    return *op;
}

static inline long double
ld_absolute(long double *op)
{
    return fabsl(*op);
}

static inline long double
ld_rint(long double *op)
{
    return rintl(*op);
}

static inline long double
ld_trunc(long double *op)
{
    return truncl(*op);
}

static inline long double
ld_floor(long double *op)
{
    return floorl(*op);
}

static inline long double
ld_ceil(long double *op)
{
    return ceill(*op);
}

static inline long double
ld_sqrt(long double *op)
{
    return sqrtl(*op);
}

static inline long double
ld_square(long double *op)
{
    return (*op) * (*op);
}

static inline long double
ld_log(long double *op)
{
    return logl(*op);
}

static inline long double
ld_log2(long double *op)
{
    return log2l(*op);
}

static inline long double
ld_log10(long double *op)
{
    return log10l(*op);
}

static inline long double
ld_log1p(long double *op)
{
    return log1pl(*op);
}

static inline long double
ld_exp(long double *op)
{
    return expl(*op);
}

static inline long double
ld_exp2(long double *op)
{
    return exp2l(*op);
}

static inline long double
ld_sin(long double *op)
{
    return sinl(*op);
}

static inline long double
ld_cos(long double *op)
{
    return cosl(*op);
}

static inline long double
ld_tan(long double *op)
{
    return tanl(*op);
}

static inline long double
ld_asin(long double *op)
{
    return asinl(*op);
}

static inline long double
ld_acos(long double *op)
{
    return acosl(*op);
}

static inline long double
ld_atan(long double *op)
{
    return atanl(*op);
}

// Binary Quad operations
typedef Sleef_quad (*binary_op_quad_def)(Sleef_quad *, Sleef_quad *);

static inline Sleef_quad
quad_add(Sleef_quad *in1, Sleef_quad *in2)
{
    return Sleef_addq1_u05(*in1, *in2);
}

static inline Sleef_quad
quad_sub(Sleef_quad *in1, Sleef_quad *in2)
{
    return Sleef_subq1_u05(*in1, *in2);
}

static inline Sleef_quad
quad_mul(Sleef_quad *a, Sleef_quad *b)
{
    return Sleef_mulq1_u05(*a, *b);
}

static inline Sleef_quad
quad_div(Sleef_quad *a, Sleef_quad *b)
{
    return Sleef_divq1_u05(*a, *b);
}

static inline Sleef_quad
quad_pow(Sleef_quad *a, Sleef_quad *b)
{
    return Sleef_powq1_u10(*a, *b);
}

static inline Sleef_quad
quad_mod(Sleef_quad *a, Sleef_quad *b)
{
    return Sleef_fmodq1(*a, *b);
}

static inline Sleef_quad
quad_minimum(Sleef_quad *in1, Sleef_quad *in2)
{
    return Sleef_iunordq1(*in1, *in2) ? (
        Sleef_iunordq1(*in1, *in1) ? *in1 : *in2
    ) : Sleef_icmpleq1(*in1, *in2) ? *in1 : *in2;
}

static inline Sleef_quad
quad_maximum(Sleef_quad *in1, Sleef_quad *in2)
{
    return Sleef_iunordq1(*in1, *in2) ? (
        Sleef_iunordq1(*in1, *in1) ? *in1 : *in2
    ) : Sleef_icmpgeq1(*in1, *in2) ? *in1 : *in2;
}

static inline Sleef_quad
quad_atan2(Sleef_quad *in1, Sleef_quad *in2)
{
    return Sleef_atan2q1_u10(*in1, *in2);
}

// Binary long double operations
typedef long double (*binary_op_longdouble_def)(long double *, long double *);

static inline long double
ld_add(long double *in1, long double *in2)
{
    return (*in1) + (*in2);
}

static inline long double
ld_sub(long double *in1, long double *in2)
{
    return (*in1) - (*in2);
}

static inline long double
ld_mul(long double *a, long double *b)
{
    return (*a) * (*b);
}

static inline long double
ld_div(long double *a, long double *b)
{
    return (*a) / (*b);
}

static inline long double
ld_pow(long double *a, long double *b)
{
    return powl(*a, *b);
}

static inline long double
ld_mod(long double *a, long double *b)
{
    return fmodl(*a, *b);
}

static inline long double
ld_minimum(long double *in1, long double *in2)
{
    return (*in1 < *in2) ? *in1 : *in2;
}

static inline long double
ld_maximum(long double *in1, long double *in2)
{
    return (*in1 > *in2) ? *in1 : *in2;
}

static inline long double
ld_atan2(long double *in1, long double *in2)
{
    return atan2l(*in1, *in2);
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