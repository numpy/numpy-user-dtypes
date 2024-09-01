#include <sleef.h>
#include <sleefquad.h>
#include <cmath>

// Unary Quad Operations
typedef int (*unary_op_quad_def)(Sleef_quad *, Sleef_quad *);

static int
quad_negative(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_negq1(*op);
    return 0;
}

static int
quad_positive(Sleef_quad *op, Sleef_quad *out)
{
    *out = *op;
    return 0;
}

static inline int
quad_absolute(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_fabsq1(*op);
    return 0;
}

static inline int
quad_rint(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_rintq1(*op);
    return 0;
}

static inline int
quad_trunc(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_truncq1(*op);
    return 0;
}

static inline int
quad_floor(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_floorq1(*op);
    return 0;
}

static inline int
quad_ceil(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_ceilq1(*op);
    return 0;
}

static inline int
quad_sqrt(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_sqrtq1_u05(*op);
    return 0;
}

static inline int
quad_square(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_mulq1_u05(*op, *op);
    return 0;
}

static inline int
quad_log(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_logq1_u10(*op);
    return 0;
}

static inline int
quad_log2(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_log2q1_u10(*op);
    return 0;
}

static inline int
quad_log10(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_log10q1_u10(*op);
    return 0;
}

static inline int
quad_log1p(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_log1pq1_u10(*op);
    return 0;
}

static inline int
quad_exp(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_expq1_u10(*op);
    return 0;
}

static inline int
quad_exp2(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_exp2q1_u10(*op);
    return 0;
}

// Unary long double operations
typedef int (*unary_op_longdouble_def)(long double *, long double *);

static int
ld_negative(long double *op, long double *out)
{
    *out = -(*op);
    return 0;
}

static int
ld_positive(long double *op, long double *out)
{
    *out = *op;
    return 0;
}

static inline int
ld_absolute(long double *op, long double *out)
{
    *out = fabsl(*op);
    return 0;
}

static inline int
ld_rint(long double *op, long double *out)
{
    *out = rintl(*op);
    return 0;
}

static inline int
ld_trunc(long double *op, long double *out)
{
    *out = truncl(*op);
    return 0;
}

static inline int
ld_floor(long double *op, long double *out)
{
    *out = floorl(*op);
    return 0;
}

static inline int
ld_ceil(long double *op, long double *out)
{
    *out = ceill(*op);
    return 0;
}

static inline int
ld_sqrt(long double *op, long double *out)
{
    *out = sqrtl(*op);
    return 0;
}

static inline int
ld_square(long double *op, long double *out)
{
    *out = (*op) * (*op);
    return 0;
}

static inline int
ld_log(long double *op, long double *out)
{
    *out = logl(*op);
    return 0;
}

static inline int
ld_log2(long double *op, long double *out)
{
    *out = log2l(*op);
    return 0;
}

static inline int
ld_log10(long double *op, long double *out)
{
    *out = log10l(*op);
    return 0;
}

static inline int
ld_log1p(long double *op, long double *out)
{
    *out = log1pl(*op);
    return 0;
}

static inline int
ld_exp(long double *op, long double *out)
{
    *out = expl(*op);
    return 0;
}

static inline int
ld_exp2(long double *op, long double *out)
{
    *out = exp2l(*op);
    return 0;
}

// Binary Quad operations
typedef int (*binary_op_quad_def)(Sleef_quad *, Sleef_quad *, Sleef_quad *);

static inline int
quad_add(Sleef_quad *out, Sleef_quad *in1, Sleef_quad *in2)
{
    *out = Sleef_addq1_u05(*in1, *in2);
    return 0;
}

static inline int
quad_sub(Sleef_quad *out, Sleef_quad *in1, Sleef_quad *in2)
{
    *out = Sleef_subq1_u05(*in1, *in2);
    return 0;
}

static inline int
quad_mul(Sleef_quad *res, Sleef_quad *a, Sleef_quad *b)
{
    *res = Sleef_mulq1_u05(*a, *b);
    return 0;
}

static inline int
quad_div(Sleef_quad *res, Sleef_quad *a, Sleef_quad *b)
{
    *res = Sleef_divq1_u05(*a, *b);
    return 0;
}

static inline int
quad_pow(Sleef_quad *res, Sleef_quad *a, Sleef_quad *b)
{
    *res = Sleef_powq1_u10(*a, *b);
    return 0;
}

static inline int
quad_mod(Sleef_quad *res, Sleef_quad *a, Sleef_quad *b)
{
    *res = Sleef_fmodq1(*a, *b);
    return 0;
}

static inline int
quad_minimum(Sleef_quad *out, Sleef_quad *in1, Sleef_quad *in2)
{
    *out = Sleef_icmpleq1(*in1, *in2) ? *in1 : *in2;
    return 0;
}

static inline int
quad_maximum(Sleef_quad *out, Sleef_quad *in1, Sleef_quad *in2)
{
    *out = Sleef_icmpgeq1(*in1, *in2) ? *in1 : *in2;
    return 0;
}

// Binary long double operations
typedef int (*binary_op_longdouble_def)(long double *, long double *, long double *);

static inline int
ld_add(long double *out, long double *in1, long double *in2)
{
    *out = (*in1) + (*in2);
    return 0;
}

static inline int
ld_sub(long double *out, long double *in1, long double *in2)
{
    *out = (*in1) - (*in2);
    return 0;
}

static inline int
ld_mul(long double *res, long double *a, long double *b)
{
    *res = (*a) * (*b);
    return 0;
}

static inline int
ld_div(long double *res, long double *a, long double *b)
{
    *res = (*a) / (*b);
    return 0;
}

static inline int
ld_pow(long double *res, long double *a, long double *b)
{
    *res = powl(*a, *b);
    return 0;
}

static inline int
ld_mod(long double *res, long double *a, long double *b)
{
    *res = fmodl(*a, *b);
    return 0;
}

static inline int
ld_minimum(long double *out, long double *in1, long double *in2)
{
    *out = (*in1 < *in2) ? *in1 : *in2;
    return 0;
}

static inline int
ld_maximum(long double *out, long double *in1, long double *in2)
{
    *out = (*in1 > *in2) ? *in1 : *in2;
    return 0;
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
    return Sleef_icmpneq1(*a, *b);
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
typedef npy_bool (*cmp_londouble_def)(const long double *, const double *);

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