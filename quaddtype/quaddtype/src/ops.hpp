#include <sleef.h>
#include <sleefquad.h>

typedef int (*unary_op_def)(Sleef_quad *, Sleef_quad *);

static inline int
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

// binary ops
typedef int (*binop_def)(Sleef_quad *, Sleef_quad *, Sleef_quad *);

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

// comparison functions
typedef npy_bool (*cmp_def)(const Sleef_quad *, const Sleef_quad *);

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