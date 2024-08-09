#include <sleef.h>
#include <sleefquad.h>

typedef int (*unary_op_def)(Sleef_quad *, Sleef_quad *);

static inline int
quad_negative(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_negq1(*op);
    return 0;
}

static inline int
quad_absolute(Sleef_quad *op, Sleef_quad *out)
{
    *out = Sleef_fabsq1(*op);
    return 0;
}

// binary ops
typedef int (*binop_def)(Sleef_quad *, Sleef_quad *, Sleef_quad *);

static inline int quad_add(Sleef_quad *out, Sleef_quad *in1, Sleef_quad *in2)
{
    *out = Sleef_addq1_u05(*in1, *in2);
    return 0;
}

static inline int quad_sub(Sleef_quad *out, Sleef_quad *in1, Sleef_quad *in2)
{
    *out = Sleef_subq1_u05(*in1, *in2);
    return 0;
}