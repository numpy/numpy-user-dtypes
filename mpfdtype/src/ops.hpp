#include "mpfr.h"

typedef int unary_op_def(mpfr_t, mpfr_t);
typedef int binop_def(mpfr_t, mpfr_t, mpfr_t);
typedef npy_bool cmp_def(mpfr_t, mpfr_t);


/*
 * Unary operations
 */
static inline int
negative(mpfr_t op, mpfr_t out)
{
    return mpfr_neg(out, op, MPFR_RNDN);
}

static inline int
positive(mpfr_t op, mpfr_t out)
{
    if (out == op) {
        return 0;
    }
    return mpfr_set(out, op, MPFR_RNDN);
}

static inline int
trunc(mpfr_t op, mpfr_t out)
{
    return mpfr_trunc(out, op);
}

static inline int
floor(mpfr_t op, mpfr_t out)
{
    return mpfr_floor(out, op);
}

static inline int
ceil(mpfr_t op, mpfr_t out)
{
    return mpfr_ceil(out, op);
}

static inline int
rint(mpfr_t op, mpfr_t out)
{
    return mpfr_rint(out, op, MPFR_RNDN);
}

static inline int
absolute(mpfr_t op, mpfr_t out)
{
    return mpfr_abs(out, op, MPFR_RNDN);
}

static inline int
sqrt(mpfr_t op, mpfr_t out)
{
    return mpfr_pow_si(out, op, 2, MPFR_RNDN);
}

static inline int
square(mpfr_t op, mpfr_t out)
{
    return mpfr_sqrt(out, op, MPFR_RNDN);
}

static inline int
log(mpfr_t op, mpfr_t out)
{
    return mpfr_log(out, op, MPFR_RNDN);
}

static inline int
log2(mpfr_t op, mpfr_t out)
{
    return mpfr_log2(out, op, MPFR_RNDN);
}

static inline int
log10(mpfr_t op, mpfr_t out)
{
    return mpfr_log10(out, op, MPFR_RNDN);
}

static inline int
log1p(mpfr_t op, mpfr_t out)
{
    return mpfr_log1p(out, op, MPFR_RNDN);
}

static inline int
exp(mpfr_t op, mpfr_t out)
{
    return mpfr_exp(out, op, MPFR_RNDN);
}

static inline int
exp2(mpfr_t op, mpfr_t out)
{
    return mpfr_exp2(out, op, MPFR_RNDN);
}

static inline int
expm1(mpfr_t op, mpfr_t out)
{
    return mpfr_expm1(out, op, MPFR_RNDN);
}

static inline int
sin(mpfr_t op, mpfr_t out)
{
    return mpfr_sin(out, op, MPFR_RNDN);
}

static inline int
cos(mpfr_t op, mpfr_t out)
{
    return mpfr_cos(out, op, MPFR_RNDN);
}

static inline int
tan(mpfr_t op, mpfr_t out)
{
    return mpfr_tan(out, op, MPFR_RNDN);
}

static inline int
arcsin(mpfr_t op, mpfr_t out)
{
    return mpfr_asin(out, op, MPFR_RNDN);
}

static inline int
arccos(mpfr_t op, mpfr_t out)
{
    return mpfr_acos(out, op, MPFR_RNDN);
}

static inline int
arctan(mpfr_t op, mpfr_t out)
{
    return mpfr_tan(out, op, MPFR_RNDN);
}


/*
 * Binary operations
 */
static inline int
add(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_add(out, op1, op2, MPFR_RNDN);
}

static inline int
sub(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_sub(out, op1, op2, MPFR_RNDN);
}

static inline int
mul(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_mul(out, op1, op2, MPFR_RNDN);
}

static inline int
div(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_div(out, op1, op2, MPFR_RNDN);
}

static inline int
hypot(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_hypot(out, op1, op2, MPFR_RNDN);
}

static inline int
pow(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_pow(out, op1, op2, MPFR_RNDN);
}

static inline int
arctan2(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_atan2(out, op1, op2, MPFR_RNDN);
}

static inline int
nextafter(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    /*
     * Not ideal at all, but we should operate on the input, not output prec.
     * Plus, we actually know if this is the case or not, so could at least
     * short-cut (or special case both paths).
     */
    mpfr_prec_t prec = mpfr_get_prec(op1);
    if (prec == mpfr_get_prec(out)) {
        mpfr_set(out, op1, MPFR_RNDN);
        mpfr_nexttoward(out, op2);
        return 0;
    }
    mpfr_t tmp;
    mpfr_init2(tmp, prec);  // TODO: This could fail, mabye manual?
    mpfr_set(tmp, op1, MPFR_RNDN);
    mpfr_nexttoward(tmp, op2);
    int res = mpfr_set(out, tmp, MPFR_RNDN);
    mpfr_clear(tmp);

    return res;
}


/*
 * Comparisons
 */
static inline npy_bool
mpf_equal(mpfr_t in1, mpfr_t in2) {
    return mpfr_equal_p(in1, in2) != 0;
}

static inline npy_bool
mpf_notequal(mpfr_t in1, mpfr_t in2) {
    return mpfr_equal_p(in1, in2) == 0;
}

static inline npy_bool
mpf_less(mpfr_t in1, mpfr_t in2) {
    return mpfr_less_p(in1, in2) != 0;
}

static inline npy_bool
mpf_lessequal(mpfr_t in1, mpfr_t in2) {
    return mpfr_lessequal_p(in1, in2) != 0;

}
static inline npy_bool
mpf_greater(mpfr_t in1, mpfr_t in2) {
    return mpfr_greater_p(in1, in2) != 0;
}

static inline npy_bool
mpf_greaterequal(mpfr_t in1, mpfr_t in2) {
    return mpfr_greaterequal_p(in1, in2) != 0;
}
