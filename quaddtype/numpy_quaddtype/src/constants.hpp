#ifndef QUAD_CONSTANTS_HPP
#define QUAD_CONSTANTS_HPP

#ifdef __cplusplus
extern "C" {
#endif

#include <sleef.h>
#include <sleefquad.h>
#include <stdint.h>
#include <string.h>

// Quad precision constants using sleef_q macro
#define QUAD_PRECISION_ZERO sleef_q(+0x0000000000000LL, 0x0000000000000000ULL, -16383)
#define QUAD_PRECISION_ONE sleef_q(+0x1000000000000LL, 0x0000000000000000ULL, 0)
#define QUAD_PRECISION_INF sleef_q(+0x1000000000000LL, 0x0000000000000000ULL, 16384)
#define QUAD_PRECISION_NINF sleef_q(-0x1000000000000LL, 0x0000000000000000ULL, 16384)
#define QUAD_PRECISION_NAN sleef_q(+0x1ffffffffffffLL, 0xffffffffffffffffULL, 16384)

// Additional constants
#define QUAD_PRECISION_MAX_FINITE SLEEF_QUAD_MAX
#define QUAD_PRECISION_MIN_FINITE Sleef_negq1(SLEEF_QUAD_MAX)
#define QUAD_PRECISION_RADIX sleef_q(+0x1000000000000LL, 0x0000000000000000ULL, 1)  // 2.0

#ifdef SLEEF_QUAD_C
static const Sleef_quad SMALLEST_SUBNORMAL_VALUE = SLEEF_QUAD_DENORM_MIN;
#else
static const union {
    struct {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        uint64_t h, l;
#else
        uint64_t l, h;
#endif
    } parts;
    Sleef_quad value;
} smallest_subnormal_const = {.parts = {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
                                      .h = 0x0000000000000000ULL, .l = 0x0000000000000001ULL
#else
                                      .l = 0x0000000000000001ULL, .h = 0x0000000000000000ULL
#endif
                              }};
#define SMALLEST_SUBNORMAL_VALUE (smallest_subnormal_const.value)
#endif

// Integer constants for finfo
#define QUAD_NMANT 112           // mantissa bits (excluding implicit bit)
#define QUAD_MIN_EXP -16382      // minimum exponent for normalized numbers
#define QUAD_MAX_EXP 16384       // maximum exponent
#define QUAD_DECIMAL_DIGITS 33   // decimal digits of precision

typedef enum ConstantResultType {
    CONSTANT_QUAD,      // Sleef_quad value
    CONSTANT_INT64,     // int64_t value
    CONSTANT_ERROR      // Error occurred
} ConstantResultType;

typedef struct ConstantResult {
    ConstantResultType type;
    union {
        Sleef_quad quad_value;
        int64_t int_value;
    } data;
} ConstantResult;


static inline ConstantResult get_sleef_constant_by_name(const char* constant_name) {
    ConstantResult result;
    
    if (strcmp(constant_name, "pi") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SLEEF_M_PIq;
    }
    else if (strcmp(constant_name, "e") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SLEEF_M_Eq;
    }
    else if (strcmp(constant_name, "log2e") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SLEEF_M_LOG2Eq;
    }
    else if (strcmp(constant_name, "log10e") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SLEEF_M_LOG10Eq;
    }
    else if (strcmp(constant_name, "ln2") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SLEEF_M_LN2q;
    }
    else if (strcmp(constant_name, "ln10") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SLEEF_M_LN10q;
    }
    else if (strcmp(constant_name, "max_value") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SLEEF_QUAD_MAX;
    }
    else if (strcmp(constant_name, "epsilon") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SLEEF_QUAD_EPSILON;
    }
    else if (strcmp(constant_name, "smallest_normal") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SLEEF_QUAD_MIN;
    }
    else if (strcmp(constant_name, "smallest_subnormal") == 0) {
        result.type = CONSTANT_QUAD;
        result.data.quad_value = SMALLEST_SUBNORMAL_VALUE;
    }
    else if (strcmp(constant_name, "bits") == 0) {
        result.type = CONSTANT_INT64;
        result.data.int_value = sizeof(Sleef_quad) * 8;
    }
    else if (strcmp(constant_name, "precision") == 0) {
        result.type = CONSTANT_INT64;
        // precision = int(-log10(epsilon))
        result.data.int_value = 
            Sleef_cast_to_int64q1(Sleef_negq1(Sleef_log10q1_u10(SLEEF_QUAD_EPSILON)));
    }
    else if (strcmp(constant_name, "resolution") == 0) {
        result.type = CONSTANT_QUAD;
        // precision = int(-log10(epsilon))
        int64_t precision = 
            Sleef_cast_to_int64q1(Sleef_negq1(Sleef_log10q1_u10(SLEEF_QUAD_EPSILON)));
        // resolution = 10 ** (-precision)
        result.data.quad_value = 
            Sleef_powq1_u10(Sleef_cast_from_int64q1(10), Sleef_cast_from_int64q1(-precision));
    }
    else {
        result.type = CONSTANT_ERROR;
    }
    
    return result;
}

#ifdef __cplusplus
}
#endif

#endif // QUAD_CONSTANTS_HPP