
#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY

extern "C" {
    #include <Python.h>

    #include "numpy/arrayobject.h"
    #include "numpy/ndarraytypes.h"
    #include "numpy/experimental_dtype_api.h"
}

#include <vector>
#include "mpfr.h"

#include "scalar.h"
#include "casts.h"
#include "dtype.h"


static NPY_CASTING
mpf_to_mpf_resolve_descriptors(
        PyObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        MPFDTypeObject *given_descrs[2],
        MPFDTypeObject *loop_descrs[2],
        npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    if (loop_descrs[0]->precision == loop_descrs[1]->precision) {
        return NPY_EQUIV_CASTING;
    }
    if (loop_descrs[0]->precision < loop_descrs[1]->precision) {
        return NPY_SAFE_CASTING;
    }

    return NPY_SAME_KIND_CASTING;
}


static int
mpf_to_mof_strided_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    mpfr_prec_t prec_in = ((MPFDTypeObject *)context->descriptors[0])->precision;
    mpfr_prec_t prec_out = ((MPFDTypeObject *)context->descriptors[1])->precision;

    mpfr_ptr in, out;

    while (N--) {
        mpf_load(in, in_ptr, prec_in);
        mpf_load(out, out_ptr, prec_out);

        mpfr_set(out, in, MPFR_RNDN);
        mpf_store(out_ptr, out);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}


/*
 * Cast functions/helpers.  We use only the max-precision integer ones here
 * for simplicity.  MPFR has more.
 */
template <typename conv_T, typename T>
int C_to_mpfr(T val, mpfr_t res);

template<> int
C_to_mpfr<uintmax_t, npy_ubyte>(npy_ubyte val, mpfr_t res) {return mpfr_set_uj(res, val, MPFR_RNDN);}
template<> int
C_to_mpfr<uintmax_t, npy_ushort>(npy_ushort val, mpfr_t res) {return mpfr_set_uj(res, val, MPFR_RNDN);}
template<> int
C_to_mpfr<uintmax_t, npy_uint>(npy_uint val, mpfr_t res) {return mpfr_set_uj(res, val, MPFR_RNDN);}
template<> int
C_to_mpfr<uintmax_t, npy_ulong>(npy_ulong val, mpfr_t res) {return mpfr_set_uj(res, val, MPFR_RNDN);}
template<> int
C_to_mpfr<uintmax_t, npy_ulonglong>(npy_ulonglong val, mpfr_t res) {return mpfr_set_uj(res, val, MPFR_RNDN);}

template<> int
C_to_mpfr<intmax_t, npy_byte>(npy_byte val, mpfr_t res) {return mpfr_set_sj(res, val, MPFR_RNDN);}
template<> int
C_to_mpfr<intmax_t, npy_short>(npy_short val, mpfr_t res) {return mpfr_set_sj(res, val, MPFR_RNDN);}
template<> int
C_to_mpfr<intmax_t, npy_int>(npy_int val, mpfr_t res) {return mpfr_set_sj(res, val, MPFR_RNDN);}
template<> int
C_to_mpfr<intmax_t, npy_long>(npy_long val, mpfr_t res) {return mpfr_set_sj(res, val, MPFR_RNDN);}
template<> int
C_to_mpfr<intmax_t, npy_longlong>(npy_longlong val, mpfr_t res) {return mpfr_set_sj(res, val, MPFR_RNDN);}


template<> int
C_to_mpfr<float, float>(float val, mpfr_t res)
{
    return mpfr_set_flt(res, val, MPFR_RNDN);
}

template<> int
C_to_mpfr<double, double>(double val, mpfr_t res)
{
    return mpfr_set_d(res, val, MPFR_RNDN);
}

template<> int
C_to_mpfr<long double, long double>(long double val, mpfr_t res)
{
    return mpfr_set_ld(res, val, MPFR_RNDN);
}

template<> int
C_to_mpfr<bool, npy_bool>(npy_bool val, mpfr_t res)
{
    return mpfr_set_si(res, val != 0, MPFR_RNDN);
}


template <int precision>
static NPY_CASTING
numpy_to_mpf_resolve_descriptors(
        PyObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = (PyArray_Descr *)new_MPFDType_instance(precision);
        if (loop_descrs[1] == nullptr) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    /* Never fails for builtin NumPy: */
    loop_descrs[0] = PyArray_GetDefaultDescr(dtypes[0]);

    if (precision <= ((MPFDTypeObject *)loop_descrs[1])->precision) {
        return NPY_SAFE_CASTING;
    }

    return NPY_SAME_KIND_CASTING;
}


template <typename T, typename conv_T>
static int
numpy_to_mpf_strided_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    mpfr_prec_t prec_out = ((MPFDTypeObject *)context->descriptors[1])->precision;

    mpfr_ptr out;

    while (N--) {
        T *in = (T *)in_ptr;
        mpf_load(out, out_ptr, prec_out);

        int res = C_to_mpfr<conv_T, T>(*in, out);
        // TODO: At least for ints, could flag out of bounds...
        mpf_store(out_ptr, out);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}


/*
 * Casts back to NumPy
 */

template <typename T, typename conv_T>
int mpfr_to_C(mpfr_t val, T *res);

// TODO: Is there a way to do this partial templating?
template <typename UINT>
int
mpfr_to_uint(mpfr_t val, UINT *res)
{
    if (mpfr_fits_intmax_p(val, MPFR_RNDN)) {
        uintmax_t mval = mpfr_get_uj(val, MPFR_RNDN);
        *res = mval;
        if (mval == *res) {
            return 0;
        }
    }

    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    PyErr_SetString(PyExc_ValueError,
            "MPFloat value too large to be converted to unsigned integer.");
    NPY_DISABLE_C_API;
    return -1;
}

template<>
int mpfr_to_C<npy_ubyte, uintmax_t>(mpfr_t val, npy_ubyte *res) {return mpfr_to_uint(val, res);}
template<>
int mpfr_to_C<npy_ushort, uintmax_t>(mpfr_t val, npy_ushort *res) {return mpfr_to_uint(val, res);}
template<>
int mpfr_to_C<npy_uint, uintmax_t>(mpfr_t val, npy_uint *res) {return mpfr_to_uint(val, res);}
template<>
int mpfr_to_C<npy_ulong, uintmax_t>(mpfr_t val, npy_ulong *res) {return mpfr_to_uint(val, res);}
template<>
int mpfr_to_C<npy_ulonglong, uintmax_t>(mpfr_t val, npy_ulonglong *res) {return mpfr_to_uint(val, res);}


template <typename INT>
int
mpfr_to_int(mpfr_t val, INT *res)
{
    if (mpfr_fits_intmax_p(val, MPFR_RNDN)) {
        intmax_t mval = mpfr_get_sj(val, MPFR_RNDN);
        *res = mval;
        if (mval == *res) {
            return 0;
        }
    }

    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    PyErr_SetString(PyExc_ValueError,
            "MPFloat value too large to be converted to integer.");
    NPY_DISABLE_C_API;
    return -1;
}

template<>
int mpfr_to_C<npy_byte, intmax_t>(mpfr_t val, npy_byte *res) {return mpfr_to_int(val, res);}
template<>
int mpfr_to_C<npy_short, intmax_t>(mpfr_t val, npy_short *res) {return mpfr_to_int(val, res);}
template<>
int mpfr_to_C<npy_int, intmax_t>(mpfr_t val, npy_int *res) {return mpfr_to_int(val, res);}
template<>
int mpfr_to_C<npy_long, intmax_t>(mpfr_t val, npy_long *res) {return mpfr_to_int(val, res);}
template<>
int mpfr_to_C<npy_longlong, intmax_t>(mpfr_t val, npy_longlong *res) {return mpfr_to_int(val, res);}


template<> int
mpfr_to_C<float, float>(mpfr_t val, float *res)
{
    *res = mpfr_get_flt(val, MPFR_RNDN);
    return 0;
}

template<> int
mpfr_to_C<double, double>(mpfr_t val, double *res)
{
    *res = mpfr_get_d(val, MPFR_RNDN);
    return 0;
}

template<> int
mpfr_to_C<long double, long double>(mpfr_t val, long double *res)
{
    *res = mpfr_get_ld(val, MPFR_RNDN);
    return 0;
}

template<> int
mpfr_to_C<npy_bool, bool>(mpfr_t val, npy_bool *res)
{
    // TODO: This is god awful, but should work. C++ bool may not be correct
    //       (I think it may be int size), so we reinterpret cast here.
    *res = !mpfr_zero_p(val);
    return 0;
}


template <typename T, typename conv_T>
static int
mpf_to_numpy_strided_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    mpfr_prec_t prec_in = ((MPFDTypeObject *)context->descriptors[0])->precision;

    mpfr_ptr in;

    while (N--) {
        mpf_load(in, in_ptr, prec_in);
        T *out = (T *)out_ptr;

        mpfr_to_C<T, conv_T>(in, out);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}


/*
 * Just define them statically, we do clear them anyway (since they are really)
 * not needed...
 */
static std::vector<PyArrayMethod_Spec *>specs;


template <typename T, typename conv_T>
void
add_cast_from(PyArray_DTypeMeta *to)
{
    PyArray_DTypeMeta **dtypes = new PyArray_DTypeMeta *[2]{nullptr, to};

    PyType_Slot *slots = new PyType_Slot [4]{
        {NPY_METH_strided_loop,
            (void *)&mpf_to_numpy_strided_loop<T, conv_T>},
        {0, nullptr}
    };

    specs.push_back(new PyArrayMethod_Spec {
        .name = "cast_MPF_to_NumPy",
        .nin = 1,
        .nout = 1,
        /* Always unsafe, at least the exponent has larger range... */
        .casting = NPY_UNSAFE_CASTING,
        .flags = (NPY_ARRAYMETHOD_FLAGS)0,
        .dtypes = dtypes,
        .slots = slots,
    });
}


template <typename T, typename conv_T, int precision>
void
add_cast_to(PyArray_DTypeMeta *from)
{
    PyArray_DTypeMeta **dtypes = new PyArray_DTypeMeta *[2]{from, nullptr};

    PyType_Slot *slots = new PyType_Slot [4]{
        {NPY_METH_resolve_descriptors,
            (void *)&numpy_to_mpf_resolve_descriptors<precision>},
        {NPY_METH_strided_loop,
            (void *)&numpy_to_mpf_strided_loop<T, conv_T>},
        {0, nullptr}
    };

    specs.push_back(new PyArrayMethod_Spec {
        .name = "cast_NumPy_to_MPF",
        .nin = 1,
        .nout = 1,
        .casting = NPY_SAME_KIND_CASTING,
        .flags = (NPY_ARRAYMETHOD_FLAGS)0,
        .dtypes = dtypes,
        .slots = slots,
    });
}


/*
 * My C++ is not good enough to know whether the memory management and
 * error handling, etc. is remotely correct.  I also suspect this can be
 * done nicer...
 */
PyArrayMethod_Spec **
init_casts_internal(void)
{
    PyArray_DTypeMeta **mpf2mpf_dtypes = new PyArray_DTypeMeta *[2]{nullptr, nullptr};

    PyType_Slot *mpf2mpf_slots = new PyType_Slot [4]{
        {NPY_METH_resolve_descriptors,
            (void *)&mpf_to_mpf_resolve_descriptors},
        {NPY_METH_strided_loop, (void *)&mpf_to_mof_strided_loop},
        /* We don't actually support unaligned access... */
        {NPY_METH_unaligned_strided_loop, (void *)&mpf_to_mof_strided_loop},
        {0, nullptr}
    };

    specs.push_back(new PyArrayMethod_Spec {
        .name = "cast_MPF_to_MPF",
        .nin = 1,
        .nout = 1,
        .casting = NPY_SAME_KIND_CASTING,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,  /* not really ... */
        .dtypes = mpf2mpf_dtypes,
        .slots = mpf2mpf_slots,
    });

    add_cast_to<npy_bool, bool, 1>(&PyArray_BoolDType);

    add_cast_to<npy_byte, intmax_t, 8*NPY_SIZEOF_BYTE>(&PyArray_ByteDType);
    add_cast_to<npy_ubyte, uintmax_t, 8*NPY_SIZEOF_BYTE>(&PyArray_UByteDType);
    add_cast_to<npy_short, intmax_t, 8*NPY_SIZEOF_SHORT>(&PyArray_ShortDType);
    add_cast_to<npy_ushort, uintmax_t, 8*NPY_SIZEOF_SHORT>(&PyArray_UShortDType);
    add_cast_to<npy_int, intmax_t, 8*NPY_SIZEOF_INT>(&PyArray_IntDType);
    add_cast_to<npy_uint, uintmax_t, 8*NPY_SIZEOF_INT>(&PyArray_UIntDType);
    add_cast_to<npy_long, intmax_t, 8*NPY_SIZEOF_LONG>(&PyArray_LongDType);
    add_cast_to<npy_ulong, uintmax_t, 8*NPY_SIZEOF_LONG>(&PyArray_ULongDType);
    add_cast_to<npy_longlong, intmax_t, 8*NPY_SIZEOF_LONGLONG>(&PyArray_LongLongDType);
    add_cast_to<npy_ulonglong, uintmax_t, 8*NPY_SIZEOF_LONGLONG>(&PyArray_ULongLongDType);

    add_cast_to<float, float, 24>(&PyArray_FloatDType);
    add_cast_to<double, double, 53>(&PyArray_DoubleDType);
    // TODO: This is 80 bit for extended precision and otherwise not correct
    add_cast_to<long double, long double, 80>(&PyArray_LongDoubleDType);

    add_cast_from<npy_bool, bool>(&PyArray_BoolDType);

    add_cast_from<npy_byte, intmax_t>(&PyArray_ByteDType);
    add_cast_from<npy_ubyte, uintmax_t>(&PyArray_UByteDType);
    add_cast_from<npy_short, intmax_t>(&PyArray_ShortDType);
    add_cast_from<npy_ushort, uintmax_t>(&PyArray_UShortDType);
    add_cast_from<npy_int, intmax_t>(&PyArray_IntDType);
    add_cast_from<npy_uint, uintmax_t>(&PyArray_UIntDType);
    add_cast_from<npy_long, intmax_t>(&PyArray_LongDType);
    add_cast_from<npy_ulong, uintmax_t>(&PyArray_ULongDType);
    add_cast_from<npy_longlong, intmax_t>(&PyArray_LongLongDType);
    add_cast_from<npy_ulonglong, uintmax_t>(&PyArray_ULongLongDType);

    add_cast_from<float, float>(&PyArray_FloatDType);
    add_cast_from<double, double>(&PyArray_DoubleDType);
    add_cast_from<long double, long double>(&PyArray_LongDoubleDType);

    specs.push_back(nullptr);
    return specs.data();
}


PyArrayMethod_Spec **
init_casts(void)
{
    try {
        return init_casts_internal();
    }
    catch (int e) {
        /* Must be a memory error */
        PyErr_NoMemory();
        return nullptr;
    }
}


void
free_casts(void)
{
    for (auto cast : specs) {
        if (cast == nullptr) {
            continue;
        }
        delete cast->dtypes;
        delete cast->slots;
        delete cast;
    }
    specs.clear();
}
