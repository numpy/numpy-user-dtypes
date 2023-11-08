#include <Python.h>

#include "umath.h"

#include "dtype.h"
#include "static_string.h"

static NPY_CASTING
multiply_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *dtypes[], PyArray_Descr *given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    PyArray_Descr *ldescr = given_descrs[0];
    PyArray_Descr *rdescr = given_descrs[1];
    StringDTypeObject *odescr = NULL;
    PyArray_Descr *out_descr = NULL;

    if (dtypes[0] == (PyArray_DTypeMeta *)&StringDType) {
        odescr = (StringDTypeObject *)ldescr;
    }
    else {
        odescr = (StringDTypeObject *)rdescr;
    }

    if (given_descrs[2] == NULL) {
        out_descr = (PyArray_Descr *)new_stringdtype_instance(
                odescr->na_object, odescr->coerce);
        if (out_descr == NULL) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[2]);
        out_descr = given_descrs[2];
    }

    Py_INCREF(ldescr);
    loop_descrs[0] = ldescr;
    Py_INCREF(rdescr);
    loop_descrs[1] = rdescr;
    loop_descrs[2] = out_descr;

    return NPY_NO_CASTING;
}

#define MULTIPLY_IMPL(shortname)                                              \
    static int multiply_loop_core_##shortname(                                \
            npy_intp N, char *sin, char *iin, char *out, npy_intp s_stride,   \
            npy_intp i_stride, npy_intp o_stride, int has_null,               \
            int has_nan_na, int has_string_na,                                \
            const npy_static_string *default_string,                          \
            StringDTypeObject *idescr, StringDTypeObject *odescr)             \
    {                                                                         \
        NPY_STRING_ACQUIRE_ALLOCATOR2(idescr, odescr);                        \
        while (N--) {                                                         \
            const npy_packed_static_string *ips =                             \
                    (npy_packed_static_string *)sin;                          \
            npy_static_string is = {0, NULL};                                 \
            npy_packed_static_string *ops = (npy_packed_static_string *)out;  \
            int is_isnull = npy_string_load(idescr->allocator, ips, &is);     \
            if (is_isnull == -1) {                                            \
                gil_error(PyExc_MemoryError,                                  \
                          "Failed to load string in multiply");               \
                goto fail;                                                    \
            }                                                                 \
            else if (is_isnull) {                                             \
                if (has_nan_na) {                                             \
                    if (npy_string_free(ops, odescr->allocator) < 0) {        \
                        gil_error(PyExc_MemoryError,                          \
                                  "Failed to deallocate string in multiply"); \
                        goto fail;                                            \
                    }                                                         \
                                                                              \
                    *ops = *NPY_NULL_STRING;                                  \
                    sin += s_stride;                                          \
                    iin += i_stride;                                          \
                    out += o_stride;                                          \
                    continue;                                                 \
                }                                                             \
                else if (has_string_na || !has_null) {                        \
                    is = *(npy_static_string *)default_string;                \
                }                                                             \
                else {                                                        \
                    gil_error(PyExc_TypeError,                                \
                              "Cannot multiply null that is not a nan-like "  \
                              "value");                                       \
                    goto fail;                                                \
                }                                                             \
            }                                                                 \
            npy_##shortname factor = *(npy_##shortname *)iin;                 \
            size_t cursize = is.size;                                         \
            size_t newsize = cursize * factor;                                \
            /* check for overflow */                                          \
            if (newsize < cursize) {                                          \
                gil_error(PyExc_MemoryError,                                  \
                          "Failed to allocate string in string mutiply");     \
                goto fail;                                                    \
            }                                                                 \
                                                                              \
            char *buf = NULL;                                                 \
            npy_static_string os = {0, NULL};                                 \
            if (odescr == idescr) {                                           \
                buf = PyMem_RawMalloc(newsize);                               \
                if (buf == NULL) {                                            \
                    gil_error(PyExc_MemoryError,                              \
                              "Failed to allocate string in multiply");       \
                    goto fail;                                                \
                }                                                             \
            }                                                                 \
            else {                                                            \
                if (npy_string_free(ops, odescr->allocator) < 0) {            \
                    gil_error(PyExc_MemoryError,                              \
                              "Failed to deallocate string in multiply");     \
                    goto fail;                                                \
                }                                                             \
                if (npy_string_newemptysize(newsize, ops,                     \
                                            odescr->allocator) < 0) {         \
                    gil_error(PyExc_MemoryError,                              \
                              "Failed to allocate string in multiply");       \
                    goto fail;                                                \
                }                                                             \
                if (npy_string_load(odescr->allocator, ops, &os) < 0) {       \
                    gil_error(PyExc_MemoryError,                              \
                              "Failed to load string in multiply");           \
                    goto fail;                                                \
                }                                                             \
                /* excplicitly discard const; initializing new buffer */      \
                buf = (char *)os.buf;                                         \
            }                                                                 \
                                                                              \
            for (size_t i = 0; i < (size_t)factor; i++) {                     \
                /* multiply can't overflow because cursize * factor */        \
                /* has already been checked and doesn't overflow */           \
                memcpy((char *)buf + i * cursize, is.buf, cursize);           \
            }                                                                 \
                                                                              \
            if (idescr == odescr) {                                           \
                if (npy_string_free(ops, odescr->allocator) < 0) {            \
                    gil_error(PyExc_MemoryError,                              \
                              "Failed to deallocate string in multiply");     \
                    goto fail;                                                \
                }                                                             \
                                                                              \
                if (npy_string_newsize(buf, newsize, ops,                     \
                                       odescr->allocator) < 0) {              \
                    gil_error(PyExc_MemoryError,                              \
                              "Failed to allocate string in multiply");       \
                    goto fail;                                                \
                }                                                             \
                                                                              \
                PyMem_RawFree(buf);                                           \
            }                                                                 \
                                                                              \
            sin += s_stride;                                                  \
            iin += i_stride;                                                  \
            out += o_stride;                                                  \
        }                                                                     \
        NPY_STRING_RELEASE_ALLOCATOR(idescr);                                 \
        NPY_STRING_RELEASE_ALLOCATOR(odescr);                                 \
        return 0;                                                             \
                                                                              \
    fail:                                                                     \
        NPY_STRING_RELEASE_ALLOCATOR(idescr);                                 \
        NPY_STRING_RELEASE_ALLOCATOR(odescr);                                 \
        return -1;                                                            \
    }                                                                         \
                                                                              \
    static int multiply_right_##shortname##_strided_loop(                     \
            PyArrayMethod_Context *context, char *const data[],               \
            npy_intp const dimensions[], npy_intp const strides[],            \
            NpyAuxData *NPY_UNUSED(auxdata))                                  \
    {                                                                         \
        StringDTypeObject *idescr =                                           \
                (StringDTypeObject *)context->descriptors[0];                 \
        StringDTypeObject *odescr =                                           \
                (StringDTypeObject *)context->descriptors[2];                 \
        int has_null = odescr->na_object != NULL;                             \
        int has_nan_na = odescr->has_nan_na;                                  \
        int has_string_na = odescr->has_string_na;                            \
        const npy_static_string *default_string = &idescr->default_string;    \
        npy_intp N = dimensions[0];                                           \
        char *in1 = data[0];                                                  \
        char *in2 = data[1];                                                  \
        char *out = data[2];                                                  \
        npy_intp in1_stride = strides[0];                                     \
        npy_intp in2_stride = strides[1];                                     \
        npy_intp out_stride = strides[2];                                     \
                                                                              \
        return multiply_loop_core_##shortname(                                \
                N, in1, in2, out, in1_stride, in2_stride, out_stride,         \
                has_null, has_nan_na, has_string_na, default_string, idescr,  \
                odescr);                                                      \
    }                                                                         \
                                                                              \
    static int multiply_left_##shortname##_strided_loop(                      \
            PyArrayMethod_Context *context, char *const data[],               \
            npy_intp const dimensions[], npy_intp const strides[],            \
            NpyAuxData *NPY_UNUSED(auxdata))                                  \
    {                                                                         \
        StringDTypeObject *idescr =                                           \
                (StringDTypeObject *)context->descriptors[1];                 \
        StringDTypeObject *odescr =                                           \
                (StringDTypeObject *)context->descriptors[2];                 \
        int has_null = idescr->na_object != NULL;                             \
        int has_nan_na = idescr->has_nan_na;                                  \
        int has_string_na = idescr->has_string_na;                            \
        const npy_static_string *default_string = &idescr->default_string;    \
        npy_intp N = dimensions[0];                                           \
        char *in1 = data[0];                                                  \
        char *in2 = data[1];                                                  \
        char *out = data[2];                                                  \
        npy_intp in1_stride = strides[0];                                     \
        npy_intp in2_stride = strides[1];                                     \
        npy_intp out_stride = strides[2];                                     \
                                                                              \
        return multiply_loop_core_##shortname(                                \
                N, in2, in1, out, in2_stride, in1_stride, out_stride,         \
                has_null, has_nan_na, has_string_na, default_string, idescr,  \
                odescr);                                                      \
    }

MULTIPLY_IMPL(int8);
MULTIPLY_IMPL(int16);
MULTIPLY_IMPL(int32);
MULTIPLY_IMPL(int64);
MULTIPLY_IMPL(uint8);
MULTIPLY_IMPL(uint16);
MULTIPLY_IMPL(uint32);
MULTIPLY_IMPL(uint64);
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
MULTIPLY_IMPL(byte);
MULTIPLY_IMPL(ubyte);
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
MULTIPLY_IMPL(short);
MULTIPLY_IMPL(ushort);
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
MULTIPLY_IMPL(long);
MULTIPLY_IMPL(ulong);
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
MULTIPLY_IMPL(longlong);
MULTIPLY_IMPL(ulonglong);
#endif

static NPY_CASTING
binary_resolve_descriptors(struct PyArrayMethodObject_tag *NPY_UNUSED(method),
                           PyArray_DTypeMeta *NPY_UNUSED(dtypes[]),
                           PyArray_Descr *given_descrs[],
                           PyArray_Descr *loop_descrs[],
                           npy_intp *NPY_UNUSED(view_offset))
{
    StringDTypeObject *descr1 = (StringDTypeObject *)given_descrs[0];
    StringDTypeObject *descr2 = (StringDTypeObject *)given_descrs[1];

    // RichCompareBool has a short-circuit pointer comparison fast path.
    int eq_res = _eq_comparison(descr1->coerce, descr2->coerce,
                                descr1->na_object, descr2->na_object);

    if (eq_res < 0) {
        return (NPY_CASTING)-1;
    }

    if (eq_res != 1) {
        PyErr_SetString(PyExc_TypeError,
                        "Can only do binary operations with equal StringDType "
                        "instances.");
        return (NPY_CASTING)-1;
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    PyArray_Descr *out_descr = NULL;

    if (given_descrs[2] == NULL) {
        out_descr = (PyArray_Descr *)new_stringdtype_instance(
                ((StringDTypeObject *)given_descrs[1])->na_object,
                ((StringDTypeObject *)given_descrs[1])->coerce);

        if (out_descr == NULL) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[2]);
        out_descr = given_descrs[2];
    }

    loop_descrs[2] = out_descr;

    return NPY_NO_CASTING;
}

static int
add_strided_loop(PyArrayMethod_Context *context, char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *s1descr = (StringDTypeObject *)context->descriptors[0];
    StringDTypeObject *s2descr = (StringDTypeObject *)context->descriptors[1];
    StringDTypeObject *odescr = (StringDTypeObject *)context->descriptors[2];
    int has_null = s1descr->na_object != NULL;
    int has_nan_na = s1descr->has_nan_na;
    int has_string_na = s1descr->has_string_na;
    const npy_static_string *default_string = &s1descr->default_string;
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    NPY_STRING_ACQUIRE_ALLOCATOR3(s1descr, s2descr, odescr);

    while (N--) {
        const npy_packed_static_string *ps1 = (npy_packed_static_string *)in1;
        npy_static_string s1 = {0, NULL};
        int s1_isnull = npy_string_load(s1descr->allocator, ps1, &s1);
        const npy_packed_static_string *ps2 = (npy_packed_static_string *)in2;
        npy_static_string s2 = {0, NULL};
        int s2_isnull = npy_string_load(s2descr->allocator, ps2, &s2);
        if (s1_isnull == -1 || s2_isnull == -1) {
            gil_error(PyExc_MemoryError, "Failed to load string in add");
            goto fail;
        }
        npy_packed_static_string *ops = (npy_packed_static_string *)out;
        if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            if (has_nan_na) {
                if (npy_string_free(ops, odescr->allocator) < 0) {
                    gil_error(PyExc_MemoryError,
                              "Failed to deallocate string in add");
                    goto fail;
                }
                *ops = *NPY_NULL_STRING;
                goto next_step;
            }
            else if (has_string_na || !has_null) {
                if (s1_isnull) {
                    s1 = *default_string;
                }
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
            else {
                gil_error(PyExc_TypeError,
                          "Cannot add null that is not a nan-like value");
                goto fail;
            }
        }

        // check for overflow
        size_t newsize = s1.size + s2.size;
        if (newsize < s1.size) {
            gil_error(PyExc_MemoryError, "Failed to allocate string in add");
            goto fail;
        }

        char *buf = NULL;
        npy_static_string os = {0, NULL};
        if (odescr == s1descr || odescr == s2descr) {
            buf = PyMem_RawMalloc(newsize);

            if (buf == NULL) {
                gil_error(PyExc_MemoryError,
                          "Failed to allocate string in add");
                goto fail;
            }
        }
        else {
            if (npy_string_free(ops, odescr->allocator) < 0) {
                gil_error(PyExc_MemoryError,
                          "Failed to deallocate string in add");
                goto fail;
            }
            if (npy_string_newemptysize(newsize, ops, odescr->allocator) < 0) {
                gil_error(PyExc_MemoryError,
                          "Failed to allocate string in add");
                goto fail;
            }
            if (npy_string_load(odescr->allocator, ops, &os) < 0) {
                gil_error(PyExc_MemoryError, "Failed to load string in add");
                goto fail;
            }
            // excplicitly discard const; initializing new buffer
            buf = (char *)os.buf;
        }

        memcpy(buf, s1.buf, s1.size);
        memcpy(buf + s1.size, s2.buf, s2.size);

        if (s1descr == odescr || s2descr == odescr) {
            if (npy_string_free(ops, odescr->allocator) < 0) {
                gil_error(PyExc_MemoryError,
                          "Failed to deallocate string in add");
                goto fail;
            }

            if (npy_string_newsize(buf, newsize, ops, odescr->allocator) < 0) {
                gil_error(PyExc_MemoryError,
                          "Failed to allocate string in add");
                goto fail;
            }

            PyMem_RawFree(buf);
        }

    next_step:
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }
    NPY_STRING_RELEASE_ALLOCATOR(s1descr);
    NPY_STRING_RELEASE_ALLOCATOR(s2descr);
    NPY_STRING_RELEASE_ALLOCATOR(odescr);
    return 0;

fail:
    NPY_STRING_RELEASE_ALLOCATOR(s1descr);
    NPY_STRING_RELEASE_ALLOCATOR(s2descr);
    NPY_STRING_RELEASE_ALLOCATOR(odescr);
    return -1;
}

static int
maximum_strided_loop(PyArrayMethod_Context *context, char *const data[],
                     npy_intp const dimensions[], npy_intp const strides[],
                     NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *in1_descr =
            ((StringDTypeObject *)context->descriptors[0]);
    StringDTypeObject *in2_descr =
            ((StringDTypeObject *)context->descriptors[1]);
    StringDTypeObject *out_descr =
            ((StringDTypeObject *)context->descriptors[2]);
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    NPY_STRING_ACQUIRE_ALLOCATOR3(in1_descr, in2_descr, out_descr);

    while (N--) {
        const npy_packed_static_string *sin1 = (npy_packed_static_string *)in1;
        const npy_packed_static_string *sin2 = (npy_packed_static_string *)in2;
        npy_packed_static_string *sout = (npy_packed_static_string *)out;
        if (_compare(in1, in2, in1_descr, in2_descr) > 0) {
            // if in and out are the same address, do nothing to avoid a
            // use-after-free
            if (in1 != out) {
                if (free_and_copy(in1_descr->allocator, out_descr->allocator,
                                  sin1, sout, "maximum") == -1) {
                    goto fail;
                }
            }
        }
        else {
            if (in2 != out) {
                if (free_and_copy(in2_descr->allocator, out_descr->allocator,
                                  sin2, sout, "maximum") == -1) {
                    goto fail;
                }
            }
        }
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    NPY_STRING_RELEASE_ALLOCATOR(in1_descr);
    NPY_STRING_RELEASE_ALLOCATOR(in2_descr);
    NPY_STRING_RELEASE_ALLOCATOR(out_descr);
    return 0;

fail:
    NPY_STRING_RELEASE_ALLOCATOR(in1_descr);
    NPY_STRING_RELEASE_ALLOCATOR(in2_descr);
    NPY_STRING_RELEASE_ALLOCATOR(out_descr);
    return -1;
}

static int
minimum_strided_loop(PyArrayMethod_Context *context, char *const data[],
                     npy_intp const dimensions[], npy_intp const strides[],
                     NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *in1_descr =
            ((StringDTypeObject *)context->descriptors[0]);
    StringDTypeObject *in2_descr =
            ((StringDTypeObject *)context->descriptors[1]);
    StringDTypeObject *out_descr =
            ((StringDTypeObject *)context->descriptors[2]);
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    NPY_STRING_ACQUIRE_ALLOCATOR3(in1_descr, in2_descr, out_descr);

    while (N--) {
        const npy_packed_static_string *sin1 = (npy_packed_static_string *)in1;
        const npy_packed_static_string *sin2 = (npy_packed_static_string *)in2;
        npy_packed_static_string *sout = (npy_packed_static_string *)out;
        if (_compare(in1, in2, in1_descr, in2_descr) < 0) {
            // if in and out are the same address, do nothing to avoid a
            // use-after-free
            if (in1 != out) {
                if (free_and_copy(in1_descr->allocator, out_descr->allocator,
                                  sin1, sout, "minimum") == -1) {
                    goto fail;
                }
            }
        }
        else {
            if (in2 != out) {
                if (free_and_copy(in2_descr->allocator, out_descr->allocator,
                                  sin2, sout, "minimum") == -1) {
                    goto fail;
                }
            }
        }
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    NPY_STRING_RELEASE_ALLOCATOR(in1_descr);
    NPY_STRING_RELEASE_ALLOCATOR(in2_descr);
    NPY_STRING_RELEASE_ALLOCATOR(out_descr);
    return 0;

fail:
    NPY_STRING_RELEASE_ALLOCATOR(in1_descr);
    NPY_STRING_RELEASE_ALLOCATOR(in2_descr);
    NPY_STRING_RELEASE_ALLOCATOR(out_descr);
    return -1;
}

static int
string_equal_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[],
                          npy_intp const strides[],
                          NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *descr1 = (StringDTypeObject *)context->descriptors[0];
    StringDTypeObject *descr2 = (StringDTypeObject *)context->descriptors[1];
    int has_null = descr1->na_object != NULL;
    int has_nan_na = descr1->has_nan_na;
    int has_string_na = descr1->has_string_na;
    const npy_static_string *default_string = &descr1->default_string;
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    npy_bool *out = (npy_bool *)data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    NPY_STRING_ACQUIRE_ALLOCATOR2(descr1, descr2);

    while (N--) {
        const npy_packed_static_string *ps1 = (npy_packed_static_string *)in1;
        npy_static_string s1 = {0, NULL};
        int s1_isnull = npy_string_load(descr1->allocator, ps1, &s1);
        const npy_packed_static_string *ps2 = (npy_packed_static_string *)in2;
        npy_static_string s2 = {0, NULL};
        int s2_isnull = npy_string_load(descr2->allocator, ps2, &s2);
        if (NPY_UNLIKELY(s1_isnull < 0 || s2_isnull < 0)) {
            gil_error(PyExc_MemoryError, "Failed to load string in equal");
            goto fail;
        }
        else if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            if (has_nan_na) {
                // s1 or s2 is NA
                *out = (npy_bool)0;
                goto next_step;
            }
            else if (has_null && !has_string_na) {
                if (s1_isnull && s2_isnull) {
                    *out = (npy_bool)1;
                }
                else {
                    *out = (npy_bool)0;
                }
            }
            else {
                if (s1_isnull) {
                    s1 = *default_string;
                }
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
        }
        if (s1.size == s2.size &&
            (s1.size == 0 || strncmp(s1.buf, s2.buf, s1.size) == 0)) {
            *out = (npy_bool)1;
        }
        else {
            *out = (npy_bool)0;
        }

    next_step:
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return 0;

fail:
    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return -1;
}

static int
string_not_equal_strided_loop(PyArrayMethod_Context *context,
                              char *const data[], npy_intp const dimensions[],
                              npy_intp const strides[],
                              NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *descr1 = (StringDTypeObject *)context->descriptors[0];
    StringDTypeObject *descr2 = (StringDTypeObject *)context->descriptors[1];
    int has_null = descr1->na_object != NULL;
    int has_nan_na = descr1->has_nan_na;
    int has_string_na = descr1->has_string_na;
    const npy_static_string *default_string = &descr1->default_string;
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    npy_bool *out = (npy_bool *)data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    NPY_STRING_ACQUIRE_ALLOCATOR2(descr1, descr2);

    while (N--) {
        const npy_packed_static_string *ps1 = (npy_packed_static_string *)in1;
        npy_static_string s1 = {0, NULL};
        int s1_isnull = npy_string_load(descr1->allocator, ps1, &s1);
        const npy_packed_static_string *ps2 = (npy_packed_static_string *)in2;
        npy_static_string s2 = {0, NULL};
        int s2_isnull = npy_string_load(descr2->allocator, ps2, &s2);
        if (NPY_UNLIKELY(s1_isnull < 0 || s2_isnull < 0)) {
            gil_error(PyExc_MemoryError, "Failed to load string in not equal");
            goto fail;
        }
        else if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            if (has_nan_na) {
                // s1 or s2 is NA
                *out = (npy_bool)0;
                goto next_step;
            }
            else if (has_null && !has_string_na) {
                if (s1_isnull && s2_isnull) {
                    *out = (npy_bool)0;
                }
                else {
                    *out = (npy_bool)1;
                }
            }
            else {
                if (s1_isnull) {
                    s1 = *default_string;
                }
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
        }

        if (s1.size == s2.size && strncmp(s1.buf, s2.buf, s1.size) == 0) {
            *out = (npy_bool)0;
        }
        else {
            *out = (npy_bool)1;
        }

    next_step:
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return 0;

fail:
    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return -1;
}

static int
string_greater_strided_loop(PyArrayMethod_Context *context, char *const data[],
                            npy_intp const dimensions[],
                            npy_intp const strides[],
                            NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *descr1 = (StringDTypeObject *)context->descriptors[0];
    StringDTypeObject *descr2 = (StringDTypeObject *)context->descriptors[1];
    int has_null = descr1->na_object != NULL;
    int has_nan_na = descr1->has_nan_na;
    int has_string_na = descr1->has_string_na;
    const npy_static_string *default_string = &descr1->default_string;
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    npy_bool *out = (npy_bool *)data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    NPY_STRING_ACQUIRE_ALLOCATOR2(descr1, descr2);

    while (N--) {
        const npy_packed_static_string *ps1 = (npy_packed_static_string *)in1;
        npy_static_string s1 = {0, NULL};
        int s1_isnull = npy_string_load(descr1->allocator, ps1, &s1);
        const npy_packed_static_string *ps2 = (npy_packed_static_string *)in2;
        npy_static_string s2 = {0, NULL};
        int s2_isnull = npy_string_load(descr2->allocator, ps2, &s2);
        if (NPY_UNLIKELY(s1_isnull < 0 || s2_isnull < 0)) {
            gil_error(PyExc_MemoryError, "Failed to load string in greater");
            goto fail;
        }
        else if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            if (has_nan_na) {
                // s1 or s2 is NA
                *out = (npy_bool)0;
                goto next_step;
            }
            else if (has_null && !has_string_na) {
                gil_error(PyExc_TypeError,
                          "'>' not supported for null values that are not "
                          "nan-like.");
                goto fail;
            }
            else {
                if (s1_isnull) {
                    s1 = *default_string;
                }
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
        }
        if (npy_string_cmp(&s1, &s2) > 0) {
            *out = (npy_bool)1;
        }
        else {
            *out = (npy_bool)0;
        }

    next_step:
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return 0;

fail:
    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return -1;
}

static int
string_greater_equal_strided_loop(PyArrayMethod_Context *context,
                                  char *const data[],
                                  npy_intp const dimensions[],
                                  npy_intp const strides[],
                                  NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *descr1 = (StringDTypeObject *)context->descriptors[0];
    StringDTypeObject *descr2 = (StringDTypeObject *)context->descriptors[1];
    int has_null = descr1->na_object != NULL;
    int has_nan_na = descr1->has_nan_na;
    int has_string_na = descr1->has_string_na;
    const npy_static_string *default_string = &descr1->default_string;
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    npy_bool *out = (npy_bool *)data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    NPY_STRING_ACQUIRE_ALLOCATOR2(descr1, descr2);

    while (N--) {
        const npy_packed_static_string *ps1 = (npy_packed_static_string *)in1;
        npy_static_string s1 = {0, NULL};
        int s1_isnull = npy_string_load(descr1->allocator, ps1, &s1);
        const npy_packed_static_string *ps2 = (npy_packed_static_string *)in2;
        npy_static_string s2 = {0, NULL};
        int s2_isnull = npy_string_load(descr2->allocator, ps2, &s2);
        if (NPY_UNLIKELY(s1_isnull < 0 || s2_isnull < 0)) {
            gil_error(PyExc_MemoryError,
                      "Failed to load string in greater equal");
            goto fail;
        }
        else if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            if (has_nan_na) {
                // s1 or s2 is NA
                *out = (npy_bool)0;
                goto next_step;
            }
            else if (has_null && !has_string_na) {
                gil_error(PyExc_TypeError,
                          "'>=' not supported for null values that are not "
                          "nan-like.");
                goto fail;
            }
            else {
                if (s1_isnull) {
                    s1 = *default_string;
                }
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
        }
        if (npy_string_cmp(&s1, &s2) >= 0) {
            *out = (npy_bool)1;
        }
        else {
            *out = (npy_bool)0;
        }

    next_step:
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return 0;

fail:
    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return -1;
}

static int
string_less_strided_loop(PyArrayMethod_Context *context, char *const data[],
                         npy_intp const dimensions[], npy_intp const strides[],
                         NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *descr1 = (StringDTypeObject *)context->descriptors[0];
    StringDTypeObject *descr2 = (StringDTypeObject *)context->descriptors[1];
    int has_null = descr1->na_object != NULL;
    int has_nan_na = descr1->has_nan_na;
    int has_string_na = descr1->has_string_na;
    const npy_static_string *default_string = &descr1->default_string;
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    npy_bool *out = (npy_bool *)data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    NPY_STRING_ACQUIRE_ALLOCATOR2(descr1, descr2);

    while (N--) {
        const npy_packed_static_string *ps1 = (npy_packed_static_string *)in1;
        npy_static_string s1 = {0, NULL};
        int s1_isnull = npy_string_load(descr1->allocator, ps1, &s1);
        const npy_packed_static_string *ps2 = (npy_packed_static_string *)in2;
        npy_static_string s2 = {0, NULL};
        int s2_isnull = npy_string_load(descr2->allocator, ps2, &s2);
        if (NPY_UNLIKELY(s1_isnull < 0 || s2_isnull < 0)) {
            gil_error(PyExc_MemoryError, "Failed to load string in less");
            goto fail;
        }
        else if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            if (has_nan_na) {
                // s1 or s2 is NA
                *out = (npy_bool)0;
                goto next_step;
            }
            else if (has_null && !has_string_na) {
                gil_error(PyExc_TypeError,
                          "'<' not supported for null values that are not "
                          "nan-like.");
                goto fail;
            }
            else {
                if (s1_isnull) {
                    s1 = *default_string;
                }
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
        }
        if (npy_string_cmp(&s1, &s2) < 0) {
            *out = (npy_bool)1;
        }
        else {
            *out = (npy_bool)0;
        }

    next_step:
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return 0;

fail:
    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return -1;
}

static int
string_less_equal_strided_loop(PyArrayMethod_Context *context,
                               char *const data[], npy_intp const dimensions[],
                               npy_intp const strides[],
                               NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *descr1 = (StringDTypeObject *)context->descriptors[0];
    StringDTypeObject *descr2 = (StringDTypeObject *)context->descriptors[1];
    int has_null = descr1->na_object != NULL;
    int has_nan_na = descr1->has_nan_na;
    int has_string_na = descr1->has_string_na;
    const npy_static_string *default_string = &descr1->default_string;
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    npy_bool *out = (npy_bool *)data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    NPY_STRING_ACQUIRE_ALLOCATOR2(descr1, descr2);

    while (N--) {
        const npy_packed_static_string *ps1 = (npy_packed_static_string *)in1;
        npy_static_string s1 = {0, NULL};
        int s1_isnull = npy_string_load(descr1->allocator, ps1, &s1);
        const npy_packed_static_string *ps2 = (npy_packed_static_string *)in2;
        npy_static_string s2 = {0, NULL};
        int s2_isnull = npy_string_load(descr2->allocator, ps2, &s2);
        if (NPY_UNLIKELY(s1_isnull < 0 || s2_isnull < 0)) {
            gil_error(PyExc_MemoryError,
                      "Failed to load string in less equal");
            goto fail;
        }
        else if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            if (has_nan_na) {
                // s1 or s2 is NA
                *out = (npy_bool)0;
                goto next_step;
            }
            else if (has_null && !has_string_na) {
                gil_error(PyExc_TypeError,
                          "'<=' not supported for null values that are not "
                          "nan-like.");
                goto fail;
            }
            else {
                if (s1_isnull) {
                    s1 = *default_string;
                }
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
        }
        if (npy_string_cmp(&s1, &s2) <= 0) {
            *out = (npy_bool)1;
        }
        else {
            *out = (npy_bool)0;
        }

    next_step:
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return 0;

fail:
    NPY_STRING_RELEASE_ALLOCATOR(descr1);
    NPY_STRING_RELEASE_ALLOCATOR(descr2);

    return -1;
}

static NPY_CASTING
string_comparison_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[]), PyArray_Descr *given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);  // cannot fail

    return NPY_NO_CASTING;
}

static int
string_isnan_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[],
                          npy_intp const strides[],
                          NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *descr = (StringDTypeObject *)context->descriptors[0];
    int has_nan_na = descr->has_nan_na;

    npy_intp N = dimensions[0];
    char *in = data[0];
    npy_bool *out = (npy_bool *)data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        const npy_packed_static_string *s = (npy_packed_static_string *)in;
        if (has_nan_na && npy_string_isnull(s)) {
            *out = (npy_bool)1;
        }
        else {
            *out = (npy_bool)0;
        }

        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static NPY_CASTING
string_isnan_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[]), PyArray_Descr *given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    loop_descrs[1] = PyArray_DescrFromType(NPY_BOOL);  // cannot fail

    return NPY_NO_CASTING;
}

/*
 * Copied from NumPy, because NumPy doesn't always use it :)
 */
static int
ufunc_promoter_internal(PyUFuncObject *ufunc, PyArray_DTypeMeta *op_dtypes[],
                        PyArray_DTypeMeta *signature[],
                        PyArray_DTypeMeta *new_op_dtypes[],
                        PyArray_DTypeMeta *final_dtype)
{
    /* If nin < 2 promotion is a no-op, so it should not be registered */
    assert(ufunc->nin > 1);
    if (op_dtypes[0] == NULL) {
        assert(ufunc->nin == 2 && ufunc->nout == 1); /* must be reduction */
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[0] = op_dtypes[1];
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[1] = op_dtypes[1];
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[2] = op_dtypes[1];
        return 0;
    }
    PyArray_DTypeMeta *common = NULL;
    /*
     * If a signature is used and homogeneous in its outputs use that
     * (Could/should likely be rather applied to inputs also, although outs
     * only could have some advantage and input dtypes are rarely enforced.)
     */
    for (int i = ufunc->nin; i < ufunc->nargs; i++) {
        if (signature[i] != NULL) {
            if (common == NULL) {
                Py_INCREF(signature[i]);
                common = signature[i];
            }
            else if (common != signature[i]) {
                Py_CLEAR(common); /* Not homogeneous, unset common */
                break;
            }
        }
    }
    Py_XDECREF(common);

    /* Otherwise, set all input operands to final_dtype */
    for (int i = 0; i < ufunc->nargs; i++) {
        PyArray_DTypeMeta *tmp = final_dtype;
        if (signature[i]) {
            tmp = signature[i]; /* never replace a fixed one. */
        }
        Py_INCREF(tmp);
        new_op_dtypes[i] = tmp;
    }
    for (int i = ufunc->nin; i < ufunc->nargs; i++) {
        Py_XINCREF(op_dtypes[i]);
        new_op_dtypes[i] = op_dtypes[i];
    }

    return 0;
}

static int
string_object_promoter(PyObject *ufunc, PyArray_DTypeMeta *op_dtypes[],
                       PyArray_DTypeMeta *signature[],
                       PyArray_DTypeMeta *new_op_dtypes[])
{
    return ufunc_promoter_internal((PyUFuncObject *)ufunc, op_dtypes,
                                   signature, new_op_dtypes,
                                   (PyArray_DTypeMeta *)&PyArray_ObjectDType);
}

static int
string_unicode_promoter(PyObject *ufunc, PyArray_DTypeMeta *op_dtypes[],
                        PyArray_DTypeMeta *signature[],
                        PyArray_DTypeMeta *new_op_dtypes[])
{
    return ufunc_promoter_internal((PyUFuncObject *)ufunc, op_dtypes,
                                   signature, new_op_dtypes,
                                   (PyArray_DTypeMeta *)&StringDType);
}

// Register a ufunc.
//
// Pass NULL for resolve_func to use the default_resolve_descriptors.
int
init_ufunc(PyObject *numpy, const char *ufunc_name, PyArray_DTypeMeta **dtypes,
           resolve_descriptors_function *resolve_func,
           PyArrayMethod_StridedLoop *loop_func, const char *loop_name,
           int nin, int nout, NPY_CASTING casting, NPY_ARRAYMETHOD_FLAGS flags)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArrayMethod_Spec spec = {
            .name = loop_name,
            .nin = nin,
            .nout = nout,
            .casting = casting,
            .flags = flags,
            .dtypes = dtypes,
            .slots = NULL,
    };

    PyType_Slot resolve_slots[] = {
            {NPY_METH_resolve_descriptors, resolve_func},
            {NPY_METH_strided_loop, loop_func},
            {0, NULL}};

    PyType_Slot strided_slots[] = {{NPY_METH_strided_loop, loop_func},
                                   {0, NULL}};

    if (resolve_func == NULL) {
        spec.slots = strided_slots;
    }
    else {
        spec.slots = resolve_slots;
    }

    if (PyUFunc_AddLoopFromSpec(ufunc, &spec) < 0) {
        Py_DECREF(ufunc);
        return -1;
    }

    Py_DECREF(ufunc);
    return 0;
}

int
add_promoter(PyObject *numpy, const char *ufunc_name,
             PyArray_DTypeMeta *ldtype, PyArray_DTypeMeta *rdtype,
             PyArray_DTypeMeta *edtype, promoter_function *promoter_impl)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);

    if (ufunc == NULL) {
        return -1;
    }

    PyObject *DType_tuple = PyTuple_Pack(3, ldtype, rdtype, edtype);

    if (DType_tuple == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }

    PyObject *promoter_capsule = PyCapsule_New((void *)promoter_impl,
                                               "numpy._ufunc_promoter", NULL);

    if (promoter_capsule == NULL) {
        Py_DECREF(ufunc);
        Py_DECREF(DType_tuple);
        return -1;
    }

    if (PyUFunc_AddPromoter(ufunc, DType_tuple, promoter_capsule) < 0) {
        Py_DECREF(promoter_capsule);
        Py_DECREF(DType_tuple);
        Py_DECREF(ufunc);
        return -1;
    }

    Py_DECREF(promoter_capsule);
    Py_DECREF(DType_tuple);
    Py_DECREF(ufunc);

    return 0;
}

#define INIT_MULTIPLY(typename, shortname)                                 \
    PyArray_DTypeMeta *multiply_right_##shortname##_types[] = {            \
            (PyArray_DTypeMeta *)&StringDType, &PyArray_##typename##DType, \
            (PyArray_DTypeMeta *)&StringDType};                            \
                                                                           \
    if (init_ufunc(numpy, "multiply", multiply_right_##shortname##_types,  \
                   &multiply_resolve_descriptors,                          \
                   &multiply_right_##shortname##_strided_loop,             \
                   "string_multiply", 2, 1, NPY_NO_CASTING, 0) < 0) {      \
        goto error;                                                        \
    }                                                                      \
                                                                           \
    PyArray_DTypeMeta *multiply_left_##shortname##_types[] = {             \
            &PyArray_##typename##DType, (PyArray_DTypeMeta *)&StringDType, \
            (PyArray_DTypeMeta *)&StringDType};                            \
                                                                           \
    if (init_ufunc(numpy, "multiply", multiply_left_##shortname##_types,   \
                   &multiply_resolve_descriptors,                          \
                   &multiply_left_##shortname##_strided_loop,              \
                   "string_multiply", 2, 1, NPY_NO_CASTING, 0) < 0) {      \
        goto error;                                                        \
    }

int
init_ufuncs(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        return -1;
    }

    static char *comparison_ufunc_names[6] = {"equal",   "not_equal",
                                              "greater", "greater_equal",
                                              "less",    "less_equal"};

    PyArray_DTypeMeta *comparison_dtypes[] = {
            (PyArray_DTypeMeta *)&StringDType,
            (PyArray_DTypeMeta *)&StringDType, &PyArray_BoolDType};

    if (init_ufunc(numpy, "equal", comparison_dtypes,
                   &string_comparison_resolve_descriptors,
                   &string_equal_strided_loop, "string_equal", 2, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    if (init_ufunc(numpy, "not_equal", comparison_dtypes,
                   &string_comparison_resolve_descriptors,
                   &string_not_equal_strided_loop, "string_not_equal", 2, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    if (init_ufunc(numpy, "greater", comparison_dtypes,
                   &string_comparison_resolve_descriptors,
                   &string_greater_strided_loop, "string_greater", 2, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    if (init_ufunc(numpy, "greater_equal", comparison_dtypes,
                   &string_comparison_resolve_descriptors,
                   &string_greater_equal_strided_loop, "string_greater_equal",
                   2, 1, NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    if (init_ufunc(numpy, "less", comparison_dtypes,
                   &string_comparison_resolve_descriptors,
                   &string_less_strided_loop, "string_less", 2, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    if (init_ufunc(numpy, "less_equal", comparison_dtypes,
                   &string_comparison_resolve_descriptors,
                   &string_less_equal_strided_loop, "string_less_equal", 2, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    for (int i = 0; i < 6; i++) {
        if (add_promoter(numpy, comparison_ufunc_names[i],
                         (PyArray_DTypeMeta *)&StringDType,
                         &PyArray_UnicodeDType, &PyArray_BoolDType,
                         string_unicode_promoter) < 0) {
            goto error;
        }

        if (add_promoter(numpy, comparison_ufunc_names[i],
                         &PyArray_UnicodeDType,
                         (PyArray_DTypeMeta *)&StringDType, &PyArray_BoolDType,
                         string_unicode_promoter) < 0) {
            goto error;
        }

        if (add_promoter(numpy, comparison_ufunc_names[i],
                         &PyArray_ObjectDType,
                         (PyArray_DTypeMeta *)&StringDType, &PyArray_BoolDType,
                         &string_object_promoter) < 0) {
            goto error;
        }

        if (add_promoter(numpy, comparison_ufunc_names[i],
                         (PyArray_DTypeMeta *)&StringDType,
                         &PyArray_ObjectDType, &PyArray_BoolDType,
                         &string_object_promoter) < 0) {
            goto error;
        }
    }

    PyArray_DTypeMeta *isnan_dtypes[] = {(PyArray_DTypeMeta *)&StringDType,
                                         &PyArray_BoolDType};

    if (init_ufunc(numpy, "isnan", isnan_dtypes,
                   &string_isnan_resolve_descriptors,
                   &string_isnan_strided_loop, "string_isnan", 1, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    PyArray_DTypeMeta *binary_dtypes[] = {
            (PyArray_DTypeMeta *)&StringDType,
            (PyArray_DTypeMeta *)&StringDType,
            (PyArray_DTypeMeta *)&StringDType,
    };

    if (init_ufunc(numpy, "maximum", binary_dtypes, binary_resolve_descriptors,
                   &maximum_strided_loop, "string_maximum", 2, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    if (init_ufunc(numpy, "minimum", binary_dtypes, binary_resolve_descriptors,
                   &minimum_strided_loop, "string_minimum", 2, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    if (init_ufunc(numpy, "add", binary_dtypes, binary_resolve_descriptors,
                   &add_strided_loop, "string_add", 2, 1, NPY_NO_CASTING,
                   0) < 0) {
        goto error;
    }

    INIT_MULTIPLY(Int8, int8);
    INIT_MULTIPLY(Int16, int16);
    INIT_MULTIPLY(Int32, int32);
    INIT_MULTIPLY(Int64, int64);
    INIT_MULTIPLY(UInt8, uint8);
    INIT_MULTIPLY(UInt16, uint16);
    INIT_MULTIPLY(UInt32, uint32);
    INIT_MULTIPLY(UInt64, uint64);
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    INIT_MULTIPLY(Byte, byte);
    INIT_MULTIPLY(UByte, ubyte);
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    INIT_MULTIPLY(Short, short);
    INIT_MULTIPLY(UShort, ushort);
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    INIT_MULTIPLY(Long, long);
    INIT_MULTIPLY(ULong, ulong);
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    INIT_MULTIPLY(LongLong, longlong);
    INIT_MULTIPLY(ULongLong, ulonglong);
#endif

    Py_DECREF(numpy);
    return 0;

error:
    Py_DECREF(numpy);
    return -1;
}
