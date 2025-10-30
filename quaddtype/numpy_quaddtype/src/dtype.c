#include <Python.h>
#include <sleef.h>
#include <sleefquad.h>
#include <ctype.h>

#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"

#include "scalar.h"
#include "casts.h"
#include "dtype.h"
#include "dragon4.h"
#include "constants.hpp"

static inline int
quad_load(void *x, char *data_ptr, QuadBackendType backend)
{
    if (data_ptr == NULL || x == NULL) {
        return -1;
    }
    if (backend == BACKEND_SLEEF) {
        *(Sleef_quad *)x = *(Sleef_quad *)data_ptr;
    }
    else {
        *(long double *)x = *(long double *)data_ptr;
    }
    return 0;
}

static inline int
quad_store(char *data_ptr, void *x, QuadBackendType backend)
{
    if (data_ptr == NULL || x == NULL) {
        return -1;
    }
    if (backend == BACKEND_SLEEF) {
        *(Sleef_quad *)data_ptr = *(Sleef_quad *)x;
    }
    else {
        *(long double *)data_ptr = *(long double *)x;
    }
    return 0;
}

QuadPrecDTypeObject *
new_quaddtype_instance(QuadBackendType backend)
{
    QuadBackendType target_backend = backend;
    if (backend != BACKEND_SLEEF && backend != BACKEND_LONGDOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Backend must be sleef or longdouble");
        return NULL;
        // target_backend = BACKEND_SLEEF;
    }

    QuadPrecDTypeObject *new = (QuadPrecDTypeObject *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)&QuadPrecDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    new->base.elsize = (target_backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);
    new->base.alignment =
            (target_backend == BACKEND_SLEEF) ? _Alignof(Sleef_quad) : _Alignof(long double);
    new->backend = target_backend;
    return new;
}

static QuadPrecDTypeObject *
ensure_canonical(QuadPrecDTypeObject *self)
{
    Py_INCREF(self);
    return self;
}

static QuadPrecDTypeObject *
common_instance(QuadPrecDTypeObject *dtype1, QuadPrecDTypeObject *dtype2)
{
    // if backend mismatch then return SLEEF one (safe to cast ld to quad)
    if (dtype1->backend != dtype2->backend) {
        if (dtype1->backend == BACKEND_SLEEF) {
            Py_INCREF(dtype1);
            return dtype1;
        }

        Py_INCREF(dtype2);
        return dtype2;
    }
    Py_INCREF(dtype1);
    return dtype1;
}

static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // Handle Python abstract dtypes (PyLongDType, PyFloatDType)
    // These have type_num = -1 
    if (other == &PyArray_PyLongDType || other == &PyArray_PyFloatDType) {
        Py_INCREF(cls);
        return cls;
    }
    
    // Promote integer and floating-point types to QuadPrecDType
    if (other->type_num >= 0 &&
        (PyTypeNum_ISINTEGER(other->type_num) || PyTypeNum_ISFLOAT(other->type_num))) {
        Py_INCREF(cls);
        return cls;
    }
    // Don't promote complex types
    if (PyTypeNum_ISCOMPLEX(other->type_num)) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }

    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}

static PyArray_Descr *
quadprec_discover_descriptor_from_pyobject(PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    if (Py_TYPE(obj) == &QuadPrecision_Type) {
        /* QuadPrecision scalar: use its backend */
        QuadPrecisionObject *quad_obj = (QuadPrecisionObject *)obj;
        return (PyArray_Descr *)new_quaddtype_instance(quad_obj->backend);
    }
    
    /* For Python int/float/other numeric types: return default descriptor */
    /* The casting machinery will handle conversion to QuadPrecision */
    if (PyLong_Check(obj) || PyFloat_Check(obj)) {
        return (PyArray_Descr *)new_quaddtype_instance(BACKEND_SLEEF);
    }
    
    /* Unknown type - ERROR */
    PyErr_SetString(PyExc_TypeError, "Can only store QuadPrecision, int, or float in a QuadPrecDType array.");
    return NULL;
}

static int
quadprec_setitem(QuadPrecDTypeObject *descr, PyObject *obj, char *dataptr)
{
    QuadPrecisionObject *value;
    if (PyObject_TypeCheck(obj, &QuadPrecision_Type)) {
        Py_INCREF(obj);
        value = (QuadPrecisionObject *)obj;
    }
    else {
        value = QuadPrecision_from_object(obj, descr->backend);
        if (value == NULL) {
            return -1;
        }
    }

    if (quad_store(dataptr, &value->value, descr->backend) < 0) {
        Py_DECREF(value);
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Invalid memory location %p", (void *)dataptr);
        PyErr_SetString(PyExc_ValueError, error_msg);
        return -1;
    }

    Py_DECREF(value);
    return 0;
}

static PyObject *
quadprec_getitem(QuadPrecDTypeObject *descr, char *dataptr)
{
    QuadPrecisionObject *new = QuadPrecision_raw_new(descr->backend);
    if (!new) {
        return NULL;
    }
    if (quad_load(&new->value, dataptr, descr->backend) < 0) {
        Py_DECREF(new);
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Invalid memory location %p", (void *)dataptr);
        PyErr_SetString(PyExc_ValueError, error_msg);
        return NULL;
    }
    return (PyObject *)new;
}

static PyArray_Descr *
quadprec_default_descr(PyArray_DTypeMeta *cls)
{
    QuadPrecDTypeObject *temp = new_quaddtype_instance(BACKEND_SLEEF);
    return (PyArray_Descr *)temp;
}

static int
quadprec_get_constant(PyArray_Descr *descr, int constant_id, void *ptr)
{
    QuadPrecDTypeObject *quad_descr = (QuadPrecDTypeObject *)descr;
    
    if (quad_descr->backend != BACKEND_SLEEF) {
        /* Long double backend not yet implemented */
        return 0;
    }
    
    Sleef_quad val;
    
    switch (constant_id) {
        case NPY_CONSTANT_zero:
            val = QUAD_PRECISION_ZERO;
            break;
            
        case NPY_CONSTANT_one:
            val = QUAD_PRECISION_ONE;
            break;
            
        case NPY_CONSTANT_minimum_finite:
            val = QUAD_PRECISION_MIN_FINITE;
            break;
            
        case NPY_CONSTANT_maximum_finite:
            val = QUAD_PRECISION_MAX_FINITE;
            break;
            
        case NPY_CONSTANT_inf:
            val = QUAD_PRECISION_INF;
            break;
            
        case NPY_CONSTANT_ninf:
            val = QUAD_PRECISION_NINF;
            break;
            
        case NPY_CONSTANT_nan:
            val = QUAD_PRECISION_NAN;
            break;
            
        case NPY_CONSTANT_finfo_radix:
            val = QUAD_PRECISION_RADIX;
            break;
            
        case NPY_CONSTANT_finfo_eps:
            val = SLEEF_QUAD_EPSILON;
            break;
            
        case NPY_CONSTANT_finfo_smallest_normal:
            val = SLEEF_QUAD_MIN;
            break;
            
        case NPY_CONSTANT_finfo_smallest_subnormal:
            val = SMALLEST_SUBNORMAL_VALUE;
            break;
            
        /* Integer constants - these return npy_intp values */
        case NPY_CONSTANT_finfo_nmant:
            *(npy_intp *)ptr = QUAD_NMANT;
            return 1;
            
        case NPY_CONSTANT_finfo_min_exp:
            *(npy_intp *)ptr = QUAD_MIN_EXP;
            return 1;
            
        case NPY_CONSTANT_finfo_max_exp:
            *(npy_intp *)ptr = QUAD_MAX_EXP;
            return 1;
            
        case NPY_CONSTANT_finfo_decimal_digits:
            *(npy_intp *)ptr = QUAD_DECIMAL_DIGITS;
            return 1;
            
        default:
            /* Constant not supported */
            return 0;
    }
    
    /* Store the Sleef_quad value to the provided pointer */
    memcpy(ptr, &val, sizeof(Sleef_quad));
    return 1;
}

/*
 * Fill function.
 * The buffer already has the first two elements set:
 *   buffer[0] = start
 *   buffer[1] = start + step
 * We need to fill buffer[2..length-1] with the arithmetic progression.
 */
static int
quadprec_fill(void *buffer, npy_intp length, void *arr_)
{
    PyArrayObject *arr = (PyArrayObject *)arr_;
    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)PyArray_DESCR(arr);
    QuadBackendType backend = descr->backend;
    npy_intp i;
    
    if (length < 2) {
        return 0;  // Nothing to fill
    }
    
    if (backend == BACKEND_SLEEF) {
        Sleef_quad *buf = (Sleef_quad *)buffer;
        Sleef_quad start = buf[0];
        Sleef_quad delta = Sleef_subq1_u05(buf[1], start);  // delta = buf[1] - start
        
        for (i = 2; i < length; ++i) {
            // buf[i] = start + i * delta
            Sleef_quad i_quad = Sleef_cast_from_doubleq1(i);
            Sleef_quad i_delta = Sleef_mulq1_u05(i_quad, delta);
            buf[i] = Sleef_addq1_u05(start, i_delta);
        }
    }
    else { 
        long double *buf = (long double *)buffer;
        long double start = buf[0];
        long double delta = buf[1] - start;
        
        for (i = 2; i < length; ++i) {
            buf[i] = start + i * delta;
        }
    }
    
    return 0;
}

static int
quadprec_scanfunc(FILE *fp, void *dptr, char *ignore, PyArray_Descr *descr_generic)
{
    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)descr_generic;
    char buffer[512];
    int ch;
    size_t i = 0;
    
    /* Skip whitespace */
    while ((ch = fgetc(fp)) != EOF && isspace(ch)) {
        /* continue */
    }
    
    if (ch == EOF) {
        return EOF;  /* Return EOF when end of file is reached */
    }
    
    /* Read characters until we hit whitespace or EOF */
    buffer[i++] = (char)ch;
    while (i < sizeof(buffer) - 1) {
        ch = fgetc(fp);
        if (ch == EOF || isspace(ch)) {
            if (ch != EOF) {
                ungetc(ch, fp);  /* Put back the whitespace for separator handling */
            }
            break;
        }
        buffer[i++] = (char)ch;
    }
    buffer[i] = '\0';
    
    /* Convert string to quad precision */
    char *endptr;
    if (descr->backend == BACKEND_SLEEF) {
        Sleef_quad val = Sleef_strtoq(buffer, &endptr);
        if (endptr == buffer) {
            return 0;  /* Return 0 on parse error (no items read) */
        }
        *(Sleef_quad *)dptr = val;
    }
    else {
        long double val = strtold(buffer, &endptr);
        if (endptr == buffer) {
            return 0;  /* Return 0 on parse error (no items read) */
        }
        *(long double *)dptr = val;
    }
    
    return 1;  /* Return 1 on success (1 item read) */
}

static int
quadprec_fromstr(char *s, void *dptr, char **endptr, PyArray_Descr *descr_generic)
{
    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)descr_generic;
    
    if (descr->backend == BACKEND_SLEEF) {
        Sleef_quad val = Sleef_strtoq(s, endptr);
        if (*endptr == s) {
            return -1;
        }
        *(Sleef_quad *)dptr = val;
    }
    else {
        long double val = strtold(s, endptr);
        if (*endptr == s) {
            return -1;
        }
        *(long double *)dptr = val;
    }
    
    return 0;
}

static PyType_Slot QuadPrecDType_Slots[] = {
        {NPY_DT_ensure_canonical, &ensure_canonical},
        {NPY_DT_common_instance, &common_instance},
        {NPY_DT_common_dtype, &common_dtype},
        {NPY_DT_discover_descr_from_pyobject, &quadprec_discover_descriptor_from_pyobject},
        {NPY_DT_setitem, &quadprec_setitem},
        {NPY_DT_getitem, &quadprec_getitem},
        {NPY_DT_default_descr, &quadprec_default_descr},
        {NPY_DT_get_constant, &quadprec_get_constant},
        {NPY_DT_PyArray_ArrFuncs_fill, &quadprec_fill},
        {NPY_DT_PyArray_ArrFuncs_scanfunc, &quadprec_scanfunc},
        {NPY_DT_PyArray_ArrFuncs_fromstr, &quadprec_fromstr},
        {0, NULL}};

static PyObject *
QuadPrecDType_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"backend", NULL};
    const char *backend_str = "sleef";

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &backend_str)) {
        return NULL;
    }

    QuadBackendType backend = BACKEND_SLEEF;
    if (strcmp(backend_str, "longdouble") == 0) {
        backend = BACKEND_LONGDOUBLE;
    }
    else if (strcmp(backend_str, "sleef") != 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid backend. Use 'sleef' or 'longdouble'.");
        return NULL;
    }

    return (PyObject *)quadprec_discover_descriptor_from_pyobject(
            &QuadPrecDType, (PyObject *)QuadPrecision_raw_new(backend));
}

static PyObject *
QuadPrecDType_repr(QuadPrecDTypeObject *self)
{
    const char *backend_str = (self->backend == BACKEND_SLEEF) ? "sleef" : "longdouble";
    return PyUnicode_FromFormat("QuadPrecDType(backend='%s')", backend_str);
}

static PyObject *
QuadPrecDType_str(QuadPrecDTypeObject *self)
{
    const char *backend_str = (self->backend == BACKEND_SLEEF) ? "sleef" : "longdouble";
    return PyUnicode_FromFormat("QuadPrecDType(backend='%s')", backend_str);
}

PyArray_DTypeMeta QuadPrecDType = {
        {{
                PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numpy_quaddtype.QuadPrecDType",
                .tp_basicsize = sizeof(QuadPrecDTypeObject),
                .tp_new = QuadPrecDType_new,
                .tp_repr = (reprfunc)QuadPrecDType_repr,
                .tp_str = (reprfunc)QuadPrecDType_str,
        }},
};

int
init_quadprec_dtype(void)
{
    PyArrayMethod_Spec **casts = init_casts();
    if (!casts)
        return -1;

    PyArrayDTypeMeta_Spec QuadPrecDType_DTypeSpec = {
            .flags = NPY_DT_PARAMETRIC | NPY_DT_NUMERIC,
            .casts = casts,
            .typeobj = &QuadPrecision_Type,
            .slots = QuadPrecDType_Slots,
    };

    ((PyObject *)&QuadPrecDType)->ob_type = &PyArrayDTypeMeta_Type;

    ((PyTypeObject *)&QuadPrecDType)->tp_base = &PyArrayDescr_Type;

    if (PyType_Ready((PyTypeObject *)&QuadPrecDType) < 0) {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(&QuadPrecDType, &QuadPrecDType_DTypeSpec) < 0) {
        return -1;
    }

    free_casts();

    return 0;
}