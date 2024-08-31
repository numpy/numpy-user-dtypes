#include <Python.h>
#include <sleef.h>
#include <sleefquad.h>

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
    QuadPrecDTypeObject *new = (QuadPrecDTypeObject *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)&QuadPrecDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    new->base.elsize = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);
    new->base.alignment = (backend == BACKEND_SLEEF) ? _Alignof(Sleef_quad) : _Alignof(long double);
    new->backend = backend;

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
    if (dtype1->backend != dtype2->backend) {
        PyErr_SetString(PyExc_TypeError,
                        "Cannot combine QuadPrecDType instances with different backends");
        return NULL;
    }
    Py_INCREF(dtype1);
    return dtype1;
}

static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
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
    if (Py_TYPE(obj) != &QuadPrecision_Type) {
        PyErr_SetString(PyExc_TypeError, "Can only store QuadPrecision in a QuadPrecDType array.");
        return NULL;
    }
    QuadPrecisionObject *quad_obj = (QuadPrecisionObject *)obj;
    return (PyArray_Descr *)new_quaddtype_instance(quad_obj->backend);
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
quadprec_default_descr(PyArray_DTypeMeta *NPY_UNUSED(cls))
{
    return (PyArray_Descr *)new_quaddtype_instance(BACKEND_SLEEF);
}

static PyType_Slot QuadPrecDType_Slots[] = {
        {NPY_DT_ensure_canonical, &ensure_canonical},
        {NPY_DT_common_instance, &common_instance},
        {NPY_DT_common_dtype, &common_dtype},
        {NPY_DT_discover_descr_from_pyobject, &quadprec_discover_descriptor_from_pyobject},
        {NPY_DT_setitem, &quadprec_setitem},
        {NPY_DT_getitem, &quadprec_getitem},
        {NPY_DT_default_descr, &quadprec_default_descr},
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

    return (PyObject *)new_quaddtype_instance(backend);
}

static PyObject *
QuadPrecDType_repr(QuadPrecDTypeObject *self)
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
                .tp_str = (reprfunc)QuadPrecDType_repr,
        }},
};

int
init_quadprec_dtype(void)
{
    PyArrayMethod_Spec **casts = init_casts();
    if (!casts)
        return -1;

    PyArrayDTypeMeta_Spec QuadPrecDType_DTypeSpec = {
            .flags = NPY_DT_NUMERIC,
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