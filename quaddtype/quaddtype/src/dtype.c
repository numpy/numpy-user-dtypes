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

static inline int quad_load(Sleef_quad *x, char *data_ptr) 
{
    if (data_ptr == NULL || x == NULL) 
    {
        return -1;
    }
    *x = *(Sleef_quad *)data_ptr;
    return 0;
}

static inline int quad_store(char *data_ptr, Sleef_quad x) 
{
    if (data_ptr == NULL) 
    {
        return -1;
    }
    *(Sleef_quad *)data_ptr = x;
    return 0;
}

QuadPrecDTypeObject  * new_quaddtype_instance(void)
{
    QuadPrecDTypeObject *new = (QuadPrecDTypeObject *)PyArrayDescr_Type.tp_new((PyTypeObject *)&QuadPrecDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    new->base.elsize = sizeof(Sleef_quad);
    new->base.alignment = _Alignof(Sleef_quad);

    return new;
}

static QuadPrecDTypeObject * ensure_canonical(QuadPrecDTypeObject *self)
{
    Py_INCREF(self);
    return self;
}

static QuadPrecDTypeObject * common_instance(QuadPrecDTypeObject *dtype1, QuadPrecDTypeObject *dtype2)
{
    Py_INCREF(dtype1);
    return dtype1;
}


static PyArray_DTypeMeta * common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // Promote integer and floating-point types to QuadPrecDType
    if (other->type_num >= 0 && 
        (PyTypeNum_ISINTEGER(other->type_num) || 
         PyTypeNum_ISFLOAT(other->type_num))) {
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
    if (Py_TYPE(obj) != &QuadPrecision_Type) 
    {
        PyErr_SetString(PyExc_TypeError, "Can only store QuadPrecision in a QuadPrecDType array.");
        return NULL;
    }
    return (PyArray_Descr *)new_quaddtype_instance();
}

static int quadprec_setitem(QuadPrecDTypeObject *descr, PyObject *obj, char *dataptr)
{
    QuadPrecisionObject *value;
    if (PyObject_TypeCheck(obj, &QuadPrecision_Type)) 
    {
        Py_INCREF(obj);
        value = (QuadPrecisionObject *)obj;
    }
    else 
    {
        value = QuadPrecision_from_object(obj);
        if (value == NULL) {
            return -1;
        }
    }

    if (quad_store(dataptr, value->quad.value) < 0)
    {
        Py_DECREF(value);
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Invalid memory location %p", (void*)dataptr);
        PyErr_SetString(PyExc_ValueError, error_msg);
        return -1;
    }

    Py_DECREF(value);
    return 0;
}

static PyObject * quadprec_getitem(QuadPrecDTypeObject *descr, char *dataptr)
{
    QuadPrecisionObject *new = QuadPrecision_raw_new();
    if (!new) 
    {
        return NULL;
    }
    if (quad_load(&new->quad.value, dataptr) < 0) 
    {
        Py_DECREF(new);
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Invalid memory location %p", (void*)dataptr);
        PyErr_SetString(PyExc_ValueError, error_msg);
        return NULL;
    }
    return (PyObject *)new;
}

static PyArray_Descr *quadprec_default_descr(PyArray_DTypeMeta *NPY_UNUSED(cls))
{
    return (PyArray_Descr *)new_quaddtype_instance();
}

static PyType_Slot QuadPrecDType_Slots[] = 
{
    {NPY_DT_ensure_canonical, &ensure_canonical},
    {NPY_DT_common_instance, &common_instance},
    {NPY_DT_common_dtype, &common_dtype},
    {NPY_DT_discover_descr_from_pyobject, &quadprec_discover_descriptor_from_pyobject},
    {NPY_DT_setitem, &quadprec_setitem},
    {NPY_DT_getitem, &quadprec_getitem},
    {NPY_DT_default_descr, &quadprec_default_descr},
    {0, NULL}
};


static PyObject * QuadPrecDType_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    if (PyTuple_GET_SIZE(args) != 0 || (kwds != NULL && PyDict_Size(kwds) != 0)) {
        PyErr_SetString(PyExc_TypeError,
                        "QuadPrecDType takes no arguments");
        return NULL;
    }

    return (PyObject *)new_quaddtype_instance();
}

static PyObject * QuadPrecDType_repr(QuadPrecDTypeObject *self)
{
    return PyUnicode_FromString("QuadPrecDType()");
}

PyArray_DTypeMeta QuadPrecDType = {
    {{
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "QuadPrecDType.QuadPrecDType",
        .tp_basicsize = sizeof(QuadPrecDTypeObject),
        .tp_new = QuadPrecDType_new,
        .tp_repr = (reprfunc)QuadPrecDType_repr,
        .tp_str = (reprfunc)QuadPrecDType_repr,
    }},
};

int init_quadprec_dtype(void)
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

    if (PyType_Ready((PyTypeObject *)&QuadPrecDType) < 0) 
    {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(&QuadPrecDType, &QuadPrecDType_DTypeSpec) < 0)
    {
        return -1;
    }

    free_casts();
    
    return 0;
}