#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL quaddtype_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL quaddtype_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"

#include "dtype.h"


PyTypeObject *QuadScalar_Type = NULL;

QuadDTypeObject *new_quaddtype_instance(void)
{
    QuadDTypeObject *new =
            (QuadDTypeObject *)PyArrayDescr_Type.tp_new((PyTypeObject *)&QuadDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }

    new->base.elsize = sizeof(__float128);
    new->base.alignment = _Alignof(__float128);
    return new;
}

static PyObject *quaddtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwargs)
{
    return (PyObject *)new_quaddtype_instance();
}

static void quaddtype_dealloc(QuadDTypeObject *self)
{
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *quaddtype_repr(QuadDTypeObject *self)
{
    PyObject *res = PyUnicode_FromString("This is a quad (128-bit float) dtype.");
    return res;
}

PyArray_DTypeMeta QuadDType = {
        {{
                PyVarObject_HEAD_INIT(NULL, 0).tp_name = "quaddtype.QuadDType",
                .tp_basicsize = sizeof(QuadDTypeObject),
                .tp_new = quaddtype_new,
                .tp_dealloc = (destructor)quaddtype_dealloc,
                .tp_repr = (reprfunc)quaddtype_repr,
                .tp_str = (reprfunc)quaddtype_repr,
        }},
};

int init_quad_dtype(void)
{
    PyArrayMethod_Spec *casts[] = {
            NULL,
    };

    PyArrayDTypeMeta_Spec QuadDType_DTypeSpec = {
            .flags = NPY_DT_NUMERIC,
            .casts = casts,
            .typeobj = QuadScalar_Type,
            .slots = NULL,
    };

    ((PyObject *)&QuadDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&QuadDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&QuadDType) < 0) {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(&QuadDType, &QuadDType_DTypeSpec) < 0) {
        return -1;
    }

    QuadDType.singleton = PyArray_GetDefaultDescr(&QuadDType);
    return 0;
}




