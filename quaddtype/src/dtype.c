#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

// #include "casts.h"
#include "dtype.h"

QuadDTypeObject * new_quaddtype_instance(void) {
    QuadDTypeObject *new = (QuadDTypeObject *)PyArrayDescr_Type.tp_new(
        (PyTypeObject *)&QuadDType, NULL, NULL
    );
    if (new == NULL) return NULL;

    new->base.elsize = sizeof(double);
    new->base.alignment = _Alignof(double);
    return new;
}

// Take an python double and put a copy into the array
static int quad_setitem(QuadDTypeObject *descr, PyObject *obj, char *dataptr) {
    double val = PyFloat_AsDouble(obj);
    memcpy(dataptr, &val, sizeof(double));
    return 0;
}

static PyObject *quad_getitem(QuadDTypeObject *descr, char *dataptr) {
    double val;
    memcpy(&val, dataptr, sizeof(double));

    PyObject *val_obj = PyFloat_FromDouble(val);
    if (val_obj == NULL) {
        return NULL;
    }

    Py_DECREF(val_obj); // Why decrement this pointer? Shouldn't this be Py_INCREF?
    return val_obj;
}

static PyType_Slot QuadDType_Slots[] = {
    // {NPY_DT_common_instance, &common_instance},
    // {NPY_DT_common_dtype, &common_dtype},
    // {NPY_DT_discover_descr_from_pyobject, &unit_discover_descriptor_from_pyobject},
    /* The header is wrong on main :(, so we add 1 */
    {NPY_DT_setitem, &quad_setitem},
    {NPY_DT_getitem, &quad_getitem},
    {0, NULL}
};

/*
 * The following defines everything type object related (i.e. not NumPy
 * specific).
 *
 * Note that this function is by default called without any arguments to fetch
 * a default version of the descriptor (in principle at least).  During init
 * we fill in `cls->singleton` though for the dimensionless unit.
 */
static PyObject *quaddtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwargs) {
    return (PyObject *)new_quaddtype_instance();
}

static void quaddtype_dealloc(QuadDTypeObject *self) {
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *quaddtype_repr(QuadDTypeObject *self) {
    return PyUnicode_FromString("This is a quad (128-bit float) dtype.");
}

/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
PyArray_DTypeMeta QuadDType = {
    {
        {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "quaddtype.QuadDType",
            .tp_basicsize = sizeof(QuadDTypeObject),
            .tp_new = quaddtype_new,
            .tp_dealloc = (destructor)quaddtype_dealloc,
            .tp_repr = (reprfunc)quaddtype_repr,
            .tp_str = (reprfunc)quaddtype_repr,
        }
    },
    /* rest, filled in during DTypeMeta initialization */
};

int init_quad_dtype(void);


// int init_unit_dtype(void)
// {
//     /*
//      * To create our DType, we have to use a "Spec" that tells NumPy how to
//      * do it.  You first have to create a static type, but see the note there!
//      */
//     PyArrayMethod_Spec *casts[] = {
//             &UnitToUnitCastSpec,
//             NULL};

//     PyArrayDTypeMeta_Spec QuadDType_DTypeSpec = {
//             .flags = NPY_DT_PARAMETRIC,
//             .casts = casts,
//             .typeobj = QuantityScalar_Type,
//             .slots = QuadDType_Slots,
//     };
//     /* Loaded dynamically, so may need to be set here: */
//     ((PyObject *)&QuadDType)->ob_type = &PyArrayDTypeMeta_Type;
//     ((PyTypeObject *)&QuadDType)->tp_base = &PyArrayDescr_Type;
//     if (PyType_Ready((PyTypeObject *)&QuadDType) < 0) {
//         return -1;
//     }

//     if (PyArrayInitDTypeMeta_FromSpec(
//             &QuadDType, &QuadDType_DTypeSpec) < 0) {
//         return -1;
//     }

//     /*
//      * Ensure that `singleton` is filled in (we rely on that).  It is possible
//      * to provide a custom `default_descr`, but it is filled in automatically
//      * to just call `DType()` -- however, it does not cache the result
//      * automatically (right now).  This is because it can make sense for a
//      * DType to requiring a new one each time (e.g. a Categorical that needs
//      * to be able to add new Categories).
//      * TODO: Consider flipping this around, so that if you need a new one
//      *       each time, you have to provide a custom `default_descr`.
//      */
//     QuadDType.singleton = PyArray_GetDefaultDescr(&QuadDType);

//     return 0;
// }
