#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "umath.h"


/*
 * The multiplication loop, this is very minimal!  Look at the cast to see
 * some of the more advanced things you can do for optimization!
 *
 * Look at the seberg/unitdtype exaple repository for how this can be done
 * more generically without implementing a multiply loop!
 */
static int
unit_multiply_strided_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1 = data[0], *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[0];
    npy_intp out_stride = strides[2];

    while (N--) {
        *(double *)out = *(double *)in1 * *(double *)in2;
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }
    return 0;
}


/*
 * This is the "dtype/descriptor resolver".  Its main job is to fill in the
 * result dtype in this case, that is:
 *
 *     given_descrs[0] = arr1.dtype
 *     given_descrs[1] = arr2.dtype
 *     given_descrs[2] = NULL or out.dtype
 *
 * The code now sets `loop_descrs` to what we actually get as a result.
 * In practice that is just fill in the last `out.dtype`.  But in principle,
 * we might need casting (e.g. swap `>f8` to `<f8`), so the input dtypes could
 * also be changed (but we do not do this here!).
 */
static NPY_CASTING
unit_multiply_resolve_descriptors(PyObject *self,
        PyArray_DTypeMeta *dtypes[], PyArray_Descr *given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *unused)
{
    /* Fetch the unyt based units: */
    PyObject *unit1 = ((UnitDTypeObject *)given_descrs[0])->unit;
    PyObject *unit2 = ((UnitDTypeObject *)given_descrs[1])->unit;
    /* Find the correct result unit: */
    PyObject *new_unit = PyNumber_Multiply(unit1, unit2);
    if (new_unit == NULL) {
        return -1;
    }
    /* Create new DType from the new unit: */
    loop_descrs[2] = new_unitdtype_instance(new_unit);
    if (loop_descrs[2] == NULL) {
        return -1;
    }
    /* The other operand units can be used as-is: */
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    return NPY_NO_CASTING;
}


/*
 * Function that adds our multiply loop to NumPy's multiply ufunc.
 */
int
init_multiply_ufunc(void)
{
    /*
     * Get the multiply ufunc:
     */
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        return -1;
    }
    PyObject *multiply = PyObject_GetAttrString(numpy, "multiply");
    Py_DECREF(numpy);
    if (multiply == NULL) {
        return -1;
    }

    /*
     * The initializing "wrap up" code from the slides (plus one error check)
     */
    static PyArray_DTypeMeta *dtypes[3] = {
       &UnitDType, &UnitDType, &UnitDType};

    static PyType_Slot slots[] = {
       {NPY_METH_resolve_descriptors,
            &unit_multiply_resolve_descriptors},
       {NPY_METH_strided_loop,
            &unit_multiply_strided_loop},
       {0, NULL}
    };

    PyArrayMethod_Spec MultiplySpec = {
        .name = "unit_multiply",
        .nin = 2,
        .nout = 1,
        .dtypes = dtypes,
        .slots = slots,
        .flags = 0,
        .casting = NPY_NO_CASTING,
    };

    /* Register */
    if (PyUFunc_AddLoopFromSpec(multiply, &MultiplySpec) < 0) {
        Py_DECREF(multiply);
        return -1;
    }
    Py_DECREF(multiply);
    return 0;
}
