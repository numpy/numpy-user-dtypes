#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

#include<Python.h>
#include<sleef.h>
#include<sleefquad.h>
#include<vector>

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"

#include "scalar.h"
#include "casts.h"
#include "dtype.h"


static NPY_CASTING quad_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        QuadPrecDTypeObject *given_descrs[2],
        QuadPrecDTypeObject *loop_descrs[2],
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

    *view_offset = 0;
    return NPY_SAME_KIND_CASTING;
}

static int quad_to_quad_strided_loop(
        PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    while (N--) {
        Sleef_quad *in = (Sleef_quad *)in_ptr;
        Sleef_quad *out = (Sleef_quad *)out_ptr;

        *out = *in;

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

static std::vector<PyArrayMethod_Spec *>specs;


PyArrayMethod_Spec ** init_casts_internal(void)
{
    PyArray_DTypeMeta **quad2quad_dtypes = new PyArray_DTypeMeta *[2]{nullptr, nullptr};

    specs.push_back(new PyArrayMethod_Spec {
    .name = "cast_QuadPrec_to_QuadPrec",
    .nin = 1,
    .nout = 1,
    .casting = NPY_SAME_KIND_CASTING,
    .flags = NPY_METH_SUPPORTS_UNALIGNED,
    .dtypes = quad2quad_dtypes,
    .slots = new PyType_Slot[3]{
        {NPY_METH_resolve_descriptors, (void *)&quad_to_quad_resolve_descriptors},
        {NPY_METH_strided_loop, (void *)&quad_to_quad_strided_loop},
        {0, NULL}
    }});

   
    return specs.data();
}

PyArrayMethod_Spec ** init_casts(void)
{
    try
    {
        return init_casts_internal();
    }
    catch(const std::exception& e)
    {
        PyErr_NoMemory();
        return nullptr;
    }
    
}

void free_casts(void)
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
