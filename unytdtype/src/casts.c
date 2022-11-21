#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unytdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"

#include "casts.h"
#include "dtype.h"

/*
 * Helper function also used elsewhere to make sure the unit is a unit.
 *
 * obj is assumed to be a python string containing the name of a unit
 *
 * NOTE: Supports if `obj` is NULL (meaning nothing is passed on)
 */
int
UnitConverter(PyObject *obj, PyObject **unit)
{
    static PyObject *unyt_mod = NULL;
    if (NPY_UNLIKELY(unyt_mod == NULL)) {
        unyt_mod = PyImport_ImportModule("unyt");
        if (unyt_mod == NULL) {
            Py_DECREF(unyt_mod);
            return -1;
        }
    }
    *unit = PyObject_GetAttr(unyt_mod, obj);
    if (*unit == NULL) {
        return 0;
    }
    return 1;
}

/*
 * Find the conversion from one unit to another, may use NULL to denote
 * dimensionless without any scaling/offset.
 *
 */

int
get_conversion_factor(PyObject *from_unit, PyObject *to_unit, double *factor,
                      double *offset)
{
    if (from_unit == to_unit) {
        *factor = 1;
        *offset = 0;
        return 0;
    }

    PyObject *tmp = PyUnicode_FromString("");
    static PyObject *dimensionless = NULL;
    int res = UnitConverter(tmp, &dimensionless);
    Py_DECREF(tmp);
    if (res == 0) {
        return -1;
    }

    if (from_unit == NULL) {
        from_unit = dimensionless;
    }
    if (to_unit == NULL) {
        to_unit = dimensionless;
    }

    PyObject *get_conversion_factor =
            PyObject_GetAttrString(from_unit, "get_conversion_factor");

    PyObject *conv =
            PyObject_CallFunctionObjArgs(get_conversion_factor, to_unit, NULL);

    if (conv == NULL) {
        return -1;
    }

    if (!PyTuple_CheckExact(conv) || PyTuple_GET_SIZE(conv) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "unit error, conversion was not a two-element tuple.");
        return -1;
    }

    // TODO: Do I need to check types better?
    *factor = PyFloat_AsDouble(PyTuple_GET_ITEM(conv, 0));
    if (*factor == -1 && PyErr_Occurred()) {
        Py_DECREF(conv);
        return -1;
    }

    *offset = 0;

    if (PyTuple_GET_ITEM(conv, 1) != Py_None) {
        double off = PyFloat_AsDouble(PyTuple_GET_ITEM(conv, 1));
        if (off == -1 && PyErr_Occurred()) {
            Py_DECREF(conv);
            return -1;
        }
        *offset = -off;
    }
    Py_DECREF(conv);

    return 0;
}

/*
 * And now the actual cast code!  Starting with the "resolver" which tells
 * us about cast safety.
 * Note also the `view_offset`!  It is a way for you to tell NumPy, that this
 * cast does not require anything at all, but the cast can simply be done as
 * a view.
 * For `arr.astype()` it might mean returning a view (eventually, not yet).
 * For ufuncs, it already means that they don't have to do a cast at all!
 */
static NPY_CASTING
unit_to_unit_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                 PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                 PyArray_Descr *given_descrs[2],
                                 PyArray_Descr *loop_descrs[2],
                                 npy_intp *view_offset)
{
    double factor = 1., offset = 0.;
    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        if (get_conversion_factor(((UnytDTypeObject *)given_descrs[0])->unit,
                                  ((UnytDTypeObject *)given_descrs[1])->unit,
                                  &factor, &offset) < 0) {
            return -1;
        }
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (factor == 1 && offset == 0) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }

    /* Should this use safe casting? */
    return NPY_SAME_KIND_CASTING;
}

typedef struct {
    NpyAuxData base;
    double factor;
    double offset;
} conv_auxdata;

static void
conv_auxdata_free(conv_auxdata *conv_auxdata)
{
    PyMem_Free(conv_auxdata);
}

/*
 * Get the conversion factor and offset for conversion between units and
 * store it into "auxdata" for the strided-loop!
 */
static conv_auxdata *
get_conv_factor_auxdata(PyArray_Descr *from_dt, PyArray_Descr *to_dt)
{
    conv_auxdata *res = PyMem_Calloc(1, sizeof(conv_auxdata));
    if (res < 0) {
        PyErr_NoMemory();
        return NULL;
    }
    res->base.free = (void *)conv_auxdata_free;

    PyObject *from_unit = NULL, *to_unit = NULL;
    if (Py_TYPE(from_dt) == (PyTypeObject *)&UnytDType) {
        from_unit = ((UnytDTypeObject *)from_dt)->unit;
    }
    if (Py_TYPE(to_dt) == (PyTypeObject *)&UnytDType) {
        to_unit = ((UnytDTypeObject *)to_dt)->unit;
    }

    if (get_conversion_factor(from_unit, to_unit, &res->factor, &res->offset) <
        0) {
        NPY_AUXDATA_FREE((NpyAuxData *)res);
        return NULL;
    }
    return res;
}

static int
unit_to_unit_contiguous_no_offset(PyArrayMethod_Context *NPY_UNUSED(context),
                                  char *const data[],
                                  npy_intp const dimensions[],
                                  npy_intp const NPY_UNUSED(strides[]),
                                  conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    npy_intp N = dimensions[0];
    double *in = (double *)data[0];
    double *out = (double *)data[1];

    while (N--) {
        *out = factor * *in;
        out++;
        in++;
    }
    return 0;
}

static int
unit_to_unit_strided_no_offset(PyArrayMethod_Context *NPY_UNUSED(context),
                               char *const data[], npy_intp const dimensions[],
                               npy_intp const strides[], conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        *(double *)out = factor * *(double *)in;
        in += in_stride;
        out += out_stride;
    }
    return 0;
}

static int
unit_to_unit_contiguous_offset(PyArrayMethod_Context *NPY_UNUSED(context),
                               char *const data[], npy_intp const dimensions[],
                               npy_intp const NPY_UNUSED(strides[]),
                               conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    double offset = auxdata->offset;
    npy_intp N = dimensions[0];
    double *in = (double *)data[0];
    double *out = (double *)data[1];

    while (N--) {
        *out = factor * *in + offset;
        out++;
        in++;
    }
    return 0;
}

static int
unit_to_unit_strided_offset(PyArrayMethod_Context *NPY_UNUSED(context),
                            char *const data[], npy_intp const dimensions[],
                            npy_intp const strides[], conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    double offset = auxdata->offset;
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        *(double *)out = factor * *(double *)in + offset;
        in += in_stride;
        out += out_stride;
    }
    return 0;
}

static int
unit_to_unit_unaligned(PyArrayMethod_Context *NPY_UNUSED(context),
                       char *const data[], npy_intp const dimensions[],
                       npy_intp const strides[], conv_auxdata *auxdata)
{
    double factor = auxdata->factor;
    double offset = auxdata->offset;
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        double in_val, out_val;
        memcpy(&in_val, in, sizeof(double));
        out_val = factor * (in_val + offset);
        memcpy(out, &out_val, sizeof(double));
        in += in_stride;
        out += out_stride;
    }
    return 0;
}

static int
unit_to_unit_get_loop(PyArrayMethod_Context *context, int aligned,
                      int NPY_UNUSED(move_references), const npy_intp *strides,
                      PyArrayMethod_StridedLoop **out_loop,
                      NpyAuxData **out_transferdata,
                      NPY_ARRAYMETHOD_FLAGS *flags)
{
    conv_auxdata *conv_auxdata = get_conv_factor_auxdata(
            context->descriptors[0], context->descriptors[1]);
    if (conv_auxdata == NULL) {
        return -1;
    }
    *out_transferdata = (NpyAuxData *)conv_auxdata;

    int contig =
            (strides[0] == sizeof(double) && strides[1] == sizeof(double));
    int no_offset = conv_auxdata->offset == 0;

    if (aligned && contig && no_offset) {
        *out_loop = (PyArrayMethod_StridedLoop
                             *)&unit_to_unit_contiguous_no_offset;
    }
    else if (aligned && contig) {
        *out_loop =
                (PyArrayMethod_StridedLoop *)&unit_to_unit_contiguous_offset;
    }
    else if (aligned && no_offset) {
        *out_loop =
                (PyArrayMethod_StridedLoop *)&unit_to_unit_strided_no_offset;
    }
    else if (aligned) {
        *out_loop = (PyArrayMethod_StridedLoop *)&unit_to_unit_strided_offset;
    }
    else {
        *out_loop = (PyArrayMethod_StridedLoop *)&unit_to_unit_unaligned;
    }

    *flags = 0;
    return 0;
}


static NPY_CASTING unit_to_float64_resolve_descriptors(PyObject *NPY_UNUSED(self),
													   PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
													   PyArray_Descr *NPY_UNUSED(given_descrs[2]),
													   PyArray_Descr *NPY_UNUSED(loop_descrs[2]),
													   npy_intp *NPY_UNUSED(view_offset))
{
	return NPY_SAME_KIND_CASTING;
}

static int
unit_to_float64(PyArrayMethod_Context *NPY_UNUSED(context),
				char *const *NPY_UNUSED(data[]),
				npy_intp const *NPY_UNUSED(dimensions[]),
				npy_intp const *NPY_UNUSED(strides[]),
				conv_auxdata *NPY_UNUSED(auxdata))
{
    return 0;
}

static int
unit_to_float64_get_loop(PyArrayMethod_Context *context, int aligned,
						 int NPY_UNUSED(move_references), const npy_intp *strides,
						 PyArrayMethod_StridedLoop **out_loop,
						 NpyAuxData **out_transferdata,
						 NPY_ARRAYMETHOD_FLAGS *flags)
{
	*out_loop = (PyArrayMethod_StridedLoop *)&unit_to_float64;

    *flags = 0;
    return 0;
}

PyArrayMethod_Spec** get_casts(void) {

    /*
	 * NumPy currently allows NULL for the own DType/"cls".
	 */
	PyArray_DTypeMeta *u2u_dtypes[2] = {NULL, NULL};

	PyType_Slot u2u_slots[] = {
        {NPY_METH_resolve_descriptors, &unit_to_unit_resolve_descriptors},
        {_NPY_METH_get_loop, &unit_to_unit_get_loop},
        {0, NULL}
	};

	PyArrayMethod_Spec UnitToUnitCastSpec = {
        .name = "cast_UnytDType_to_UnytDType",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .casting = NPY_SAME_KIND_CASTING,
        .dtypes = u2u_dtypes,
        .slots = u2u_slots,
	};

	PyArray_DTypeMeta *u2f_dtypes[2] = {NULL, &PyArray_CDoubleDType};

	PyType_Slot u2f_slots[] = {
		{NPY_METH_resolve_descriptors, &unit_to_float64_resolve_descriptors},
		{_NPY_METH_get_loop, &unit_to_float64_get_loop},
		{0, NULL}
	};

	PyArrayMethod_Spec UnitToFloat64CastSpec = {
		.name = "cast_UnytDType_to_Float64",
		.nin = 1,
		.nout = 1,
		.flags = NPY_METH_SUPPORTS_UNALIGNED,
		.casting = NPY_SAME_KIND_CASTING,
		.dtypes = u2f_dtypes,
		.slots = u2f_slots,
	};

	PyArrayMethod_Spec** casts = malloc(3*sizeof(PyArrayMethod_Spec*));
	memcpy(casts, (PyArrayMethod_Spec*[]){
			&UnitToUnitCastSpec, &UnitToFloat64CastSpec, NULL
		}, 3*sizeof(PyArrayMethod_Spec*));
	return casts;
}
