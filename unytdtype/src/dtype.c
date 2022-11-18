#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unytdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"

#include "casts.h"
#include "dtype.h"

PyTypeObject *UnytScalar_Type = NULL;

/*
 * `get_value` and `get_unit` are small helpers to deal with the scalar.
 */

// NJG hack: get_value assumes scalar is a float64 - possible to generalize?
static double get_value(PyObject *scalar) {
  PyTypeObject* scalar_type = Py_TYPE(scalar);
  if (scalar_type != UnytScalar_Type) {
    double res = PyFloat_AsDouble(scalar);
    if (res == -1 && PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError,
                      "UnytDType arrays can only store numeric data");
      return -1;
    }
    return res;
  }

  PyObject *value = PyObject_GetAttrString(scalar, "value");
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot get value from scalar");
    return -1;
  }
  double res = PyFloat_AsDouble(value);
  if (res == -1 && PyErr_Occurred()) {
    return -1;
  }
  Py_DECREF(value);
  return res;
}

static PyObject *get_unit(PyObject *scalar, UnytDTypeObject *descr) {
  if (Py_TYPE(scalar) != UnytScalar_Type) {
    return descr->unit;
  }

  PyObject *unit = PyObject_GetAttrString(scalar, "unit");
  if (unit == NULL) {
    return NULL;
  }

  return unit;
}

/*
 * Internal helper to create new instances
 */
UnytDTypeObject *new_unytdtype_instance(PyObject *unit) {
  UnytDTypeObject *new = (UnytDTypeObject *)PyArrayDescr_Type.tp_new(
      /* TODO: Using NULL for args here works, but seems not clean? */
      (PyTypeObject *)&UnytDType, NULL, NULL);
  if (new == NULL) {
    return NULL;
  }
  Py_INCREF(unit);
  new->unit = unit;
  new->base.elsize = sizeof(double);
  new->base.alignment = _Alignof(double); /* is there a better spelling? */
  /* do not support byte-order for now */

  return new;
}

/*
 * This is used to determine the correct dtype to return when operations mix
 * dtypes (I think?). For now just return the first one.
 */
static UnytDTypeObject *common_instance(UnytDTypeObject *dtype1,
                                        UnytDTypeObject *dtype2) {
  double factor, offset;
  if (get_conversion_factor(dtype1->unit, dtype2->unit, &factor, &offset) < 0) {
    return NULL;
  }
  if (offset != 0 || fabs(factor) > 1.) {
    Py_INCREF(dtype1);
    return dtype1;
  } else {
    Py_INCREF(dtype2);
    return dtype2;
  }
}

static PyArray_DTypeMeta *common_dtype(PyArray_DTypeMeta *cls,
                                       PyArray_DTypeMeta *other) {
  /*
   * Typenum is useful for NumPy, but there it can still be convenient.
   * (New-style user dtypes will probably get -1 as type number...)
   */
  if (other->type_num >= 0 && PyTypeNum_ISNUMBER(other->type_num) &&
      !PyTypeNum_ISCOMPLEX(other->type_num) &&
      other != &PyArray_LongDoubleDType) {
    /*
     * A (simple) builtin numeric type that is not a complex or longdouble
     * will always promote to double (cls).
     */
    Py_INCREF(cls);
    return cls;
  }
  Py_INCREF(Py_NotImplemented);
  return (PyArray_DTypeMeta *)Py_NotImplemented;
}

static PyArray_Descr *
unit_discover_descriptor_from_pyobject(PyArray_DTypeMeta *NPY_UNUSED(cls),
                                       PyObject *obj) {
  if (Py_TYPE(obj) != UnytScalar_Type) {
    PyErr_SetString(PyExc_TypeError,
                    "UnytDType arrays can only store UnytScalar instances.");
    return NULL;
  }

  PyObject *unit = PyObject_GetAttrString(obj, "unit");
  if (unit == NULL) {
    return NULL;
  }

  return (PyArray_Descr *)new_unytdtype_instance(unit);
}

static int unytdtype_setitem(UnytDTypeObject *descr, PyObject *obj,
                             char *dataptr) {
  double value = get_value(obj);
  if (value == -1 && PyErr_Occurred()) {
    return -1;
  }
  PyObject *unit = get_unit(obj, descr);
  if (unit == NULL) {
    return -1;
  }

  double factor, offset;
  if (get_conversion_factor(unit, descr->unit, &factor, &offset) < 0) {
    return -1;
  }

  value = factor * (value + offset);

  memcpy(dataptr, &value, sizeof(double)); // NOLINT

  return 0;
}

static PyObject *unytdtype_getitem(UnytDTypeObject *descr, char *dataptr) {
  double val;
  /* get the value */
  memcpy(&val, dataptr, sizeof(double)); // NOLINT

  PyObject *val_obj = PyFloat_FromDouble(val);
  if (val_obj == NULL) {
    return NULL;
  }

  PyObject *res = PyObject_CallFunctionObjArgs((PyObject *)UnytScalar_Type,
                                               val_obj, descr->unit, NULL);
  if (res == NULL) {
    return NULL;
  }
  Py_DECREF(val_obj);

  return res;
}

static PyType_Slot UnytDType_Slots[] = {
    {NPY_DT_common_instance, &common_instance},
    {NPY_DT_common_dtype, &common_dtype},
    {NPY_DT_discover_descr_from_pyobject,
     &unit_discover_descriptor_from_pyobject},
    /* The header is wrong on main :(, so we add 1 */
    {NPY_DT_setitem, &unytdtype_setitem},
    {NPY_DT_getitem, &unytdtype_getitem},
    {0, NULL}};

static PyObject *unytdtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args,
                               PyObject *kwds) {
  static char *kwargs_strs[] = {"unit", NULL};

  PyObject *unit = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&:UnytDType", kwargs_strs,
                                   &UnitConverter, &unit)) {
    return NULL;
  }

  if (unit == NULL) {
    PyObject *tmp = PyUnicode_FromString("dimensionless");
    if (tmp == NULL) {
      return NULL;
    }
    if (!UnitConverter(tmp, &unit)) {
      return NULL;
    }
    Py_DECREF(tmp);
  }

  PyObject *res = (PyObject *)new_unytdtype_instance(unit);
  Py_DECREF(unit);
  return res;
}

static void unytdtype_dealloc(UnytDTypeObject *self) {
  Py_CLEAR(self->unit);
  PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *unytdtype_repr(UnytDTypeObject *self) {
  PyObject *res = PyUnicode_FromFormat("UnytDType('%R')", self->unit);
  return res;
}

/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
PyArray_DTypeMeta UnytDType = {
    {{
        PyVarObject_HEAD_INIT(NULL, 0).tp_name = "unytdtype.UnytDType",
        .tp_basicsize = sizeof(UnytDTypeObject),
        .tp_new = unytdtype_new,
        .tp_dealloc = (destructor)unytdtype_dealloc,
        .tp_repr = (reprfunc)unytdtype_repr,
        .tp_str = (reprfunc)unytdtype_repr,
    }},
    /* rest, filled in during DTypeMeta initialization */
};

int init_unyt_dtype(void) {
  /*
   * To create our DType, we have to use a "Spec" that tells NumPy how to
   * do it.  You first have to create a static type, but see the note there!
   */
  PyArrayMethod_Spec *casts[] = {&UnitToUnitCastSpec, NULL};

  PyArrayDTypeMeta_Spec UnytDType_DTypeSpec = {
      .flags = NPY_DT_PARAMETRIC,
      .casts = casts,
      .typeobj = UnytScalar_Type,
      .slots = UnytDType_Slots,
  };
  /* Loaded dynamically, so may need to be set here: */
  ((PyObject *)&UnytDType)->ob_type = &PyArrayDTypeMeta_Type;
  ((PyTypeObject *)&UnytDType)->tp_base = &PyArrayDescr_Type;
  if (PyType_Ready((PyTypeObject *)&UnytDType) < 0) {
    return -1;
  }

  if (PyArrayInitDTypeMeta_FromSpec(&UnytDType, &UnytDType_DTypeSpec) < 0) {
    return -1;
  }

  UnytDType.singleton = PyArray_GetDefaultDescr(&UnytDType);

  return 0;
}
