#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H

// clang-format off
#include <Python.h>
#include "structmember.h"
// clang-format on

#include "static_string.h"

#define PY_ARRAY_UNIQUE_SYMBOL stringdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/halffloat.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

// not publicly exposed by the static string library so we need to define
// this here so we can define `elsize` and `alignment` on the descr
//
// if the layout of npy_packed_static_string ever changes in the future
// this may need to be updated.
#define SIZEOF_NPY_PACKED_STATIC_STRING 2 * sizeof(size_t)
#define ALIGNOF_NPY_PACKED_STATIC_STRING _Alignof(size_t)

typedef struct {
    PyArray_Descr base;
    PyObject *na_object;
    int coerce;
    int has_nan_na;
    int has_string_na;
    int array_owned;
    npy_static_string default_string;
    npy_static_string na_name;
    PyThread_type_lock *allocator_lock;
    // the allocator should only be directly accessed after
    // acquiring the allocator_lock and the lock should
    // be released immediately after the allocator is
    // no longer needed
    npy_string_allocator *allocator;
} StringDTypeObject;

typedef struct {
    PyArray_DTypeMeta base;
} StringDType_type;

extern StringDType_type StringDType;
extern PyTypeObject *StringScalar_Type;

static inline npy_string_allocator *
NpyString_acquire_allocator(StringDTypeObject *descr)
{
    if (!PyThread_acquire_lock(descr->allocator_lock, NOWAIT_LOCK)) {
        PyThread_acquire_lock(descr->allocator_lock, WAIT_LOCK);
    }
    return descr->allocator;
}

static inline void
NpyString_acquire_allocator2(StringDTypeObject *descr1,
                             StringDTypeObject *descr2,
                             npy_string_allocator **allocator1,
                             npy_string_allocator **allocator2)
{
    *allocator1 = NpyString_acquire_allocator(descr1);
    if (descr1 != descr2) {
        *allocator2 = NpyString_acquire_allocator(descr2);
    }
    else {
        *allocator2 = *allocator1;
    }
}

static inline void
NpyString_acquire_allocator3(StringDTypeObject *descr1,
                             StringDTypeObject *descr2,
                             StringDTypeObject *descr3,
                             npy_string_allocator **allocator1,
                             npy_string_allocator **allocator2,
                             npy_string_allocator **allocator3)
{
    NpyString_acquire_allocator2(descr1, descr2, allocator1, allocator2);
    if (descr1 != descr3 && descr2 != descr3) {
        *allocator3 = NpyString_acquire_allocator(descr3);
    }
    else {
        *allocator3 = descr3->allocator;
    }
}

static inline void
NpyString_release_allocator(StringDTypeObject *descr)
{
    PyThread_release_lock(descr->allocator_lock);
}

static inline void
NpyString_release_allocator2(StringDTypeObject *descr1,
                             StringDTypeObject *descr2)
{
    NpyString_release_allocator(descr1);
    if (descr1 != descr2) {
        NpyString_release_allocator(descr2);
    }
}

static inline void
NpyString_release_allocator3(StringDTypeObject *descr1,
                             StringDTypeObject *descr2,
                             StringDTypeObject *descr3)
{
    NpyString_release_allocator2(descr1, descr2);
    if (descr1 != descr3 && descr2 != descr3) {
        NpyString_release_allocator(descr3);
    }
}

PyObject *
new_stringdtype_instance(PyObject *na_object, int coerce);

int
init_string_dtype(void);

// Assumes that the caller has already acquired the allocator locks for both
// descriptors
int
_compare(void *a, void *b, StringDTypeObject *descr_a,
         StringDTypeObject *descr_b);

int
init_string_na_object(PyObject *mod);

int
stringdtype_setitem(StringDTypeObject *descr, PyObject *obj, char **dataptr);

// set the python error indicator when the gil is released
void
gil_error(PyObject *type, const char *msg);

// the locks on both allocators must be acquired before calling this function
int
free_and_copy(npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator,
              const npy_packed_static_string *in,
              npy_packed_static_string *out, const char *location);

PyArray_Descr *
stringdtype_finalize_descr(PyArray_Descr *dtype);

int
_eq_comparison(int scoerce, int ocoerce, PyObject *sna, PyObject *ona);

#endif /*_NPY_DTYPE_H*/
