import concurrent.futures
import os
import pickle
import string
import tempfile

import numpy as np

try:
    from pandas import NA as pd_NA
except ImportError:
    pd_NA = None
import pytest

from stringdtype import StringDType, StringScalar, _memory_usage


@pytest.fixture
def string_list():
    return ["abc", "def", "ghi" * 10, "A¢☃€ 😊" * 100, "Abc" * 1000, "DEF"]


@pytest.fixture
def random_string_list():
    chars = list(string.ascii_letters + string.digits)
    chars = np.array(chars, dtype="U1")
    ret = np.random.choice(chars, size=100 * 1000, replace=True)
    return ret.view("U100")


pd_param = pytest.param(
    pd_NA,
    marks=pytest.mark.skipif(pd_NA is None, reason="pandas is not installed"),
)


@pytest.fixture(params=[True, False])
def coerce(request):
    return request.param


@pytest.fixture(
    params=["unset", None, pd_param, np.nan, float("nan"), "__nan__"],
    ids=["unset", "None", "pandas.NA", "np.nan", "float('nan')", "string nan"],
)
def na_object(request):
    return request.param


@pytest.fixture()
def dtype(na_object, coerce):
    # explicit is check for pd_NA because != with pd_NA returns pd_NA
    if na_object is pd_NA or na_object != "unset":
        return StringDType(na_object=na_object, coerce=coerce)
    else:
        return StringDType(coerce=coerce)


# second copy for cast tests to do a cartesian product over dtypes
@pytest.fixture()
def dtype2(na_object, coerce):
    # explicit is check for pd_NA because != with pd_NA returns pd_NA
    if na_object is pd_NA or na_object != "unset":
        return StringDType(na_object=na_object, coerce=coerce)
    else:
        return StringDType(coerce=coerce)


def test_dtype_creation():
    hashes = set()
    dt = StringDType()
    assert not hasattr(dt, "na_object") and dt.coerce == 1
    hashes.add(hash(dt))

    dt = StringDType(na_object=None)
    assert dt.na_object is None and dt.coerce == 1
    hashes.add(hash(dt))

    dt = StringDType(coerce=False)
    assert not hasattr(dt, "na_object") and dt.coerce == 0
    hashes.add(hash(dt))

    dt = StringDType(na_object=None, coerce=False)
    assert dt.na_object is None and dt.coerce == 0
    hashes.add(hash(dt))

    assert len(hashes) == 4


def test_scalar_creation():
    assert str(StringScalar("abc")) == "abc"


def test_dtype_equality(dtype):
    assert dtype == dtype
    assert dtype != np.dtype("U")
    assert dtype != np.dtype("U8")


def test_dtype_repr(dtype):
    if not hasattr(dtype, "na_object") and dtype.coerce == 1:
        assert repr(dtype) == "StringDType()"
    elif dtype.coerce == 1:
        assert repr(dtype) == f"StringDType(na_object={repr(dtype.na_object)})"
    elif not hasattr(dtype, "na_object"):
        assert repr(dtype) == "StringDType(coerce=False)"
    else:
        assert (
            repr(dtype)
            == f"StringDType(na_object={repr(dtype.na_object)}, coerce=False)"
        )


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "def", "ghi"],
        ["🤣", "📵", "😰"],
        ["🚜", "🙃", "😾"],
        ["😹", "🚠", "🚌"],
    ],
)
def test_array_creation_utf8(dtype, data):
    arr = np.array(data, dtype=dtype)
    assert str(arr) == "[" + " ".join(["'" + str(d) + "'" for d in data]) + "]"
    assert arr.dtype == dtype


def test_array_creation_scalars(string_list):
    arr = np.array([StringScalar(s) for s in string_list])
    assert (
        str(arr).replace("\n", "")
        == "[" + " ".join(["'" + str(s) + "'" for s in string_list]) + "]"
    )
    assert arr.dtype == StringDType()


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [b"abc", b"def", b"ghi"],
        [object, object, object],
    ],
)
def test_scalars_string_conversion(data, dtype):
    if dtype.coerce != 0:
        np.testing.assert_array_equal(
            np.array(data, dtype=dtype),
            np.array([str(d) for d in data], dtype=dtype),
        )
    else:
        with pytest.raises(ValueError):
            np.array(data, dtype=dtype)


@pytest.mark.parametrize(
    ("strings"),
    [
        ["this", "is", "an", "array"],
        ["€", "", "😊"],
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
    ],
)
def test_self_casts(dtype, dtype2, strings):
    if hasattr(dtype, "na_object"):
        strings = strings + [dtype.na_object]
    arr = np.array(strings, dtype=dtype)
    newarr = arr.astype(dtype2)

    if hasattr(dtype, "na_object") and not hasattr(dtype2, "na_object"):
        assert newarr[-1] == str(dtype.na_object)
        with pytest.raises(TypeError):
            arr.astype(dtype2, casting="safe")
        arr.astype(dtype2, casting="unsafe")
    elif hasattr(dtype, "na_object") and hasattr(dtype2, "na_object"):
        assert newarr[-1] is dtype2.na_object
        arr.astype(dtype2, casting="safe")
    else:
        arr.astype(dtype2, casting="safe")

    np.testing.assert_array_equal(arr[:-1], newarr[:-1])


@pytest.mark.parametrize(
    ("strings"),
    [
        ["this", "is", "an", "array"],
        ["€", "", "😊"],
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
    ],
)
def test_unicode_casts(dtype, strings):
    arr = np.array(strings, dtype=np.str_).astype(dtype)
    expected = np.array(strings, dtype=dtype)
    np.testing.assert_array_equal(arr, expected)

    arr = np.array(strings, dtype=dtype)

    np.testing.assert_array_equal(
        arr.astype("U8"), np.array(strings, dtype="U8")
    )
    np.testing.assert_array_equal(arr.astype("U8").astype(dtype), arr)
    np.testing.assert_array_equal(
        arr.astype("U3"), np.array(strings, dtype="U3")
    )
    np.testing.assert_array_equal(
        arr.astype("U3").astype(dtype),
        np.array([s[:3] for s in strings], dtype=dtype),
    )


def test_additional_unicode_cast(random_string_list, dtype):
    arr = np.array(random_string_list, dtype=dtype)
    np.testing.assert_array_equal(arr, arr.astype(dtype))
    np.testing.assert_array_equal(
        arr, arr.astype(dtype).astype(random_string_list.dtype)
    )


def test_insert_scalar(dtype, string_list):
    """Test that inserting a scalar works."""
    arr = np.array(string_list, dtype=dtype)
    for scalar_instance in ["what", StringScalar("what")]:
        arr[1] = scalar_instance
        np.testing.assert_array_equal(
            arr,
            np.array(
                string_list[:1] + ["what"] + string_list[2:], dtype=dtype
            ),
        )


comparison_operators = [
    np.equal,
    np.not_equal,
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
]


@pytest.mark.parametrize("op", comparison_operators)
@pytest.mark.parametrize("o_dtype", [np.str_, object, StringDType()])
def test_comparisons(string_list, dtype, op, o_dtype):
    sarr = np.array(string_list, dtype=dtype)
    oarr = np.array(string_list, dtype=o_dtype)

    # test that comparison operators work
    res = op(sarr, sarr)
    ores = op(oarr, oarr)
    # test that promotion works as well
    orres = op(sarr, oarr)
    olres = op(oarr, sarr)

    np.testing.assert_array_equal(res, ores)
    np.testing.assert_array_equal(res, orres)
    np.testing.assert_array_equal(res, olres)

    # test we get the correct answer for unequal length strings
    sarr2 = np.array([s + "2" for s in string_list], dtype=dtype)
    oarr2 = np.array([s + "2" for s in string_list], dtype=o_dtype)

    res = op(sarr, sarr2)
    ores = op(oarr, oarr2)
    olres = op(oarr, sarr2)
    orres = op(sarr, oarr2)

    np.testing.assert_array_equal(res, ores)
    np.testing.assert_array_equal(res, olres)
    np.testing.assert_array_equal(res, orres)

    res = op(sarr2, sarr)
    ores = op(oarr2, oarr)
    olres = op(oarr2, sarr)
    orres = op(sarr2, oarr)

    np.testing.assert_array_equal(res, ores)
    np.testing.assert_array_equal(res, olres)
    np.testing.assert_array_equal(res, orres)


def test_isnan(dtype, string_list):
    if not hasattr(dtype, "na_object"):
        pytest.skip("no na support")
    sarr = np.array(string_list + [dtype.na_object], dtype=dtype)
    is_nan = isinstance(dtype.na_object, float) and np.isnan(dtype.na_object)
    bool_errors = 0
    try:
        bool(dtype.na_object)
    except TypeError:
        bool_errors = 1
    if is_nan or bool_errors:
        # isnan is only true when na_object is a NaN
        np.testing.assert_array_equal(
            np.isnan(sarr),
            np.array([0] * len(string_list) + [1], dtype=np.bool_),
        )
    else:
        assert not np.any(np.isnan(sarr))


def test_memory_usage(dtype):
    sarr = np.array(["abcdefghijklmnopqrstuvqxyz", "def", "ghi"], dtype=dtype)
    # 26 bytes for the long string buffer in string_list
    # plus enough bytes for the size_t length
    # plus enough bytes for the pointer in the array buffer
    assert _memory_usage(sarr) == (2 * np.dtype(np.uintp).itemsize) * 3 + 26
    with pytest.raises(TypeError):
        _memory_usage("hello")
    with pytest.raises(TypeError):
        _memory_usage(np.array([1, 2, 3]))


def _pickle_load(filename):
    with open(filename, "rb") as f:
        res = pickle.load(f)

    return res


def test_pickle(dtype, string_list):
    arr = np.array(string_list, dtype=dtype)

    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        pickle.dump([arr, dtype], f)

    with open(f.name, "rb") as f:
        res = pickle.load(f)

    np.testing.assert_array_equal(res[0], arr)
    assert res[1] == dtype

    # load the pickle in a subprocess to ensure the string data are
    # actually stored in the pickle file
    with concurrent.futures.ProcessPoolExecutor() as executor:
        e = executor.submit(_pickle_load, f.name)
        res = e.result()

    np.testing.assert_array_equal(res[0], arr)
    assert res[1] == dtype

    os.remove(f.name)


@pytest.mark.parametrize(
    "strings",
    [
        ["left", "right", "leftovers", "righty", "up", "down"],
        [
            "left" * 10,
            "right" * 10,
            "leftovers" * 10,
            "righty" * 10,
            "up" * 10,
        ],
        ["🤣🤣", "🤣", "📵", "😰"],
        ["🚜", "🙃", "😾"],
        ["😹", "🚠", "🚌"],
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
    ],
)
def test_sort(dtype, strings):
    """Test that sorting matches python's internal sorting."""

    def test_sort(strings, arr_sorted):
        arr = np.array(strings, dtype=dtype)
        np.random.default_rng().shuffle(arr)
        na_object = getattr(arr.dtype, "na_object", "")
        if na_object is None and None in strings:
            with pytest.raises(ValueError):
                arr.sort()
        else:
            arr.sort()
            assert np.array_equal(arr, arr_sorted, equal_nan=True)

    # make a copy so we don't mutate the lists in the fixture
    strings = strings.copy()
    arr_sorted = np.array(sorted(strings), dtype=dtype)
    test_sort(strings, arr_sorted)

    if not hasattr(dtype, "na_object"):
        return

    # make sure NAs get sorted to the end of the array and string NAs get
    # sorted like normal strings
    strings.insert(0, dtype.na_object)
    strings.insert(2, dtype.na_object)
    # can't use append because doing that with NA converts
    # the result to object dtype
    if not isinstance(dtype.na_object, str):
        arr_sorted = np.array(
            arr_sorted.tolist() + [dtype.na_object, dtype.na_object],
            dtype=dtype,
        )
    else:
        arr_sorted = np.array(sorted(strings), dtype=dtype)

    test_sort(strings, arr_sorted)


@pytest.mark.parametrize(
    "strings",
    [
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
        ["A¢☃€ 😊", "", " ", " "],
        ["", "a", "😸", "ááðfáíóåéë"],
    ],
)
def test_nonzero(dtype, strings):
    arr = np.array(strings, dtype=dtype)
    is_nonzero = np.array([i for i, item in enumerate(arr) if len(item) != 0])
    np.testing.assert_array_equal(arr.nonzero()[0], is_nonzero)


def test_creation_functions(dtype):
    np.testing.assert_array_equal(np.zeros(3, dtype=dtype), ["", "", ""])
    np.testing.assert_array_equal(np.empty(3, dtype=dtype), ["", "", ""])

    assert np.zeros(3, dtype=dtype)[0] == ""
    assert np.empty(3, dtype=dtype)[0] == ""


def test_is_numeric(dtype):
    assert not type(dtype)._is_numeric


@pytest.mark.parametrize(
    "strings",
    [
        ["left", "right", "leftovers", "righty", "up", "down"],
        ["🤣🤣", "🤣", "📵", "😰"],
        ["🚜", "🙃", "😾"],
        ["😹", "🚠", "🚌"],
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
    ],
)
def test_argmax(dtype, strings):
    """Test that argmax/argmin matches what python calculates."""
    arr = np.array(strings, dtype=dtype)
    assert np.argmax(arr) == strings.index(max(strings))
    assert np.argmin(arr) == strings.index(min(strings))


@pytest.mark.parametrize(
    "arrfunc,expected",
    [
        [np.sort, None],
        [np.nonzero, (np.array([], dtype=np.int64),)],
        [np.argmax, 0],
        [np.argmin, 0],
    ],
)
def test_arrfuncs_zeros(dtype, arrfunc, expected):
    arr = np.zeros(10, dtype=dtype)
    result = arrfunc(arr)
    if expected is None:
        expected = arr
    np.testing.assert_array_equal(result, expected, strict=True)


@pytest.mark.parametrize(
    ("strings", "cast_answer", "any_answer", "all_answer"),
    [
        [["hello", "world"], [True, True], True, True],
        [["", ""], [False, False], False, False],
        [["hello", ""], [True, False], True, False],
        [["", "world"], [False, True], True, False],
    ],
)
def test_cast_to_bool(dtype, strings, cast_answer, any_answer, all_answer):
    sarr = np.array(strings, dtype=dtype)
    np.testing.assert_array_equal(sarr.astype("bool"), cast_answer)

    assert np.any(sarr) == any_answer
    assert np.all(sarr) == all_answer


@pytest.mark.parametrize(
    ("strings", "cast_answer"),
    [
        [[True, True], ["True", "True"]],
        [[False, False], ["False", "False"]],
        [[True, False], ["True", "False"]],
        [[False, True], ["False", "True"]],
    ],
)
def test_cast_from_bool(dtype, strings, cast_answer):
    barr = np.array(strings, dtype=bool)
    np.testing.assert_array_equal(
        barr.astype(dtype), np.array(cast_answer, dtype=dtype)
    )


@pytest.mark.parametrize("bitsize", [8, 16, 32, 64])
@pytest.mark.parametrize("signed", [True, False])
def test_sized_integer_casts(dtype, bitsize, signed):
    idtype = f"int{bitsize}"
    if signed:
        inp = [-(2**p - 1) for p in reversed(range(bitsize - 1))]
        inp += [2**p - 1 for p in range(1, bitsize - 1)]
    else:
        idtype = "u" + idtype
        inp = [2**p - 1 for p in range(bitsize)]
    ainp = np.array(inp, dtype=idtype)
    np.testing.assert_array_equal(ainp, ainp.astype(dtype).astype(idtype))

    with pytest.raises(TypeError):
        ainp.astype(dtype, casting="safe")

    with pytest.raises(TypeError):
        ainp.astype(dtype).astype(idtype, casting="safe")

    oob = [str(2**bitsize), str(-(2**bitsize))]
    with pytest.raises(OverflowError):
        np.array(oob, dtype=dtype).astype(idtype)


@pytest.mark.parametrize("typename", ["byte", "short", "int", "longlong"])
@pytest.mark.parametrize("signed", ["", "u"])
def test_unsized_integer_casts(dtype, typename, signed):
    idtype = f"{signed}{typename}"

    inp = [1, 2, 3, 4]
    ainp = np.array(inp, dtype=idtype)
    np.testing.assert_array_equal(ainp, ainp.astype(dtype).astype(idtype))


@pytest.mark.parametrize("typename", ["float64", "float32", "float16"])
def test_float_casts(dtype, typename):
    inp = [1.1, 2.8, -3.2, 2.7e4]
    ainp = np.array(inp, dtype=typename)
    np.testing.assert_array_equal(ainp, ainp.astype(dtype).astype(typename))

    fi = np.finfo(typename)

    inp = [1e-324, fi.smallest_subnormal, -1e-324, -fi.smallest_subnormal]
    eres = [0, fi.smallest_subnormal, -0, -fi.smallest_subnormal]
    res = np.array(inp, dtype=typename).astype(dtype).astype(typename)
    np.testing.assert_array_equal(eres, res)

    inp = [2e308, fi.max, -2e308, fi.min]
    eres = [np.inf, fi.max, -np.inf, fi.min]
    res = np.array(inp, dtype=typename).astype(dtype).astype(typename)
    np.testing.assert_array_equal(eres, res)

    inp = [0.1]
    sres = np.array(inp, dtype=typename).astype(dtype)
    res = sres.astype(typename)
    np.testing.assert_array_equal(np.array(inp, dtype=typename), res)
    assert sres[0] == "0.1"


def test_take(dtype, string_list):
    sarr = np.array(string_list, dtype=dtype)
    out = np.empty(len(string_list), dtype=dtype)
    res = sarr.take(np.arange(len(string_list)), out=out)
    np.testing.assert_array_equal(sarr, res)
    np.testing.assert_array_equal(res, out)

    # make sure it also works for out that isn't empty
    out[0] = "hello"
    res = sarr.take(np.arange(len(string_list)), out=out)
    np.testing.assert_array_equal(res, out)


@pytest.mark.parametrize("use_out", [[True, False]])
@pytest.mark.parametrize(
    "ufunc_name,func",
    [
        ("min", min),
        ("max", max),
    ],
)
def test_ufuncs_minmax(dtype, string_list, ufunc_name, func, use_out):
    """Test that the min/max ufuncs match Python builtin min/max behavior."""
    arr = np.array(string_list, dtype=dtype)
    uarr = np.array(string_list, dtype=str)
    res = np.array(func(string_list), dtype=dtype)
    np.testing.assert_array_equal(getattr(arr, ufunc_name)(), res)

    ufunc = getattr(np, ufunc_name + "imum")

    if use_out:
        res = ufunc(arr, arr, out=arr)
    else:
        res = ufunc(arr, arr)

    np.testing.assert_array_equal(uarr, res)


@pytest.mark.parametrize("use_out", [[True, False]])
@pytest.mark.parametrize(
    "other_strings",
    [
        ["abc", "def" * 500, "ghi" * 16, "🤣" * 100, "📵", "😰"],
        ["🚜", "🙃", "😾", "😹", "🚠", "🚌"],
        ["🥦", "¨", "⨯", "∰ ", "⨌ ", "⎶ "],
    ],
)
def test_ufunc_add(dtype, string_list, other_strings, use_out):
    arr1 = np.array(string_list, dtype=dtype)
    arr2 = np.array(other_strings, dtype=dtype)
    result = np.array([a + b for a, b in zip(arr1, arr2)], dtype=dtype)

    if use_out:
        res = np.add(arr1, arr2, out=arr1)
    else:
        res = np.add(arr1, arr2)

    np.testing.assert_array_equal(res, result)

    if not hasattr(dtype, "na_object"):
        return

    is_nan = isinstance(dtype.na_object, float) and np.isnan(dtype.na_object)
    is_str = isinstance(dtype.na_object, str)
    bool_errors = 0
    try:
        bool(dtype.na_object)
    except TypeError:
        bool_errors = 1

    arr1 = np.array([dtype.na_object] + string_list, dtype=dtype)
    arr2 = np.array(other_strings + [dtype.na_object], dtype=dtype)

    if is_nan or bool_errors or is_str:
        res = np.add(arr1, arr2)
        np.testing.assert_array_equal(res[1:-1], arr1[1:-1] + arr2[1:-1])
        if not is_str:
            assert res[0] is dtype.na_object and res[-1] is dtype.na_object
        else:
            assert res[0] == dtype.na_object + arr2[0]
            assert res[-1] == arr1[-1] + dtype.na_object
    else:
        with pytest.raises(TypeError):
            np.add(arr1, arr2)


@pytest.mark.parametrize("use_out", [[True, False]])
@pytest.mark.parametrize("other", [2, [2, 1, 3, 4, 1, 3]])
@pytest.mark.parametrize(
    "other_dtype",
    [
        None,
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "short",
        "int",
        "intp",
        "long",
        "longlong",
        "ushort",
        "uint",
        "uintp",
        "ulong",
        "ulonglong",
    ],
)
def test_ufunc_multiply(dtype, string_list, other, other_dtype, use_out):
    """Test the two-argument ufuncs match python builtin behavior."""
    arr = np.array(string_list, dtype=dtype)
    if other_dtype is not None:
        other_dtype = np.dtype(other_dtype)
    try:
        len(other)
        result = [s * o for s, o in zip(string_list, other)]
        other = np.array(other)
        if other_dtype is not None:
            other = other.astype(other_dtype)
    except TypeError:
        if other_dtype is not None:
            other = other_dtype.type(other)
        result = [s * other for s in string_list]

    if use_out:
        arr_cache = arr.copy()
        lres = np.multiply(arr, other, out=arr)
        arr[:] = arr_cache
        rres = np.multiply(other, arr, out=arr)
    else:
        lres = arr * other
        rres = other * arr

    np.testing.assert_array_equal(lres, result)
    np.testing.assert_array_equal(rres, result)

    if not hasattr(dtype, "na_object"):
        return

    is_nan = isinstance(dtype.na_object, float) and np.isnan(dtype.na_object)
    is_str = isinstance(dtype.na_object, str)
    bool_errors = 0
    try:
        bool(dtype.na_object)
    except TypeError:
        bool_errors = 1

    arr = np.array(string_list + [dtype.na_object], dtype=dtype)

    try:
        len(other)
        other = np.append(other, 3)
        if other_dtype is not None:
            other = other.astype(other_dtype)
    except TypeError:
        pass

    if is_nan or bool_errors or is_str:
        for res in [arr * other, other * arr]:
            np.testing.assert_array_equal(res[:-1], result)
            if not is_str:
                assert res[-1] is dtype.na_object
            else:
                try:
                    assert res[-1] == dtype.na_object * other[-1]
                except (IndexError, TypeError):
                    assert res[-1] == dtype.na_object * other
    else:
        with pytest.raises(TypeError):
            arr * other
        with pytest.raises(TypeError):
            other * arr


def test_create_with_na(dtype):
    if not hasattr(dtype, "na_object"):
        pytest.skip("does not have an na object")
    na_val = dtype.na_object
    string_list = ["hello", na_val, "world"]
    arr = np.array(string_list, dtype=dtype)
    assert str(arr) == "[" + " ".join([repr(s) for s in string_list]) + "]"
    assert arr[1] is dtype.na_object


def test_datetime_cast(dtype):
    a = np.array(
        [
            np.datetime64("1923-04-14T12:43:12"),
            np.datetime64("1994-06-21T14:43:15"),
            np.datetime64("2001-10-15T04:10:32"),
            np.datetime64("NaT"),
            np.datetime64("1995-11-25T16:02:16"),
            np.datetime64("2005-01-04T03:14:12"),
            np.datetime64("2041-12-03T14:05:03"),
        ]
    )

    has_na = hasattr(dtype, "na_object")
    is_str = isinstance(getattr(dtype, "na_object", None), str)

    if not has_na or is_str:
        a = np.delete(a, 3)

    sa = a.astype(dtype)
    ra = sa.astype(a.dtype)

    if has_na and not is_str:
        assert sa[3] is dtype.na_object
        assert np.isnat(ra[3])

    np.testing.assert_array_equal(a, ra)

    if has_na and not is_str:
        # don't worry about comparing how NaT is converted
        sa = np.delete(sa, 3)
        a = np.delete(a, 3)

    np.testing.assert_array_equal(sa, a.astype("U"))


def test_null_roundtripping(dtype):
    data = ["hello\0world", "ABC\0DEF\0\0"]
    arr = np.array(data, dtype=dtype)
    assert data[0] == arr[0]
    assert data[1] == arr[1]


def test_string_too_large_error():
    arr = np.array(["a", "b", "c"], dtype=StringDType())
    with pytest.raises(MemoryError):
        arr * (2**63 - 2)


def test_growing_strings(dtype):
    # growing a string leads to a heap allocation, this tests to make sure
    # we do that bookeeping correctly for all possible starting cases
    data = [
        "hello",  # a short string
        "abcdefghijklmnopqestuvwxyz",  # a medium heap-allocated string
        "hello" * 200,  # a long heap-allocated string
    ]

    arr = np.array(data, dtype=dtype)
    uarr = np.array(data, dtype=str)

    for _ in range(5):
        arr = arr + arr
        uarr = uarr + uarr

    np.testing.assert_array_equal(arr, uarr)


def test_threaded_access_and_mutation(dtype, random_string_list):
    # this test uses an RNG and may crash or cause deadlocks if there is a
    # threading bug
    rng = np.random.default_rng(0x4D3D3D3)

    def func(arr):
        rnd = rng.random()
        # either write to random locations in the array, compute a ufunc, or
        # re-initialize the array
        if rnd < 0.25:
            num = np.random.randint(0, arr.size)
            arr[num] = arr[num] + "hello"
        elif rnd < 0.5:
            if rnd < 0.375:
                np.add(arr, arr)
            else:
                np.add(arr, arr, out=arr)
        elif rnd < 0.75:
            if rnd < 0.875:
                np.multiply(arr, np.int64(2))
            else:
                np.multiply(arr, np.int64(2), out=arr)
        else:
            arr[:] = random_string_list

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as tpe:
        arr = np.array(random_string_list, dtype=dtype)
        futures = [tpe.submit(func, arr) for _ in range(500)]

        for f in futures:
            f.result()
