import concurrent.futures
import os
import pickle
import string
import tempfile

import numpy as np
import pandas as pd
import pytest

from stringdtype import NA, StringDType, StringScalar, _memory_usage


@pytest.fixture
def string_list():
    return ["abc", "def", "ghi", "A¢☃€ 😊", "Abc", "DEF"]


@pytest.fixture(
    params=[None, NA, pd.NA], ids=["None", "stringdtype.NA", "pandas.NA"]
)
def dtype(request):
    return StringDType(na_object=request.param)


def test_scalar_creation():
    assert str(StringScalar("abc")) == "abc"


def test_dtype_equality(dtype):
    assert dtype == dtype
    assert dtype != np.dtype("U")
    assert dtype != np.dtype("U8")


def test_dtype_repr(dtype):
    if dtype.na_object is NA:
        assert repr(dtype) == "StringDType()"
    else:
        assert repr(dtype) == f"StringDType(na_object={dtype.na_object})"


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
    assert repr(arr) == f"array({str(data)}, dtype={dtype})"


def test_array_creation_scalars(string_list):
    arr = np.array([StringScalar(s) for s in string_list])
    assert repr(arr) == repr(np.array(string_list, dtype=StringDType()))


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [b"abc", b"def", b"ghi"],
        [object, object, object],
    ],
)
def test_scalars_string_conversion(data):
    np.testing.assert_array_equal(
        np.array(data, dtype=StringDType()),
        np.array([str(d) for d in data], dtype=StringDType()),
    )


@pytest.mark.parametrize(
    ("strings"),
    [
        ["this", "is", "an", "array"],
        ["€", "", "😊"],
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
    ],
)
def test_unicode_casts(dtype, strings):
    arr = np.array(strings, dtype=np.unicode_).astype(dtype)
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


def test_additional_unicode_cast(dtype):
    RANDS_CHARS = np.array(
        list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
    )
    arr = np.random.choice(RANDS_CHARS, size=10 * 100_000, replace=True).view(
        "U10"
    )
    np.testing.assert_array_equal(arr, arr.astype(dtype))
    np.testing.assert_array_equal(arr, arr.astype(dtype).astype("U10"))


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
    sarr = np.array(string_list + [dtype.na_object], dtype=dtype)
    np.testing.assert_array_equal(
        np.isnan(sarr), np.array([0] * len(string_list) + [1], dtype=np.bool_)
    )


def test_memory_usage(dtype):
    sarr = np.array(["abc", "def", "ghi"], dtype=dtype)
    # 4 bytes for each ASCII string buffer in string_list
    # (three characters and null terminator)
    # plus enough bytes for the size_t length
    # plus enough bytes for the pointer in the array buffer
    assert _memory_usage(sarr) == (4 + 2 * np.dtype(np.uintp).itemsize) * 3
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
        arr.sort()
        assert np.array_equal(arr, arr_sorted, equal_nan=True)

    # make a copy so we don't mutate the lists in the fixture
    strings = strings.copy()
    arr_sorted = np.array(sorted(strings), dtype=dtype)
    test_sort(strings, arr_sorted)

    # make sure NAs get sorted to the end of the array
    strings.insert(0, dtype.na_object)
    strings.insert(2, dtype.na_object)
    # can't use append because doing that with NA converts
    # the result to object dtype
    arr_sorted = np.array(
        arr_sorted.tolist() + [dtype.na_object, dtype.na_object], dtype=dtype
    )

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

    assert np.zeros(3, dtype=dtype)[0] == ""

    assert np.all(np.isnan(np.empty(3, dtype=dtype)))

    assert np.empty(3, dtype=dtype)[0] is dtype.na_object


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


@pytest.mark.parametrize(
    "ufunc,func",
    [
        ("min", min),
        ("max", max),
    ],
)
def test_ufuncs_minmax(dtype, string_list, ufunc, func):
    """Test that the min/max ufuncs match Python builtin min/max behavior."""
    arr = np.array(string_list, dtype=dtype)
    np.testing.assert_array_equal(
        getattr(arr, ufunc)(), np.array(func(string_list), dtype=dtype)
    )


@pytest.mark.parametrize(
    "other_strings",
    [
        ["abc", "def", "ghi", "🤣", "📵", "😰"],
        ["🚜", "🙃", "😾", "😹", "🚠", "🚌"],
        ["🥦", "¨", "⨯", "∰ ", "⨌ ", "⎶ "],
    ],
)
def test_ufunc_add(dtype, string_list, other_strings):
    arr1 = np.array(string_list, dtype=dtype)
    arr2 = np.array(other_strings, dtype=dtype)
    np.testing.assert_array_equal(
        np.add(arr1, arr2),
        np.array([a + b for a, b in zip(arr1, arr2)], dtype=dtype),
    )


def test_create_with_na(dtype):
    na_val = dtype.na_object
    string_list = ["hello", na_val, "world"]
    arr = np.array(string_list, dtype=dtype)
    assert (
        repr(arr)
        == f"array(['hello', {dtype.na_object}, 'world'], dtype={dtype})"
    )
    assert arr[1] is dtype.na_object
