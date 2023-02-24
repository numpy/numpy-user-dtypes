import concurrent.futures
import os
import pickle
import tempfile

import numpy as np
import pytest

from stringdtype import StringDType, StringScalar, _memory_usage


@pytest.fixture
def string_list():
    return ["abc", "def", "ghi"]


def test_scalar_creation():
    assert str(StringScalar("abc")) == "abc"


def test_dtype_creation():
    assert str(StringDType()) == "StringDType()"


def test_dtype_equality():
    assert StringDType() == StringDType()
    assert StringDType() != np.dtype("U")
    assert StringDType() != np.dtype("U8")


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "def", "ghi"],
        ["🤣", "📵", "😰"],
        ["🚜", "🙃", "😾"],
        ["😹", "🚠", "🚌"],
    ],
)
def test_array_creation_utf8(data):
    arr = np.array(data, dtype=StringDType())
    assert repr(arr) == f"array({str(data)}, dtype=StringDType())"


def test_array_creation_scalars(string_list):
    arr = np.array(
        [
            StringScalar("abc"),
            StringScalar("def"),
            StringScalar("ghi"),
        ]
    )
    assert repr(arr) == repr(np.array(string_list, dtype=StringDType()))


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [None, None, None],
        [b"abc", b"def", b"ghi"],
        [object, object, object],
    ],
)
def test_bad_scalars(data):
    with pytest.raises(TypeError):
        np.array(data, dtype=StringDType())


@pytest.mark.parametrize(
    ("string_list"),
    [
        ["this", "is", "an", "array"],
        ["€", "", "😊"],
        ["A¢☃€ 😊", " A☃€¢😊", "☃€😊 A¢", "😊☃A¢ €"],
    ],
)
def test_unicode_casts(string_list):
    arr = np.array(string_list, dtype=np.unicode_).astype(StringDType())
    expected = np.array(string_list, dtype=StringDType())
    np.testing.assert_array_equal(arr, expected)

    arr = np.array(string_list, dtype=StringDType())

    np.testing.assert_array_equal(
        arr.astype("U8"), np.array(string_list, dtype="U8")
    )
    np.testing.assert_array_equal(arr.astype("U8").astype(StringDType()), arr)
    np.testing.assert_array_equal(
        arr.astype("U3"), np.array(string_list, dtype="U3")
    )
    np.testing.assert_array_equal(
        arr.astype("U3").astype(StringDType()),
        np.array([s[:3] for s in string_list], dtype=StringDType()),
    )


def test_insert_scalar(string_list):
    dtype = StringDType()
    arr = np.array(string_list, dtype=dtype)
    arr[1] = StringScalar("what")
    assert repr(arr) == repr(np.array(["abc", "what", "ghi"], dtype=dtype))


def test_equality_promotion(string_list):
    sarr = np.array(string_list, dtype=StringDType())
    uarr = np.array(string_list, dtype=np.str_)

    np.testing.assert_array_equal(sarr, uarr)
    np.testing.assert_array_equal(uarr, sarr)


def test_isnan(string_list):
    sarr = np.array(string_list, dtype=StringDType())
    np.testing.assert_array_equal(
        np.isnan(sarr), np.zeros_like(sarr, dtype=np.bool_)
    )


def test_memory_usage(string_list):
    sarr = np.array(string_list, dtype=StringDType())
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


def test_pickle(string_list):
    dtype = StringDType()

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
def test_sort(strings):
    """Test that sorting matches python's internal sorting."""
    arr = np.array(strings, dtype=StringDType())
    arr_sorted = np.array(sorted(strings), dtype=StringDType())

    np.random.default_rng().shuffle(arr)
    arr.sort()
    np.testing.assert_array_equal(arr, arr_sorted)


def test_creation_functions():
    np.testing.assert_array_equal(
        np.zeros(3, dtype=StringDType()), ["", "", ""]
    )

    np.testing.assert_array_equal(
        np.empty(3, dtype=StringDType()), ["", "", ""]
    )

    # make sure getitem works too
    assert np.empty(3, dtype=StringDType())[0] == ""
