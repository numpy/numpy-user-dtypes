import numpy as np
import pytest

from stringdtype import StringDType, StringScalar


@pytest.fixture
def string_list():
    return ["abc", "def", "ghi"]


def test_scalar_creation():
    assert str(StringScalar("abc", StringDType())) == "abc"


def test_dtype_creation():
    assert str(StringDType()) == "StringDType"


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
    assert repr(arr) == f"array({str(data)}, dtype=StringDType)"


def test_array_creation_scalars(string_list):
    dtype = StringDType()
    arr = np.array(
        [
            StringScalar("abc", dtype=dtype),
            StringScalar("def", dtype=dtype),
            StringScalar("ghi", dtype=dtype),
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
    arr[1] = StringScalar("what", dtype=dtype)
    assert repr(arr) == repr(np.array(["abc", "what", "ghi"], dtype=dtype))
