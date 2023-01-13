import numpy as np
import pytest

from stringdtype import StringDType, StringScalar


@pytest.fixture
def string_list():
    return ['abc', 'def', 'ghi']


def test_scalar_creation():
    assert str(StringScalar('abc', StringDType())) == 'abc'


def test_dtype_creation():
    assert str(StringDType()) == 'StringDType'


@pytest.mark.parametrize(
    'data', [
        ['abc', 'def', 'ghi'],
        ["🤣", "📵", "😰"],
        ["🚜", "🙃", "😾"],
        ["😹", "🚠", "🚌"],
    ]
)
def test_array_creation_utf8(data):
    arr = np.array(data, dtype=StringDType())
    assert repr(arr) == f'array({str(data)}, dtype=StringDType)'


def test_array_creation_scalars(string_list):
    dtype = StringDType()
    arr = np.array(
        [
            StringScalar('abc', dtype=dtype),
            StringScalar('def', dtype=dtype),
            StringScalar('ghi', dtype=dtype),
        ]
    )
    assert repr(arr) == repr(np.array(string_list, dtype=StringDType()))


@pytest.mark.parametrize(
   'data', [
       [1, 2, 3],
       [None, None, None],
       [b'abc', b'def', b'ghi'],
       [object, object, object],
   ]
)
def test_bad_scalars(data):
    with pytest.raises(TypeError):
        np.array(data, dtype=StringDType())


@pytest.mark.xfail(reason='Not yet implemented')
def test_cast_to_stringdtype(string_list):
    arr = np.array(string_list, dtype='<U3').astype(StringDType())
    expected = np.array(string_list, dtype=StringDType())
    np.testing.assert_array_equal(arr, expected)


@pytest.mark.xfail(reason='Not yet implemented')
def test_cast_to_unicode_safe(string_list):
    arr = np.array(string_list, dtype=StringDType())

    np.testing.assert_array_equal(
        arr.astype('<U3', casting='safe'),
        np.array(string_list, dtype='<U3')
    )

    # Safe casting should preserve data
    with pytest.raises(TypeError):
        arr.astype('<U2', casting='safe')


@pytest.mark.xfail(reason='Not yet implemented')
def test_cast_to_unicode_unsafe(string_list):
    arr = np.array(string_list, dtype=StringDType())

    np.testing.assert_array_equal(
        arr.astype('<U3', casting='unsafe'),
        np.array(string_list, dtype='<U3')
    )

    # Unsafe casting: each element is truncated
    np.testing.assert_array_equal(
        arr.astype('<U2', casting='unsafe'),
        np.array(string_list, dtype='<U2')
    )


def test_insert_scalar(string_list):
    dtype = StringDType()
    arr = np.array(string_list, dtype=dtype)
    arr[1] = StringScalar('what', dtype=dtype)
    assert repr(arr) == repr(np.array(['abc', 'what', 'ghi'], dtype=dtype))
