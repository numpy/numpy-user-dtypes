from asciidtype import ASCIIDType, ASCIIScalar


def test_dtype_creation():
    dtype = ASCIIDType(4)
    assert str(dtype) == "ASCIIDType(4)"


def test_scalar_creation():
    dtype = ASCIIDType(5)
    ASCIIScalar('string', dtype)
