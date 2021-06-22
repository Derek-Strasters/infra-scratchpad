"""Binary string utility functions."""

from typing import Tuple


def binary_decode(binary: str) -> str:
    """
    Decode an 8-bit binary number.

    :param binary: A binary string.
    :return: The character representation.

    >>> binary_decode("01010011")
    'S'
    """
    assert len(binary) == 8 and all(c in ("0", "1") for c in binary)
    return chr(int(binary, 2))


def binary_decode_multi(words: str) -> str:
    """
    Decode a string of binary codes.

    :param words: The binary codes.
    :return: The decoded string.

    >>> binary_decode_multi("01010011 01101111")
    'So'
    """
    return "".join(map(binary_decode, words.split()))


def invert_bits(binary: str) -> str:
    """
    Invert the bits of some binary code.

    :param binary: The binary string.
    :return: The inverted binary string.

    >>> invert_bits("0110")
    '1001'
    """
    assert all(c in ("0", "1") for c in binary)
    return "".join(str(1 - int(c)) for c in binary)


def binary_encode(string: str) -> Tuple[str]:
    """
    Encode some string as binary.

    :param string: The string to encode.
    :return: The characters encoded as binary.

    >>> binary_encode("test")
    ('01110100', '01100101', '01110011', '01110100')
    """

    def to_validated_binary(code: int) -> str:
        if not (0 < code < 0x80):
            raise ValueError("string contains non-ASCII characters")

        return bin(code)[2:].rjust(8, "0")

    return tuple(map(to_validated_binary, map(ord, string)))
