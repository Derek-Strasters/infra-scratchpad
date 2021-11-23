"""Utility functions."""
from typing import Iterable, Literal, Protocol, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
Rotatable = TypeVar("Rotatable", bound="ConcatenableSequence")
ValidBit = Union[bool, Literal[1], Literal[0], Literal["1"], Literal["0"]]
ExtendableBits = Union[Iterable[ValidBit], Tuple[int, int]]


class ConcatenableSequence(Protocol[T_co]):
    """
    Any Sequence T where +(:T, :T) -> T.

    Types must support indexing and concatenation.

    >>> def concat_from_index(sequence: ConcatenableSequence):
    ...     return sequence[:-1] + sequence[-1:]
    >>> concat_from_index("abc")
    'abc'
    >>> concat_from_index((1, 2, 3))
    (1, 2, 3)
    >>> concat_from_index(["a", "b", "c"])
    ['a', 'b', 'c']
    >>> from biterator import Bits; concat_from_index(Bits('11011'))
    Bits("0b11011")
    """

    def __add__(self, other: Rotatable) -> Rotatable:
        """Concatenate."""
        ...

    def __getitem__(self, index: int) -> T_co:
        """Retrieve element."""
        ...

    def __len__(self) -> int:
        """Length of collection."""
        ...


def reverse_byte(byte: int) -> int:
    """
    Reverse the bit order of an 8 bit integer.

    >>> bin(reverse_byte(0b00010111))
    '0b11101000'
    """
    # 0 1 2 3 4 5 6 7
    byte = (byte & 0b00001111) << 4 | (byte & 0b11110000) >> 4
    # 4 5 6 7 0 1 2 3
    byte = (byte & 0b00110011) << 2 | (byte & 0b11001100) >> 2
    # 6 7 4 5 2 3 0 1
    byte = (byte & 0b01010101) << 1 | (byte & 0b10101010) >> 1
    # 7 6 5 4 3 2 1 0
    return byte


def chunked(items: Sequence[T], n: int) -> Sequence[Tuple[T]]:
    """
    Yield successive n-sized chunks from lst.

    The last chunk is trunkated as needed.

    :param items: A collection of things to be chunked.
    :param n: The size of each chunk.

     >>> list(chunked([1, 2, 3, 4, 5], 2))
     [(1, 2), (3, 4), (5,)]
     >>> list(chunked((1, 1, 1, 1, 1), 2))
     [(1, 1), (1, 1), (1,)]
    """
    sequence_type = type(items)
    # noinspection PyArgumentList
    return sequence_type((tuple(items[i : i + n]) for i in range(0, len(items), n)))


def rotate_left(sequence: Rotatable, n: int) -> Rotatable:
    """
    Rotate a sequence to the left.

    :param sequence: The sequence to rotate.
    :param n: The amount of rotation.
    :return: The rotated sequence.

    >>> rotate_left("hello", 2)
    'llohe'
    """
    return sequence[n:] + sequence[:n]


def rotate_right(sequence: Rotatable, n: int) -> Rotatable:
    """
    Rotate a sequence to the right.

    :param sequence: The sequence to rotate.
    :param n: The amount of rotation.
    :return: The rotated sequence.

    >>> rotate_right("hello", 2)
    'lohel'
    """
    return rotate_left(sequence, len(sequence) - n)


def convert_base(value: int, base: int) -> Iterable[int]:
    """
    Express a number with a specific base.

    :param value: The number to be expressed with a different base.
    :param base: The new base.
    :return: The "digits", starting with the lowest exponent.

    >>> tuple(convert_base(0x6, 2))
    (0, 1, 1)
    >>> tuple(convert_base(0xf, 10))
    (5, 1)
    """
    assert base > 0
    while value != 0:
        yield value % base
        value //= base


def split_every(data: Sequence[T], every: int) -> Tuple[Sequence[T]]:
    """
    Split data into groups of a given maximum length.

    :param data: The data to split.
    :param every: The maximum length of the resulting sequences.
    :return: The split sequences.

    >>> split_every("ABCDE", 2)
    ('AB', 'CD', 'E')
    """
    return tuple(data[every * i : every * (i + 1)] for i in range((len(data) + every - 1) // every))


def get_pairs(n: int) -> Iterable[Tuple[int, int]]:
    """
    Get pairs of indices that are not equal.

    :param n: The range of indices.
    :return: The pairs of non-equal indices.

    >>> tuple(get_pairs(2))
    ((0, 1), (1, 0))
    >>> tuple(get_pairs(3))
    ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
    """
    return ((a, b) for a in range(n) for b in range(n) if a != b)


def inverse_cumulative_sum(values: Sequence[int]) -> Tuple[int, ...]:
    """
    Calculate the differences between adjacent numbers in a sequence.

    :param values: The values.
    :return: The differences

    >>> inverse_cumulative_sum((1, 3, 8))
    (2, 5)
    """
    return tuple(b - a for a, b in zip(values, values[1:]))


def reverse_sequence(count: int) -> Iterable[int]:
    """
    Get a reverse sequence.

    :param count: How many items to get.
    :return: The reverse sequence.

    >>> tuple(reverse_sequence(5))
    (4, 3, 2, 1, 0)
    """
    yield from range(count - 1, -1, -1)
