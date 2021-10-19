"""Utility functions."""
from abc import abstractmethod
from collections.abc import MutableSequence
from typing import Iterable, Protocol, Sequence, Tuple, TypeVar, Union, overload, Literal

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
CS = TypeVar("CS", bound="ConcatenableSequence")
ValidBits = Union[bool, Literal[1], Literal[0], Literal["1"], Literal["0"]]


class ConcatenableSequence(Protocol[T_co]):
    """
    Any Sequence T where +(:T, :T) -> T.
    Types must support indexing and concatenation.

    >>> def test(a: CS, b: CS) -> CS: return a + b
    >>> test('abc', 'def') # passes
    'abcdef'
    >>> test(tuple(1, 2, 3), tuple(4, 5)) # passes type check
    >>> test(list(1, 2), list(3, 4)) # passes type check

    """

    def __add__(self: CS, other: CS) -> CS:
        ...

    def __getitem__(self: CS, index: int) -> T_co:
        ...

    def __len__(self) -> int:
        ...


# NOTE Finland uses iso8859_10 encoding.
# NOTE The proper bitwise operation for a NOR mask is (mask_ ^ left_) & (left_ | mask_) & ~(right & mask_).


class Bits(MutableSequence[ValidBits]):
    """
    Stores bit like objects with bitwise and math special methods implemented.
    Iterating a Bits object will give one byte as an int per iteration.


    Representations:
    >>> print(Bits(bin="01100101"))
    'e'
    >>> print(Bits(bin="0110 0101"))
    'e'
    >>> a = Bits(bin='01010100 01001111 00100000 0b01001111 00001010')
    >>> print(a)
    'TO O\x0a'
    >>> print(Bits(int=362104770314))
    'TO O\x0a'
    >>> print(Bits("Hello"))
    'Hello'
    >>> print(a.pad())
    'T   O       O x0A'
    >>> print(a.pad(5))
    'T    O         O  x0A'
    >>> int(a)
    362104770314
    >>> format(a, '#042b')
    '0b0101010001001111001000000100111100001010'
    >>> a.bin
    '0b0101010001001111001000000100111100001010'
    >>> a.hex
    '0x544f204f0a'

    >>> (Bits(bin='01111000') | Bits(bin='00011110')).bin
    '0b01111110'
    >>> (Bits(bin='01111000') & Bits(bin='00011110')).bin
    '0b00011000'
    >>> (Bits(bin='01111000') ^ Bits(bin='00011110')).bin
    '0b01100110
    >>> ~Bits(bin='11110000')
    '0b00001111'

    >>> (Bits(bin='1001') + Bits(int('0001', 2)))

    >>> mask_ = Bits(bin='00001111')
    >>> left_ = Bits(bin='01010101')
    >>> right = Bits(bin='00110011')
    #      01011010
    >>> (mask_ ^ left_) & (mask_ | left_) & ~(mask_ & right)
    '0b01011000'

    >>> bits = Bits()
    >>> bits.append(0)
    >>> bits.bin
    0
    >>> bits.append(1)
    >>> bits.append('1')
    >>> bits.append(False)
    >>> bits.bin
    0110

    >>> bits.extend(Bits(16))
    >>> bits.bin
    01101000

    >>> bits[0]
    0
    >>> bits[1]
    1
    >>> bits[4:]
    1000

    """

    __slots__ = ["__data", "__len"]
    _ZEROS = {0, "0", False}
    _ONES = {1, "1", True}

    __data: int
    __len: int

    # TODO: handle bytes objects
    # TODO: return bytes objects (implement __bytes__)

    def __init__(
        self,
        value: Union[int, bytes, bool, Iterable[ValidBits], "Bits"] = 0,
        /,
        length: int = None,
        bin_: str = None,
        hex_: str = None,
        bool_: bool = None,
        bytes_: Union[bytes, bytearray] = None,
    ):
        # TODO: STUB
        if isinstance(value, int):
            self.__data = value
            self.__len = length if length else self.__data.bit_length()
            return
        raise TypeError

    def __repr__(self):
        # TODO: limit to certain size
        if self.__len <= 24:
            return self.bin

    def __str__(self):
        # TODO: STUB
        pass

    def __bytes__(self):
        num_whole_bytes = self.__len // 8
        num_leftover_bits = self.__len % 8
        whole_bytes = (self.__data >> num_leftover_bits).to_bytes(num_whole_bytes, "big")
        if num_leftover_bits:
            leftover = (self.__data & ((1 << num_leftover_bits) - 1)).to_bytes(1, "big")
            return whole_bytes + leftover
        return whole_bytes

    def __bool__(self):
        return bool(self.__data)

    # Mutable Sequence Methods

    def insert(self, index: int, value: bool) -> None:
        if value in self._ONES:
            value = True
        elif value in self._ZEROS:
            value = False
        else:
            raise TypeError(f"Could not determine bit value for {value}.")
        if self.__len < index or index < -self.__len:
            raise IndexError
        # Needed because it is allowed to insert at index = len
        index = index % self.__len if index != self.__len else index
        pos_from_end = self.__len - index
        self.__len += 1
        left_bits = self.__data >> pos_from_end
        left_bits = ((left_bits << 1) | value) << pos_from_end
        right_bits = self.__data & ((1 << pos_from_end) - 1)
        self.__data = left_bits | right_bits

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> bool:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[bool]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.__len)
            if step == 1:
                # self[2:6]  # len(self) = 8
                #     v v v v
                # 0 1 2 3 4 5 6 7
                new_int = self.__data >> self.__len - stop
                new_int &= (1 << stop - start) - 1
                return type(self)(new_int, length=stop - start)
            interval = stop - start
            new_len = interval // step + bool(interval % step)
            new_int = 0
            for i in range(start, stop, step):
                new_int <<= 1
                new_int |= self._get_bit(i)
            return type(self)(new_int, length=new_len)
        if self.__len > index >= -self.__len:
            index = index % self.__len
            mask = 1 << self.__len - index - 1
            return bool(mask & self.__data)
        raise IndexError

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: bool) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[Union[bool, int, str]]) -> None:
        ...

    def __setitem__(self, index, other):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.__len)
            range_ = range(start, stop, step)
            for i, new_val in zip(range_, other):
                if new_val in self._ONES:
                    self._bit_on(i)
                    continue
                if new_val in self._ZEROS:
                    self._bit_off(i)
                    continue
                raise TypeError(f"Could not determine single bit value for {new_val}.")
            return
        if self.__len > index >= -self.__len:
            index = index % self.__len
            if other in self._ONES:
                self._bit_on(index)
                return
            if other in self._ZEROS:
                self._bit_off(index)
                return
            raise TypeError(f"Could not determine single bit value for {other}.")
        raise IndexError

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None:
        ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None:
        ...

    def __delitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.__len)
            if step == 1:
                # del self[2:6]  # len(self) = 8
                #     v v v v
                # 0 1 2 3 4 5 6 7
                left_bits = self.__data >> self.__len - start
                left_bits = left_bits << self.__len - stop
                right_bits = self.__data & ((1 << self.__len - stop) - 1)
                self.__data = left_bits | right_bits
                self.__len -= stop - start
                return
            for i in reversed(sorted(range(start, stop, step))):
                self._delitem_at(i)
            return
        self._delitem_at(index)

    def __len__(self) -> int:
        return self.__len

    # END Mutable Sequence Methods

    def __add__(self, other: Iterable[ValidBits]) -> "Bits":
        """
        This is NOT addition, this is concatenation.

        :param other: Other object to be concatenated.
        :return: New Bits object that is a concatenation of the inputs.
        """
        # TODO: Iterable[ValidBits]
        if isinstance(other, Bits):
            return type(self)((self.__data << other.__len) | other.__data, length=self.__len + other.__len)
        return NotImplemented

    def __lshift__(self, other: int) -> "Bits":
        if isinstance(other, int):
            return type(self)(self.__data << other, length=self.__len + other)
        return NotImplemented

    def __rshift__(self, other: int) -> "Bits":
        if isinstance(other, int):
            return type(self)(self.__data >> other, length=self.__len - other)
        return NotImplemented

    def __and__(self, other: "Bits") -> "Bits":
        if isinstance(other, Bits):
            if self.__len == other.__len:
                return type(self)(self.__data & other.__data, length=self.__len)
            longest_len = max(self.__len, other.__len)
            self_data = self.__data << longest_len - self.__len
            other_data = other.__data << longest_len - other.__len
            return type(self)(self_data & other_data, length=longest_len)
        return NotImplemented

    def __xor__(self, other: "Bits") -> "Bits":
        if isinstance(other, Bits):
            if self.__len == other.__len:
                return type(self)(self.__data ^ other.__data, length=self.__len)
            longest_len = max(self.__len, other.__len)
            self_data = self.__data << longest_len - self.__len
            other_data = other.__data << longest_len - other.__len
            return type(self)(self_data ^ other_data, length=longest_len)
        return NotImplemented

    def __or__(self, other: "Bits") -> "Bits":
        if isinstance(other, Bits):
            if self.__len == other.__len:
                return type(self)(self.__data | other.__data, length=self.__len)
            longest_len = max(self.__len, other.__len)
            self_data = self.__data << longest_len - self.__len
            other_data = other.__data << longest_len - other.__len
            return type(self)(self_data | other_data, length=longest_len)
        return NotImplemented

    def __iadd__(self, other: Iterable[ValidBits]):
        # TODO: STUB
        pass

    def __ilshift__(self, other):
        # TODO: STUB
        pass

    def __irshift__(self, other):
        # TODO: STUB
        pass

    def __iand__(self, other):
        # TODO: STUB
        pass

    def __ixor__(self, other):
        # TODO: STUB
        pass

    def __ior__(self, other):
        # TODO: STUB
        pass

    def __invert__(self) -> "Bits":
        return type(self)(self.__data ^ ((1 << self.__len) - 1), length=self.__len)

    def __int__(self) -> int:
        return self.__data

    def __index__(self) -> int:
        return self.__data

    def _delitem_at(self, index: int):
        # TODO: docstrings
        if self.__len <= index or index < -self.__len:
            raise IndexError
        index = index % self.__len
        # del self[i:i+1], i=2  # len(self) = 8
        #     v
        # 0 1 2 3 4 5 6 7
        # pos_from_end = self.__len - index
        left_bits = self.__data >> self.__len - index
        left_bits = left_bits << self.__len - (index + 1)
        right_bits = self.__data & ((1 << self.__len - (index + 1)) - 1)
        self.__data = left_bits | right_bits
        self.__len -= 1

    def _digit_mask(self, index: int) -> int:
        """
        Returns a mask for the digit at the given index.

        :param index: The index of the desired digit mask from the left
        :return: The digit mask

                           v
        >>> bin(Bits(0b11001100)._digit_mask(5))
        '0b1000'
        """
        return 1 << self.__len - index - 1

    def _get_bit(self, index: int) -> bool:
        # TODO: docstrings
        return bool(self.__data & self._digit_mask(index))

    def _bit_off(self, index: int) -> None:
        # TODO: docstrings
        mask = self._digit_mask(index)
        if mask & self.__data:
            self.__data ^= mask

    def _bit_on(self, index: int) -> None:
        # TODO: docstrings
        self.__data |= self._digit_mask(index)

    @property
    def bin(self) -> str:
        # TODO: docstring
        if not self.__len % 8:
            return " ".join(f"{byte:#010b}" for byte in self.__data.to_bytes(self.__len // 8, "big"))
        num_bytes = self.__len // 8
        num_extra_bits = self.__len % 8
        bin_str = " ".join(f"{byte:#010b}" for byte in (self.__data >> num_extra_bits).to_bytes(num_bytes, "big"))
        return bin_str + " " + format(((1 << num_extra_bits) - 1) & self.__data, f"#0{num_extra_bits + 2}b")


def reverse_byte(byte: int) -> int:
    """
    Reverse the bit order of an 8 bit integer.

    >>> bin(203)
    '0b11101000'
    >>> bin(reverse_byte(203))
    '0b00010111'
    """

    # 0 1 2 3 4 5 6 7
    byte = (byte & 0b00001111) << 4 | (byte & 0b11110000) >> 4
    # 4 5 6 7 0 1 2 3
    byte = (byte & 0b00110011) << 2 | (byte & 0b11001100) >> 2
    # 6 7 4 5 2 3 0 1
    byte = (byte & 0b01010101) << 1 | (byte & 0b10101010) >> 1
    # 7 6 5 4 3 2 1 0
    return byte


def chunked(items: Sequence[T], n: int) -> Sequence[Sequence[T]]:
    """
    Yield successive n-sized chunks from lst.
    The last chunk is trunkated as needed.

    :param items: A collection of things to be chunked.
    :param n: The size of each chunk.

     >>> list(chunked([1, 2, 3, 4, 5], 2))
     [[1, 2], [3, 4], [5]]
     >>> list(chunked((1, 1, 1, 1, 1), 2))
     ((1), (1), (1))
    """
    # noinspection PyArgumentList
    return type(items)(type(items)(items[i : i + n]) for i in range(0, len(items), n))


def rotate_left(sequence: CS, n: int) -> CS:
    """
    Rotate a sequence to the left.

    :param sequence: The sequence to rotate.
    :param n: The amount of rotation.
    :return: The rotated sequence.

    >>> rotate_left("hello", 2)
    'llohe'
    """
    return sequence[n:] + sequence[:n]


def rotate_right(sequence: CS, n: int) -> CS:
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
