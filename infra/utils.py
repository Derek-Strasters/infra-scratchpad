"""Utility functions."""
from abc import abstractmethod
from collections.abc import MutableSequence
from math import log
from typing import (
    ByteString,
    Generator,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
    SupportsInt,
)

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
    Stores bits. Bitwise math operator have been implemented.
    Iterating a Bits object will give one bit as a boolean per iteration.


    Representations:
    >>> print(Bits(bin_="01100101"))
    b'e'
    >>> print(Bits(bin_="0110 0101"))
    b'e'
    >>> a = Bits(bin_='01010100 01001111 00100000 0b01001111 00001010')
    >>> print(a.decode('utf-8'))
    'TO O\x0a'
    >>> print(Bits(int_=362104770314, length=40).decode('utf-8'))
    'TO O\x0a'
    >>> print(Bits(chars=b"Hello").decode('utf-8'))
    'Hello'
    >>> int(a)
    362104770314
    >>> format(int(a), '#042b')
    '0b0101010001001111001000000100111100001010'
    >>> a.bin
    '0b0101_0100 0b0100_1111 0b0010_0000 0b0100_1111 0b0000_1010'
    >>> a.hex
    '0x544f204f0a'

    >>> (Bits(bin_='01111000') | Bits(bin_='00011110')).bin
    '0b01111110'
    >>> (Bits(bin_='01111000') & Bits(bin_='00011110')).bin
    '0b00011000'
    >>> (Bits(bin_='01111000') ^ Bits(bin_='00011110')).bin
    '0b01100110
    >>> (~Bits(bin_='11110000')).bin
    '0b00001111'

    >>> (Bits(bin='1001') + Bits(int_=int('0001', 2))).bin
    '0b1010'


    Nor mask:
    >>> mask_ = Bits(bin='00001111')
    >>> left_ = Bits(bin='01010101')
    >>> right = Bits(bin='00110011')
    >>> ((mask_ ^ left_) & (mask_ | left_) & ~(mask_ & right)).bin
    '0b01011000'

    >>> bits = Bits()
    >>> bits.bin
    ''
    >>> bits.append(0)
    >>> bits.bin
    '0b0'
    >>> bits.append(1)
    >>> bits.append('1')
    >>> bits.append(False)
    >>> bits.bin
    '0b0110'

    >>> bits.extend(Bits(16))
    >>> bits.bin
    '0b01101000'

    >>> bits.extend(255, length=8)
    >>> bits.bin
    '0b01101000 0b11111111'

    >>> bits[0]
    False
    >>> bits[1]
    True
    >>> bits[4:]
    '0b10001111 0b1111'

    TODO: Bits repr

    """

    __slots__ = ["_bytes", "_last_byte", "_len_last_byte", "_len"]
    _ZEROS = {0, "0", False}
    _ONES = {1, "1", True}

    # TODO: mangle these
    # Contains whole 8 bit bytes
    _bytes: bytearray
    # Contains the trailing incomplete 8 bits
    _last_byte: int
    _len_last_byte: int
    _len: int

    # TODO: handle bytes objects
    # TODO: return bytes objects (implement __bytes__)

    def __init__(
        self,
        /,
        length: int = None,
        int_: SupportsInt = None,
        bin_: str = None,
        hex_: str = None,
        bool_: bool = None,
        bytes_: ByteString = None,
        # TODO: add IOStream like objects?
    ):
        self._bytes = bytearray()
        self._len_last_byte = 0
        self._last_byte = 0
        self._len = 0
        if sum(kw_arg is not None for kw_arg in (int_, bin_, hex_, bool_, bytes_)) > 1:
            raise ValueError("one and only one of int_, bin_, hex_, bool_, bytes_ must be specified")
        if (length and not int_) or (int_ and not length > 0):
            raise ValueError("a positive length must be specified with an integer")
        if int_:
            self.extend(int(int_), int(length))
            return
        if bin_:
            clean_bin = bin_.replace("0b", "").replace("_", "")
            for num in clean_bin.split(" "):
                self.extend(num, len(num))
            return
        if hex_:
            clean_hex = hex_.replace("0x", "").replace("_", "")
            for num in clean_hex.split():
                self.extend(int(num, 16), len(num) * 4)
            return
        if bytes_:
            self._bytes = bytearray(bytes_)
            self._len = len(self._bytes)

        # TODO: INCOMPLETE

    def __repr__(self):
        if len(self) <= 64:
            return f'Bits(bin_="{format(int(self), f"#0{self._len + 2}b")}")'
        if self._decimal_digits() <= 64:
            return f'Bits(int_="{format(int(self), f"0{self._decimal_digits()}d")}", length={self._len})'
        if len(self) // 8 <= 64:
            return f'Bits(hex_="{format(int(self), f"#0{len(self) // 8 + 2}x")}")'
        return f'Bits(bin_="{self[:24].bin} ... {self[-(self._len_last_byte + 16):].bin}")'

    def _decimal_digits(self):
        """
        The minimum decimal digits it would take to represent the same amount of information.

        :return: TODO: DOCS
        """
        largest_possible = (1 << self._len) - 1
        return int(log(largest_possible, 10)) + 1

    def __str__(self):
        # TODO: STUB
        pass

    def __bytes__(self) -> Generator[int, None, None]:
        return self.byte_generator

    def byte_generator(self) -> Generator[int, None, None]:
        """
        TODO: DOCS

        :return:
        """
        for byte in self._bytes:
            yield byte
        yield self._last_byte << 8 - self._len_last_byte

    def __bool__(self):
        return bool(self._bytes)

    # Mutable Sequence Methods

    def insert(self, index: int, value: ValidBits) -> None:
        if value in self._ONES:
            value = True
        elif value in self._ZEROS:
            value = False
        else:
            raise TypeError(f"could not determine single bit value for {value}")
        if len(self) < index or index < -len(self) - 1:
            raise IndexError

        # The modulo corrects for negative index values.
        if index < 0:
            if index == -(len(self) + 1):
                index = 0
            else:
                index = index % len(self)

        # If appending to the end.
        if index == len(self):
            self.append_bit(value)
            return

        # The byte the index is within.
        byte_index = index // 8
        # The index of the bit within the correct byte.
        bit_index = index % 8

        # If only modifying the last (incomplete) byte.
        if byte_index == len(self._bytes):
            self._last_byte = self._insert_bit_in_byte(self._last_byte, self._len_last_byte, bit_index, value)
            self._increment_last_byte()
            return

        # Inserting anywhere else.
        # Copy all bytes left of the index.
        new_bytes = self._bytes[:byte_index]
        # Insert the bit and remove the rightmost to carry over into the next byte to the right.
        next_byte = self._insert_bit_in_byte(self._bytes[byte_index], 8, bit_index, value)
        carry = next_byte & 1
        next_byte >>= 1
        # Append the byte with the carry over bit removed.
        new_bytes.append(next_byte)
        # Repeat for the remaining whole bytes to the right of the index.
        for i in range(byte_index + 1, len(self._bytes)):
            next_byte = (carry << 8) | self._bytes[i]
            carry = next_byte & 1
            next_byte >>= 1
            new_bytes.append(next_byte)
        # Replace the old bytes with the new.
        self._bytes = new_bytes
        # Append the last carry bit to the last (incomplete) byte, and increment it's length.
        self._last_byte = (carry << self._len_last_byte) | self._last_byte
        self._increment_last_byte()
        return

    def extend(self, values: Union[Iterable[ValidBits], int], length: int = None) -> None:
        if isinstance(values, int):
            if not length > 0:
                raise ValueError("length (bit length) greater than 0 must be provided for int values")
            for i in range(length - 1, -1, -1):
                self.append(bool((1 << i) & values))
            return
        return super().extend(values)

    def _increment_last_byte(self) -> None:
        """
        Call when a digit (one bit) has been added anywhere in the last (incomplete) byte

        >>> bits = Bits(int_=0b1000, length=4)
        >>> bits._len_last_byte
        4

        # The last increases in length even if a 0 is inserted at the begginging
        >>> bits.append(1)
        >>> bits._len_last_byte
        5
        >>> bits.insert(0, 0)
        >>> bits._len_last_byte
        6
        >>> bits.bin
        '0b01_0001'

        >>> bits = Bits(int_=0b1111_1111, length=8)
        >>> bits.append(1)

        """
        self._len_last_byte += 1
        self._len += 1
        if self._len_last_byte == 8:
            self._bytes.append(self._last_byte)
            self._last_byte = 0
            self._len_last_byte = 0

    @staticmethod
    def _insert_bit_in_byte(byte: int, length: int, index: int, value: bool) -> int:
        """
        TODO: DOCS

        :param byte:
        :param length:
        :param index:
        :param value:
        :return:
        """
        right_index = length - index
        left_bits = (((byte >> right_index) << 1) | value) << right_index
        right_bits = byte & ((1 << right_index) - 1)
        return left_bits | right_bits

    def append_bit(self, bit: bool) -> None:
        """
        TODO: DOCS

        :param bit:
        :return:
        """
        self._last_byte = (self._last_byte << 1) | bit
        self._increment_last_byte()

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> bool:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[bool]:
        ...

    def __getitem__(self, index):
        """
        TODO: DOCS

        :param index:
        :return:
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))

            # If the case is simple.
            if step == 1 and start == 0:
                # The byte the stop is within.
                byte_index = stop // 8
                # The stop bit within the correct byte.
                bit_index = stop % 8
                bits = type(self)(bytes_=self._bytes[:stop])
                # Append remaining bits.
                for i in range(byte_index, byte_index + bit_index, 1):
                    bits.append_bit(self.get_bit(i))

            # For all other cases (not particularly efficient).
            bits = type(self)()
            for i in range(start, stop, step):
                bits.append_bit(self.get_bit(i))
            return bits

        # For the singular case.
        return self.get_bit(int(index))

    def get_bit(self, index: int) -> bool:
        if len(self) <= index or index < -len(self):
            raise IndexError

        # Modulo corrects negative indices
        if index < 0:
            index = index % len(self)

        # The byte the index is within.
        byte_index = index // 8
        # The index of the bit within the correct byte.
        bit_index = index % 8

        # If the index is in the last (incomplete) byte.
        if byte_index == len(self._bytes):
            return self._bit_from_byte(self._last_byte, self._len_last_byte, bit_index)

        # If the index is anywhere else.
        return self._bit_from_byte(self._bytes[byte_index], 8, bit_index)

    @staticmethod
    def _bit_from_byte(byte: int, length: int, index: int) -> bool:
        """
        TODO: DOCS

        :param index:
        :param byte:
        :param length:
        :return:
        """
        right_index = length - index - 1
        return bool((1 << right_index) & byte)

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: bool) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[ValidBits]) -> None:
        ...

    def __setitem__(self, index, other):
        """
        TODO: STUB

        :param index:
        :param other:
        :return:
        """
        pass
        # if isinstance(index, slice):
        #     start, stop, step = index.indices(self.__len)
        #     range_ = range(start, stop, step)
        #
        #     for i, new_val in zip(range_, other):
        #         if new_val in self._ONES:
        #             self._bit_on(i)
        #             continue
        #         if new_val in self._ZEROS:
        #             self._bit_off(i)
        #             continue
        #         raise TypeError(f"Could not determine single bit value for {new_val}.")
        #     return
        #
        # if self.__len > index >= -self.__len:
        #     index = index % self.__len
        #     if other in self._ONES:
        #         self._bit_on(index)
        #         return
        #     if other in self._ZEROS:
        #         self._bit_off(index)
        #         return
        #     raise TypeError(f"Could not determine single bit value for {other}.")
        # raise IndexError

    def set_bit(self, index) -> None:
        # TODO: STUB
        # if len(self) <= index or index < -len(self):
        #     raise IndexError
        #
        # # Modulo corrects negative indices
        # if index < 0:
        #     index = index % len(self)
        #
        # # The byte the index is within.
        # byte_index = index // 8
        # # The index of the bit within the correct byte.
        # bit_index = index % 8
        #
        # # If the index is in the last (incomplete) byte.
        # if byte_index == len(self._bytes):
        #     return self._bit_from_byte(self._last_byte, self._len_last_byte, bit_index)
        #
        # # If the index is anywhere else.
        # return self._bit_from_byte(self._bytes[byte_index], 8, bit_index)

    @staticmethod
    def _set_bit_in_byte(byte: int, length: int, index: int, value: int) -> int:
        """
        TODO: DOCS

        :param index:
        :param byte:
        :param length:
        :return:
        """
        # right_index = length - index - 1
        # return (1 << right_index) | byte if value else
        # TODO: STUB

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None:
        ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None:
        ...

    def __delitem__(self, index):
        # TODO: STUB
        pass
        # if isinstance(index, slice):
        #     start, stop, step = index.indices(self.__len)
        #     if step == 1:
        #         # del self[2:6]  # len(self) = 8
        #         #     v v v v
        #         # 0 1 2 3 4 5 6 7
        #         left_bits = self.__data >> self.__len - start
        #         left_bits = left_bits << self.__len - stop
        #         right_bits = self.__data & ((1 << self.__len - stop) - 1)
        #         self.__data = left_bits | right_bits
        #         self.__len -= stop - start
        #         return
        #     for i in reversed(sorted(range(start, stop, step))):
        #         self._delitem_at(i)
        #     return
        # self._delitem_at(index)

    @staticmethod
    def _del_bit_from_byte(byte: int, length: int, index: int) -> int:
        """
        TODO: DOCS

        :param byte:
        :param length:
        :param index:
        :param value:
        :return:
        """
        # right_index = length - index
        # left_bits = (((byte >> right_index) << 1) | value) << right_index
        # right_bits = byte & ((1 << right_index) - 1)
        # return left_bits | right_bits
        # TODO: STUB

    def __len__(self) -> int:
        return self._len

    # END Mutable Sequence Methods

    def __lt__(self, other: SupportsInt) -> bool:
        # TODO: STUB
        pass

    def __le__(self, other: SupportsInt) -> bool:
        # TODO: STUB
        pass

    def __eq__(self, other: SupportsInt) -> bool:
        # TODO: STUB
        pass

    def __ne__(self, other: SupportsInt) -> bool:
        # TODO: STUB
        pass

    def __gt__(self, other: SupportsInt) -> bool:
        # TODO: STUB
        pass

    def __ge__(self, other: SupportsInt) -> bool:
        # TODO: STUB
        pass

    def __add__(self, other: Union[Iterable[ValidBits], int]) -> "Bits":
        """
        This is NOT addition, this is concatenation.

        :param other: Other object to be concatenated.
        :return: New Bits object that is a concatenation of the inputs.
        """
        # TODO: STUB
        pass
        # if isinstance(other, Bits):
        #     return type(self)((self.__data << other.__len) | other.__data, length=self.__len + other.__len)
        # return NotImplemented

    def __lshift__(self, other: int) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass
        # if isinstance(other, int):
        #     return type(self)(self.__data << other, length=self.__len + other)
        # return NotImplemented

    def __rshift__(self, other: int) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass
        # if isinstance(other, int):
        #     return type(self)(self.__data >> other, length=self.__len - other)
        # return NotImplemented

    def __and__(self, other: Iterable[ValidBits]) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass
        # if isinstance(other, Bits):
        #     if self.__len == other.__len:
        #         return type(self)(self.__data & other.__data, length=self.__len)
        #     longest_len = max(self.__len, other.__len)
        #     self_data = self.__data << longest_len - self.__len
        #     other_data = other.__data << longest_len - other.__len
        #     return type(self)(self_data & other_data, length=longest_len)
        # return NotImplemented

    def __xor__(self, other: Iterable[ValidBits]) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass
        # if isinstance(other, Bits):
        #     if self.__len == other.__len:
        #         return type(self)(self.__data ^ other.__data, length=self.__len)
        #     longest_len = max(self.__len, other.__len)
        #     self_data = self.__data << longest_len - self.__len
        #     other_data = other.__data << longest_len - other.__len
        #     return type(self)(self_data ^ other_data, length=longest_len)
        # return NotImplemented

    def __or__(self, other: Iterable[ValidBits]) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass
        # if isinstance(other, Bits):
        #     if self.__len == other.__len:
        #         return type(self)(self.__data | other.__data, length=self.__len)
        #     longest_len = max(self.__len, other.__len)
        #     self_data = self.__data << longest_len - self.__len
        #     other_data = other.__data << longest_len - other.__len
        #     return type(self)(self_data | other_data, length=longest_len)
        # return NotImplemented

    def __iadd__(self, other: Iterable[ValidBits]) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass

    def __ilshift__(self, other: Iterable[ValidBits]) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass

    def __irshift__(self, other: Iterable[ValidBits]) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass

    def __iand__(self, other: Iterable[ValidBits]) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass

    def __ixor__(self, other: Iterable[ValidBits]) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass

    def __ior__(self, other: Iterable[ValidBits]) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass

    def __invert__(self) -> MutableSequence[ValidBits]:
        # TODO: STUB
        pass
        # return type(self)(self.__data ^ ((1 << self.__len) - 1), length=self._len)

    def __int__(self) -> int:
        return (int.from_bytes(self._bytes, "big") << self._len_last_byte) | self._last_byte

    def __index__(self) -> int:
        # TODO: STUB
        pass
        # return self.__data

    @property
    def hex(self) -> str:
        """
        TODO: DOCS

        :return:
        """
        if len(self) <= 4096:
            ret_str = " ".join(format(byte, "#04x") for byte in self._bytes)
            if self._len_last_byte > 0:
                # No grouping spacer for binary number with less than 4 digits
                ret_str += " " + format(self._last_byte, f"#0{3 + self._len_last_byte // 8}x")
            return ret_str
        else:
            return "Output too long... (greater than 16 bytes)"

    @property
    def bin(self) -> str:
        """
        Returns a string with the binary representation of each byte, up to 16 bytes.
        For larger strings, retrieve the raw bytes with::

            bytes(Bits())

        :return: A string binary representation of the Bits object

        >>> Bits(int_=255, length=8).bin
        '0b1111_1111'
        """
        if len(self) <= 128:
            ret_str = " ".join(format(byte, "#011_b") for byte in self._bytes)
            if self._len_last_byte > 4:
                # Add extra width to allow for grouping spacer
                ret_str += " " + format(self._last_byte, f"#0{3 + self._len_last_byte}_b")
            elif self._len_last_byte > 0:
                # No grouping spacer for binary number with less than 4 digits
                ret_str += " " + format(self._last_byte, f"#0{2 + self._len_last_byte}b")
            return ret_str
        else:
            return "Output too long... use bytes(Bits()) or int(Bits())"


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
