"""Utility functions."""
from abc import abstractmethod
from collections.abc import MutableSequence
from math import log
from typing import (
    ByteString,
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
ValidBit = Union[bool, Literal[1], Literal[0], Literal["1"], Literal["0"]]


class ConcatenableSequence(Protocol[T_co]):
    """
    Any Sequence T where +(:T, :T) -> T.
    Types must support indexing and concatenation.

    >>> def test(a: CS, b: CS) -> CS: return a + b
    >>> a = test(Bits(bin_="1111"), Bits(bin_="1111"))
    >>> x = test('abc', 'def') # passes type check
    >>> y = test(tuple(1, 2, 3), tuple(4, 5)) # passes type check
    >>> z = test(list(1, 2), list(3, 4)) # passes type check

    """

    def __add__(self, other: CS) -> CS:
        ...

    def __getitem__(self, index: int) -> T_co:
        ...

    def __len__(self) -> int:
        ...


# NOTE Finland uses iso8859_10 encoding.
# NOTE The proper bitwise operation for a NOR mask is (mask_ ^ left_) & (left_ | mask_) & ~(right & mask_).


def bin_ones(count: int) -> int:
    """
    Create an integer from successive binary ones.

    >>> bin(bin_ones(7))
    '0b1111111'

    :param count: The number of successive ones.
    :return: The integer.
    """
    return 1 << count - 1


class Bits(MutableSequence[ValidBit]):
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
    >>> format(int(a), '#042b')
    '0b0101010001001111001000000100111100001010'

    Nor mask:
    >>> mask_ = Bits(bin='00001111')
    >>> left_ = Bits(bin='01010101')
    >>> right = Bits(bin='00110011')
    >>> ((mask_ ^ left_) & (mask_ | left_) & ~(mask_ & right)).bin
    '0b01011000'

    """

    # TODO: Consider removing the slots (can we anticipate use cases that require many Bits objects?)
    __slots__ = ["_bytes", "_last_byte", "_len_last_byte", "_len"]

    # TODO: mangle these
    # Contains whole 8 bit bytes
    _bytes: bytearray
    # Contains the trailing incomplete 8 bits
    _last_byte: int
    _len_last_byte: int
    _len: int

    # TODO: handle bytes objects

    def __init__(
        self,
        /,
        length: int = None,
        int_: SupportsInt = None,
        bin_: str = None,
        hex_: str = None,
        bool_: Iterable[ValidBit] = None,
        bytes_: ByteString = None,
    ):
        self._bytes = bytearray()
        self._len_last_byte = 0
        self._last_byte = 0
        self._len = 0
        if sum(kw_arg is not None for kw_arg in (int_, bin_, hex_, bool_, bytes_)) > 1:
            raise ValueError("one and only one of int_, bin_, hex_, bool_, bytes_ must be specified")
        if (length and not int_) or (int_ and not length > 0):
            # Rather than default to len(int_) the length must be provided to
            # reduce unexpected behavior and errors in implementation.
            raise ValueError("a positive length must be specified with an integer")
        if int_:
            self.extend(int(int_), int(length))
        elif bin_:
            clean_bin = bin_.replace("0b", "").replace("_", "")
            for num in clean_bin.split(" "):
                self.extend(num, len(num))
        elif hex_:
            clean_hex = hex_.replace("0x", "").replace("_", "").replace(" ", "")
            for num in clean_hex:
                self.extend(int(num, 16), 4)
        elif bytes_:
            self._bytes = bytearray(bytes_)
            self._len = len(self._bytes)
        # TODO: INCOMPLETE

    def copy(self):
        return type(self)(int_=int(self), length=self._len)

    def __repr__(self):
        """
        TODO: DOCS
        :return:
        """
        if len(self) <= 64:
            return f'Bits(bin_="{format(int(self), f"#0{self._len + 2}b")}")'

        if self._decimal_digits() <= 64:
            return f'Bits(int_="{format(int(self), f"0{self._decimal_digits()}d")}", length={self._len})'

        if len(self) // 8 <= 64:
            return f'Bits(hex_="{format(int(self), f"#0{len(self) // 8 + 2}x")}")'
        if self._len_last_byte:
            return f'Bits(bin_="{self[:24].bin} ... {self[-(self._len_last_byte + 16):].bin}")'
        return f'Bits(bin_="{self[:24].bin} ... {self[-24:].bin}")'

    def _decimal_digits(self):
        """
        The minimum decimal digits it would take to represent the same amount of information.

        >>> Bits(int_=)

        :return:
        """
        largest_possible = (1 << self._len) - 1
        return int(log(largest_possible, 10)) + 1

    def __bytes__(self) -> Iterable[int]:
        return bytes(self.byte_gen())

    def decode(self, *args, **kwargs):
        """
        A convenience wrapper for `bytes().decode()`
        Decode the bytes using the codec registered for encoding.

          encoding
            The encoding with which to decode the bytes.
          errors
            The error handling scheme to use for the handling of decoding errors.
            The default is 'strict' meaning that decoding errors raise a
            UnicodeDecodeError. Other possible values are 'ignore' and 'replace'
            as well as any other name registered with codecs.register_error that
            can handle UnicodeDecodeErrors.
        """
        return bytes(self).decode(*args, **kwargs)

    def byte_gen(self, index: int = 0) -> Iterable[int]:
        """
        TODO: DOCS

        :return:
        """
        for i in range(index, len(self._bytes)):
            yield self._bytes[i]
        if self._len_last_byte:
            yield self._last_byte << 8 - self._len_last_byte

    def _byte_and_len_gen(self, index: int = 0) -> Iterable[Tuple[int, int]]:
        """
        TODO: DOCS
        yields bytes with length

        :return:
        """
        for i in range(index, len(self._bytes)):
            yield self._bytes[i], 8
        if self._len_last_byte:
            yield self._last_byte << 8 - self._len_last_byte, self._len_last_byte

    def __bool__(self):
        return bool(self._bytes) or bool(self._last_byte)

    def _byte_bit_indices(self, index: int) -> Tuple[int, int, int]:
        """
        TODO: DOCS

        :param index:
        :return:
        """
        if len(self) <= index or index < -len(self):
            raise IndexError

        # Modulo corrects negative indices
        if index < 0:
            index = index % len(self)

        # The first is the index of the byte that the index is within.
        # The second is the index of the bit within the byte (counting from the left).
        return index // 8, index % 8, index

    @staticmethod
    def _validate_bit(value: ValidBit):
        """
        Validates a value as a ValidBit returns a bool representation.

        :param value: The value to check.
        :return: The bool representation.
        """
        zeros = {0, "0", False}
        ones = {1, "1", True}

        if value in ones:
            value = True
        elif value in zeros:
            value = False
        else:
            raise TypeError(f"could not determine single bit value for {value}")
        return value

    # Mutable Sequence Methods

    def insert(self, index: int, value: ValidBit) -> None:
        """
        TODO: DOCS

        :param index:
        :param value:
        :return:
        """
        value = self._validate_bit(value)
        # If the index is above the length, set it to the length.
        # If the index is below the negative length, set it to the negative length.
        # Then if the new index is negative, take the modulo so that -1 accesses the last element.
        if len(self) == 0:
            index = 0
        else:
            index = min(len(self), index) if index >= 0 else max(-len(self), index) % len(self)
        byte_index, bit_index = index // 8, index % 8

        # If appending to the end.
        if index == len(self):
            self.append_bit(value)

        # If only modifying the last (incomplete) byte.
        elif byte_index == len(self._bytes):
            self._last_byte = self._insert_bit_in_byte(self._last_byte, self._len_last_byte, bit_index, value)
            self._increment_last_byte()

        # Inserting anywhere else.
        else:
            # Insert the bit then remove the rightmost bit to carry over into the next byte to the right.
            new_byte = self._insert_bit_in_byte(self._bytes[byte_index], 8, bit_index, value)
            carry = new_byte & 1
            new_byte >>= 1
            # Append the byte with the carry over bit removed.
            self._bytes[byte_index] = new_byte
            # Repeat for the remaining whole bytes to the right of the index.
            for i in range(byte_index + 1, len(self._bytes)):
                new_byte = (carry << 8) | self._bytes[i]
                carry = new_byte & 1
                new_byte >>= 1
                self._bytes[i] = new_byte
            # Append the last carry bit to the last (incomplete) byte, and increment it's length.
            self._last_byte = (carry << self._len_last_byte) | self._last_byte
            self._increment_last_byte()

    def extend(self, values: Union[Iterable[ValidBit], int], length: int = None) -> None:
        """
        TODO: DOCS
        This is to allow extending by and integer given a bit length.

        :param values:
        :param length:
        :return:
        """
        if isinstance(values, int):
            if not length > 0:
                raise ValueError("length (number of bits) must be greater than 0")
            for i in range(length - 1, -1, -1):
                self.append(bool((1 << i) & values))

        else:
            super().extend(values)

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
        right_bits = byte & bin_ones(right_index)
        return left_bits | right_bits

    def append_bit(self, value: ValidBit) -> None:
        """
        Append a bit to the end if value is a ValidBit

        :param value:
        :return:
        """
        value = self._validate_bit(value)
        self._last_byte = (self._last_byte << 1) | value
        self._increment_last_byte()

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

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> bool:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> "Bits":
        ...

    def __getitem__(self, index):
        """
        TODO: DOCS

        :param index:
        :return:
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))

            # For the case where the slice starts from the same origin.
            if step == 1 and start == 0:
                byte_index, bit_index, index = self._byte_bit_indices(stop)
                bits = type(self)(bytes_=self._bytes[:byte_index])
                # Append remaining bits.
                for i in range(byte_index, byte_index + bit_index):
                    bits.append_bit(self.get_bit(i))
                return bits

            # For all other cases (not particularly efficient).
            bits = type(self)()
            for i in range(start, stop, step):
                bits.append_bit(self.get_bit(i))
            return bits

        # For the singular case.
        return self.get_bit(index)

    def get_bit(self, index: int) -> bool:
        """
        TODO: DOCS
        :param index:
        :return:
        """
        byte_index, bit_index, index = self._byte_bit_indices(index)

        # If the index is in the last (incomplete) byte.
        if byte_index == len(self._bytes):
            return self._get_bit_from_byte(self._last_byte, self._len_last_byte, bit_index)

        # If the index is anywhere else.
        return self._get_bit_from_byte(self._bytes[byte_index], 8, bit_index)

    @staticmethod
    def _get_bit_from_byte(byte: int, length: int, index: int) -> bool:
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
    def __setitem__(self, i: int, o: ValidBit) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[ValidBit]) -> None:
        ...

    def __setitem__(self, index, other):
        """
        TODO: STUB

        :param index:
        :param other:
        :return:
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))

            # TODO: IS THIS COMPLETE ENOUGH?
            other_bit = iter(other)
            for i in range(start, stop, step):
                self.set_bit(i, next(other_bit))

        # For the singular case
        else:
            self.set_bit(index, other)

    def set_bit(self, index: int, value: ValidBit) -> None:
        """
        TODO: STUB

        :param index:
        :param value:
        :return:
        """
        value = self._validate_bit(value)
        byte_index, bit_index, index = self._byte_bit_indices(index)

        # If the index is in the last (incomplete) byte.
        if byte_index == len(self._bytes):
            self._last_byte = self._set_bit_in_byte(self._last_byte, self._len_last_byte, bit_index, value)

        # If the index is anywhere else.
        else:
            self._bytes[byte_index] = self._set_bit_in_byte(self._bytes[byte_index], 8, bit_index, value)

    @classmethod
    def _set_bit_in_byte(cls, byte: int, length: int, index: int, value: bool) -> int:
        """
        TODO: DOCS

        :param index:
        :param byte:
        :param length:
        :return:
        """
        # Setting a bit in the last (incomplete) byte
        if cls._get_bit_from_byte(byte, length, index) == value:
            return byte

        # All other cases
        right_index = length - index - 1
        return (1 << right_index) ^ byte

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None:
        ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None:
        ...

    def __delitem__(self, index):
        """
        Removes a bit or a slice.

        TODO: DOCS

        :param index:
        :return:
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))

            # ***VERY inefficient***
            # Always goes in reverse order to not mess up the indexing.
            removal_indices = sorted(list(range(start, stop, step)), reverse=True)
            for i in removal_indices:
                self.del_bit(i)

        else:
            self.del_bit(index)

    def del_bit(self, index: int) -> None:
        """
        Remove a bit.

        >>> bits = Bits(bin_="1001 0000 1")
        >>> bits.del_bit(3)
        >>> bits.bin
        '0b1000_0001'
        >>> bits.del_bit(0)
        >>> bits.bin
        '0b000_0001'
        >>> bits.del_bit(-1)
        >>> bits.bin
        '0b00_0000'

        :param index: Index of the bit to remove.
        """
        byte_index, bit_index, index = self._byte_bit_indices(index)

        # The case of deleting a bit from the last (incomplete) byte.
        if byte_index == len(self._bytes):
            self._last_byte = self._del_bit_from_byte(self._last_byte, self._len_last_byte, bit_index)
            # We know there was at least one bit in the last (incomplete) byte
            # because for a single byte with no last byte the max index is 7 and 7 // 8 = 0 (byte_index)
            # so this logic could not be reached.
            self._decrement_last_byte()

        # All other cases.
        else:
            # Remove the bit from the target byte, then append the first bit from the next byte.
            # Cascade the change through the list of bytes.
            new_byte = self._del_bit_from_byte(self._bytes[byte_index], 8, bit_index)
            for i in range(byte_index + 1, len(self._bytes)):
                first_bit = bool(self._bytes[i] & 0b1000_0000)
                self._bytes[i - 1] = (new_byte << 1) | first_bit
                new_byte = self._bytes[i] & 0b0111_1111

            # Append the first bit from the last (incomplete) byte.
            if self._len_last_byte:
                first_bit = bool(self._last_byte & bin_ones(self._len_last_byte))
                self._bytes[-1] = (new_byte << 1) | first_bit
                # Truncate the first bit of the last (incomplete) byte.
                self._last_byte = self._last_byte & (bin_ones(self._len_last_byte) - 1)

            # If the last (incomplete) byte is empty, remove the last full byte.
            else:
                self._bytes.pop()
                # The former last full byte becomes the last (incomplete) byte with it's first bit was removed.
                self._last_byte = new_byte

            # Decrement the length and last (incomplete) byte length in both cases.
            self._decrement_last_byte()

    @staticmethod
    def _del_bit_from_byte(byte: int, length: int, index: int) -> int:
        """
        Remove a bit from a byte.

        >>> Bits._del_bit_from_byte(0b00010000, 8, 3)
        0

        :param byte: Byte from which to remove a bit.
        :param length: Length of the byte.
        :param index: Index of the bit to remove.
        :return: The Byte with bit removed.
        """
        right_index = length - index
        left_bits = (byte >> right_index) << right_index - 1
        right_bits = byte & (bin_ones(right_index) - 1)
        return left_bits | right_bits

    def _decrement_last_byte(self) -> None:
        """
        Called when a digit (one bit) has been removed anywhere in the last (incomplete) byte.

        >>> bits = Bits(int_=0b010001000, length=9)
        >>> bits._len_last_byte
        1

        # The last decreases in length even if a 0 is removed from the begginging.
        >>> del bits[0]
        >>> bits._len_last_byte
        0

        >>> bits.append_bit(0)
        >>> bits._len_last_byte
        1

        >>> del bits[-1]
        >>> bits._len_last_byte
        0

        >>> del bits[-1]
        >>> bits._len_last_byte
        7
        """
        self._len_last_byte -= 1
        self._len -= 1
        if self._len_last_byte < 0:
            self._len_last_byte = 7

    def __len__(self) -> int:
        return self._len

    # END Mutable Sequence Methods

    def __lt__(self, other: SupportsInt) -> bool:
        return int(self) < int(other)

    def __le__(self, other: SupportsInt) -> bool:
        return int(self) <= int(other)

    def __eq__(self, other: SupportsInt) -> bool:
        return int(self) == int(other)

    def __ne__(self, other: SupportsInt) -> bool:
        return int(self) != int(other)

    def __gt__(self, other: SupportsInt) -> bool:
        return int(self) > int(other)

    def __ge__(self, other: SupportsInt) -> bool:
        return int(self) >= int(other)

    def __add__(self, other: Iterable[ValidBit]) -> "Bits":
        """
        This is concatenation, NOT addition.

        >>> (Bits(bin_="0110") + Bits(bin_="1001")).bin
        '0b0110_1001'
        >>> (Bits(bin_="0110") + "1001").bin
        '0b0110_1001'

        :param other: Other object to be concatenated.
        :return: New Bits object that is a concatenation of the inputs.
        """
        new = self.copy()
        new.extend(other)
        return new

    def __lshift__(self, index: int) -> "Bits":
        """
        Left shift the bits.

        >>> (Bits(bin_="1111") << 4).bin
        '0b1111_0000'

        :param index: Number of places to shift
        :return: Shifted Bits object
        """
        new = self.copy()
        new.extend(type(self)(int_=0, length=index))
        return new

    def __rshift__(self, index: int) -> "Bits":
        """
        Right shift the bits.

        >>> (Bits(bin_="11110000") >> 4).bin
        '0b1111'

        :param index: Number of places to shift
        :return: Shifted Bits object
        """
        new = type(self)(int_=0, length=index)
        new.extend(self[:-index])
        return new

    def __and__(self, other: Iterable[ValidBit]) -> "Bits":
        """
        Bitwise and operation.

        >>> (Bits(bin_='01111000') & Bits(bin_='00011110')).bin
        '0b0001_1000'
        >>> (Bits(bin_='0111') & Bits(bin_='00011110')).bin
        '0b0001'
        >>> (Bits(bin_="1110") & "0111").bin
        '0b0110'

        :param other: Other Bits to 'and' with
        :return: Combined Bits objects
        """
        new = type(self)()
        for self_bit, other_bit in zip(self, other):
            new.append(self_bit & self._validate_bit(other_bit))
        return new

    def __xor__(self, other: Iterable[ValidBit]) -> "Bits":
        """
        Bitwise xor operation.

        >>> (Bits(bin_='01111000') ^ Bits(bin_='00011110')).bin
        '0b0110_0110
        >>> (Bits(bin_='01111000') ^ Bits(bin_='00011110')).bin
        '0b0110
        >>> (Bits(bin_="1110") ^ "0111").bin
        '0b1001'

        :param other: Other Bits to 'xor' with
        :return: Combined Bits objects
        """
        new = type(self)()
        for self_bit, other_bit in zip(self, other):
            new.append(self_bit ^ self._validate_bit(other_bit))
        return new

    def __or__(self, other: Iterable[ValidBit]) -> "Bits":
        """
        Bitwise or operation.

        >>> (Bits(bin_='01111000') | Bits(bin_='00011110')).bin
        '0b0111_1110'
        >>> (Bits(bin_='01111000') | Bits(bin_='00011110')).bin
        '0b0111'
        >>> (Bits(bin_="1100") | "0011").bin
        '0b1111'

        :param other: Other Bits to 'or' with
        :return: Combined Bits objects
        """
        new = type(self)()
        for self_bit, other_bit in zip(self, other):
            new.append(self_bit | self._validate_bit(other_bit))
        return new

    def __iadd__(self, other: Iterable[ValidBit]) -> "Bits":
        """
        Extend in-place.

        >>> bits = Bits(bin_="1111")
        >>> bits += "0000"
        >>> bits.bin
        '0b1111_0000'

        :param other: Bits to extend.
        :return: The Bits object that was modified in place.
        """
        self.extend(other)
        return self

    def __ilshift__(self, index: int) -> "Bits":
        """
        Left bitshift in-place.

        >>> bits = Bits(bin_="1111")
        >>> bits <<= 4
        >>> bits.bin
        '0b1111_0000'

        :param index: Number of places to shift.
        :return: The Bits object that was modified in place.
        """
        self.extend(0, index)
        return self

    def __irshift__(self, index: int) -> "Bits":
        """
        Right bitshift in-place.

        >>> bits = Bits(bin_="1111 1111")
        >>> bits >>= 4
        >>> bits.bin
        '0b1111'

        :param index: Number of places to shift.
        :return: The Bits object that was modified in place.
        """
        if index:
            del self[-index:]
        return self

    def __iand__(self, other: Iterable[ValidBit]) -> "Bits":
        """
        Bitwise 'and' with other bits; in-place.

        >>> bits_ = Bits(bin_="1110")
        >>> bits_ &= "0111"
        >>> bits_.bin
        '0b0110'

        :param other: The Iterable bits to 'and' with.
        :return: The Bits object that was modified in place.
        """
        index = 0
        for index, bits in enumerate(zip(self, other)):
            self[index] = bits[0] & self._validate_bit(bits[1])
        if len(self) > index + 1:
            del self[-(index + 1) :]
        return self

    def __ixor__(self, other: Iterable[ValidBit]) -> "Bits":
        """
        Bitwise 'xor' with other bits; in-place.

        >>> bits_ = Bits(bin_="0110")
        >>> bits_ ^= "0101"
        >>> bits_.bin
        '0b0011'

        :param other: The Iterable bits to 'xor' with.
        :return: The Bits object that was modified in place.
        """
        index = 0
        for index, bits in enumerate(zip(self, other)):
            self[index] = bits[0] ^ self._validate_bit(bits[1])
        if len(self) > index + 1:
            del self[-(index + 1) :]
        return self

    def __ior__(self, other: Iterable[ValidBit]) -> "Bits":
        """
        Bitwise 'or' with other bits; in-place.

        >>> bits_ = Bits(bin_="1100")
        >>> bits_ |= "0011"
        >>> bits_.bin
        '0b1111'

        :param other: The Iterable bits to 'or' with.
        :return: The Bits object that was modified in place.
        """
        index = 0
        for index, bits in enumerate(zip(self, other)):
            self[index] = bits[0] | self._validate_bit(bits[1])
        if len(self) > index + 1:
            del self[-(index + 1) :]
        return self

    def __invert__(self) -> "Bits":
        """
        Return a Bits object with each bit inverted.

        >>> (~Bits(bin_='01001110')).bin
        '0b1011_0001'

        :return: The Bits object with inverted bits.
        """
        new = type(self)()
        new._bytes = bytearray(byte ^ 0b11111111 for byte in self._bytes)
        new._last_byte = self._last_byte ^ bin_ones(self._len_last_byte)
        new._len = self._len
        return new

    def __int__(self) -> int:
        """
        Represent the sequence of bits as an int.

        >>> int(Bits(hex_="0xff"))
        255
        >>> int(Bits(hex_="0xfff"))
        4095
        >>> int(Bits(hex_="0xffff"))
        65535

        :return: The int representation.
        """
        return (int.from_bytes(self._bytes, "big") << self._len_last_byte) | self._last_byte

    @property
    def hex(self) -> str:
        """
        Returns a string with hexadecimal representation of each byte.

        >>> Bits(bin_="0b1111_1111").bin
        '0xff'
        >>> Bits(bin_="0b1111_1111 0b1111")
        '0xff 0xf'
        >>> Bits(bin_="0b1111_1111 0b1111_1111").bin
        '0xff 0xff'

        :return: The string representation of the bytes as hexadecimal.
        """
        ret_str = " ".join(format(byte, "#04x") for byte in self._bytes)
        if self._len_last_byte > 0:
            ret_str += " " + format(self._last_byte, f"#0{3 + self._len_last_byte // 8}x")
        return ret_str

    @property
    def bin(self) -> str:
        """
        Returns a string with the binary representations of each byte.

        >>> Bits(int_=255, length=8).bin
        '0b1111_1111'
        >>> Bits(int_=4095, length=12).bin
        '0b1111_1111 0b1111'
        >>> Bits(int_=65535, length=16).bin
        '0b1111_1111 0b1111_1111'

        >>> Bits(bin_="1111 11").bin
        '0b11_1111'

        :return: The string representation of the bits as binary.
        """
        ret_str = " ".join(format(byte, "#011_b") for byte in self._bytes)
        if self._len_last_byte > 4:
            # Add extra width to allow for grouping spacer
            last = format(self._last_byte, f"#0{3 + self._len_last_byte}_b")
            ret_str += " " + last if ret_str else last
        elif self._len_last_byte > 0:
            # No grouping spacer for binary number with less than 4 digits
            last = format(self._last_byte, f"#0{2 + self._len_last_byte}b")
            ret_str += " " + last if ret_str else last
        return ret_str


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
