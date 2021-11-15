"""Utility functions."""
from abc import abstractmethod
from collections.abc import MutableSequence
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
    Any,
    Iterator,
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
Rotatable = TypeVar("Rotatable", bound="ConcatenableSequence")
ValidBit = Union[bool, Literal[1], Literal[0], Literal["1"], Literal["0"]]
ValidBits = Union[Iterable[ValidBit], Tuple[int, int]]


class ConcatenableSequence(Protocol[T_co]):
    """
    Any Sequence T where +(:T, :T) -> T.
    Types must support indexing and concatenation.
    """

    def __add__(self, other: Rotatable) -> Rotatable:
        ...

    def __getitem__(self, index: int) -> T_co:
        ...

    def __len__(self) -> int:
        ...


# NOTE Finland uses iso8859_10 encoding.
# NOTE The proper bitwise operation for a NOR mask is (mask_ ^ left_) & (left_ | mask_) & ~(right & mask_).


class Bits(MutableSequence[ValidBit]):
    """
    Stores bits like an array, and can be manipulated and iterated as such.
    Bits can be instantiated with:
        * A string of binary e.g. "1010" or "0b1100_0010"
        * A prefixed string of hexadecimal e.g. "0x1f 0xb2" or "0xbadc0de"
        * A bytes-like object
        * An integer-like object with a specified bit_length
        * An Iterable containing any of: True, False, 0, 1, "0", "1"
    The add (+) operator functions as concatination only, and supports all of
    the above schemes.  Addition may be done by first casting to int.
    Binary and hexadecimal representations may be accessed with the 'bin' and
    'hex' properties and the 'decode' method may be used to read the bits as
    bytes using a specified codec.

    >>> Bits("10011001").hex
    '0x99'
    >>> Bits("0xFF").bin
    '0b1111_1111'
    >>> Bits("0xFF") + "0b1001_1001"
    Bits("0b1111111110011001")
    >>> Bits() + b"Hi"
    Bits("0b0100100001101001")
    >>> Bits(x % 4 < 2 for x in range(16)) # Supports generators
    Bits("0b1100110011001100")
    >>> Bits("1111 0011 0000 1010")[0:8] # Slicing
    Bits("0b11110011")
    >>> Bits("0xAAAA")[:8:2]
    Bits("0b1111")
    >>> Bits("1111") << 4 # Shift bits left
    Bits("0b11110000")

    'nor' mask example:

    >>> mask_ = Bits('00001111')
    >>> left_ = Bits('01010101')
    >>> right = Bits('00110011')
    >>> ((mask_ ^ left_) & (mask_ | left_) & ~(mask_ & right)).bin
    '0b0101_1000'

    """

    __slots__ = ["_bytes", "_last_byte", "_len_last_byte", "_len"]

    _bytes: bytearray
    _len: int
    # Contains the trailing (incomplete) byte, that has less than 8 bits defined
    _last_byte: int
    _len_last_byte: int

    _ZEROS = {0, "0", False}
    _ONES = {1, "1", True}

    class SubscriptError(Exception):
        """
        Raised when an object is accessed by a subscript of an unsupported type.
        """

        def __init__(self, subbing_class: "Bits", subscript: Any, message="unsupported subscript"):
            self.class_name = type(subbing_class).__name__
            self.subscript_class_name = type(subscript).__name__
            self.message = message
            super().__init__(self.message)

        def __str__(self):
            return f"{self.message}, '{self.class_name}' does not support '{self.subscript_class_name}' subscripts"

    def __init__(
        self,
        bit_values: Union[Iterable[ValidBit], SupportsInt, str, ByteString] = None,
        bit_length: int = None,
    ):
        self._bytes = bytearray()
        self._len_last_byte = 0
        self._last_byte = 0
        self._len = 0

        if bit_values is not None:
            if isinstance(bit_values, type(self)):
                self._bytes = bit_values._bytes.copy()
                self._len = bit_values._len
                self._last_byte = bit_values._last_byte
                self._len_last_byte = bit_values._len_last_byte
            else:
                self.extend(bit_values, bit_length)

    def copy(self):
        new = type(self)()
        new._bytes = self._bytes.copy()
        new._len = self._len
        new._last_byte = self._last_byte
        new._len_last_byte = self._len_last_byte
        return new

    def _decimal_digits(self):
        """
        The minimum decimal digits it would take to represent the same number
        of bits if they were all true.

        >>> Bits(123456789, (123456789).bit_length())._decimal_digits()
        9

        :return: The number of decimal digits.
        """
        largest_possible = (1 << self._len) - 1
        return len(str(largest_possible))

    def __repr__(self):
        """
        A quite representation of the Bits object.  The repr is equivalent to
        code that would create an identical object, up to a size of 64 bytes;
        after which it it abbreviated.

        :return: A string representation of the Bits object.
        """
        if len(self) <= 64:
            return f'Bits("{format(int(self), f"#0{self._len + 2}b")}")'

        if self._decimal_digits() <= 64:
            return f'Bits({format(int(self), f"0{self._decimal_digits()}d")}, {self._len})'

        if len(self) // 8 <= 64:
            return f'Bits("{format(int(self), f"#0{len(self) // 8 + 2}x")}")'
        if self._len_last_byte > 0:
            return f'Bits("{self[:24].hex} ... {self[-(self._len_last_byte + 16):].hex}")'
        return f'Bits("{self[:24].hex} ... {self[-24:].hex}")'

    def bytes_gen(self, index: int = 0) -> Iterator[int]:
        """
        A generator for accessing the bytes.
        An incomplete byte will be written from the left, for example:

        >>> for byte in Bits('10101010 1111').bytes_gen():
        ...     print(bin(byte))
        0b10101010
        0b11110000

        :return: The Iterator.
        """
        for i in range(index, len(self._bytes)):
            yield self._bytes[i]
        if self._len_last_byte:
            yield self._last_byte << 8 - self._len_last_byte

    def __bytes__(self) -> Iterable[int]:
        return bytes(self.bytes_gen())

    def decode(self, *args, **kwargs):
        """
        A convenience wrapper for `bytes().decode()`
        Decode the bytes using the codec registered for encoding.
        """
        return bytes(self).decode(*args, **kwargs)

    def __bool__(self):
        return bool(self._bytes) or bool(self._last_byte)

    def _byte_bit_indices(self, index: int) -> Tuple[int, int, int]:
        """
        For a given index:
            calculates the index of _bytes the index falls in (if applicable),
            calculates the index of the bit it fall in within the byte (0-7),
            clips the value of index to within -len(self) through len(self),
        and returns these values in a tuple.

        :param index: The index to calculate.
        :return: The tuple with computed index values.
        """
        if index >= len(self) or index < -len(self):
            raise IndexError

        # Modulo corrects negative indices
        if index < 0:
            index = index % len(self)

        # The first is the index of the byte that the index is within.
        # The second is the index of the bit within the byte (counting from the left).
        return index // 8, index % 8, index

    @classmethod
    def _validate_bit(cls, value: Union[str, int, bool]):
        """
        Validates a value as a ValidBit returns a bool representation.

        >>> Bits._validate_bit(0)
        False
        >>> Bits._validate_bit(1)
        True
        >>> Bits._validate_bit("1")
        True
        >>> Bits._validate_bit(True)
        True
        >>> Bits._validate_bit("Puppy")
        Traceback (most recent call last):
         ...
        TypeError: could not determine single bit value for 'Puppy'

        :param value: The value to check.
        :return: The bool representation.
        """
        if value in cls._ONES:
            return True
        if value in cls._ZEROS:
            return False

        raise TypeError(f"could not determine single bit value for {repr(value)}")

    def insert(self, index: int, value: ValidBit) -> None:
        """
        Insert a bit at given index.

        >>> bits = Bits('001')
        >>> bits.insert(0, True)
        >>> bits.bin
        '0b1001'
        >>> for _ in range(4):
        ...     bits.insert(len(bits), False)
        >>> bits.bin
        '0b1001_0000'
        >>> bits.insert(5, "1")
        >>> bits.bin
        '0b1001_0100 0b0'
        >>> bits.insert(-2, 1)
        >>> bits.bin
        '0b1001_0101 0b00'

        :param index: The index at whitch to insert the bit.
        :param value: The bit to be inserted.
        """
        value = self._validate_bit(value)

        # If the index is above the length, set it to the length.
        # If the index is below the negative length, set it to the negative length.
        # Then if the new index is negative, take the modulo so that -1 accesses the last element.
        if len(self) == 0:
            index = 0
        else:
            if index >= 0:
                index = min(len(self), index)
            else:
                index = max(-len(self), index) % len(self)
        byte_index, bit_index = index // 8, index % 8

        # If appending to the end.
        if index == len(self):
            self._last_byte = (self._last_byte << 1) | value
            self._increment_last_byte()

        # If inserting within the last (incomplete) byte.
        elif byte_index == len(self._bytes):
            self._last_byte = self._insert_bit_in_byte(self._last_byte, self._len_last_byte, bit_index, value)
            self._increment_last_byte()

        # If inserting anywhere else.
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

    def extend(self, values: Union[Iterable[ValidBit], int], bit_length: int = None) -> None:
        """
        Allows extending an integer for a given given a bit length.
        Allows extending binary and hexidecimal strings with prefixes, spaces, and separators.
        Allows extending bytes like objects.

        >>> bits = Bits()
        >>> bits.extend('1010')
        >>> bits.bin
        '0b1010'
        >>> bits.extend(15, 4)
        >>> bits.bin
        '0b1010_1111'
        >>> bits.extend(b"A")
        >>> bits.bin
        '0b1010_1111 0b0100_0001'
        >>> bits.extend(17, 4) # 0b10001
        >>> bits.bin
        '0b1010_1111 0b0100_0001 0b1000'

        :param values: The values to extend with.
        :param bit_length: Number of bits if extending by an integer
        """
        # Attempt to interpret the bit data as provided by the user

        # Integers
        if isinstance(values, SupportsInt) and not isinstance(values, type(self)):
            # Case of a single bool:
            if isinstance(values, bool):
                raise TypeError("'bool' object is not iterable")
            if bit_length is None:
                raise ValueError("a bit_length must be specified")
            if bit_length <= 0:
                # Rather than default to len(int_) the length must be provided to
                # reduce unexpected behavior and errors in implementation.
                raise ValueError("bit_length must be greater than one")
            if int(values).bit_length() > bit_length:
                # Ensure the first bit appended is the leftmost bit of the integer
                values >>= int(values).bit_length() - bit_length
            super().extend(bool((1 << i) & int(values)) for i in range(bit_length - 1, -1, -1))

        elif bit_length is not None:
            raise ValueError("bit_length should only be provided for integers")

        # Binary and Hexadecimal strings
        elif isinstance(values, str):
            values = values.strip().replace("_", "").replace(" ", "")

            # Hexadecimal strings are require to have the 0x prefix
            if values.startswith(("0x", "0X")):
                for nibble in values.replace("0x", "").replace("0x", "0X"):
                    self.extend(int(nibble, 16), 4)

            # All other strings are assumed to be binary.
            else:
                super().extend(values.replace("0b", ""))
        elif isinstance(values, ByteString):
            self._bytes += values
            self._len += len(values) * 8
        else:
            super().extend(values)

    @staticmethod
    def _insert_bit_in_byte(byte: int, length: int, index: int, value: bool) -> int:
        """
        Insert a bit in a byte, indexed from the left.

        >>> bin(Bits._insert_bit_in_byte(0b1010010, 7, 4, True))
        '0b10101010'

        :param byte: Byte in which to insert the bit.
        :param length: Length of the Byte.
        :param index: Index at which to insert the bit.
        :param value: Value to be inserted.
        :return: Byte with new bit inserted.
        """
        right_index = length - index
        left_bits = byte >> right_index
        right_bits = byte & ((1 << right_index) - 1)
        return (((left_bits << 1) | value) << right_index) | right_bits

    def _increment_last_byte(self) -> None:
        """
        Called when a bit has been added anywhere in the last (incomplete) byte.

        >>> bits = Bits(0b111_1111, bit_length=7)
        >>> bits._len_last_byte
        7
        >>> bits.append(1)
        >>> bits._len_last_byte
        0
        >>> len(bits)
        8

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
        Retrieve a bit or a slice of bits.

        >>> Bits('0001 0000')[3]
        True
        >>> Bits('0001 1000')[3:5]
        Bits("0b11")
        >>> Bits("000011110011001101010101")[:-16]
        Bits("0b00001111")
        >>> Bits("000011110011001101010101")[-8:]
        Bits("0b01010101")

        :param index: The index or slice to retrieve.
        :return: The new Bits object or a bit value.
        """
        if isinstance(index, int):
            byte_index, bit_index, index = self._byte_bit_indices(index)

            # If the index is in the last (incomplete) byte.
            if byte_index == len(self._bytes):
                return self._get_bit_from_byte(self._last_byte, self._len_last_byte, bit_index)

            # If the index is anywhere else.
            return self._get_bit_from_byte(self._bytes[byte_index], 8, bit_index)

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))

            # For the case where the slice starts from a whole byte.
            if step == 1 and start % 8 == 0:
                last_byte_index, last_bit_index = stop // 8, stop % 8
                start_byte_index = start // 8
                new = type(self)(self._bytes[start_byte_index:last_byte_index])
                # Append any remaining bits.
                if last_bit_index:
                    for i in range(stop - last_bit_index, stop):
                        # Recurse into the branch for integers
                        new.append(self[i])
                return new

            # For all other cases (not particularly efficient).
            new = type(self)()
            for i in range(start, stop, step):
                # Recurse into the branch for integers
                new.append(self[i])
            return new

        raise self.SubscriptError(self, index)

    @staticmethod
    def _get_bit_from_byte(byte: int, length: int, index: int) -> bool:
        """
        Return the bit value at the given index, indexed from the left.

        >>> Bits._get_bit_from_byte(0b00000100, 8, 5)
        True

        :param byte: Byte from which to get a bit.
        :param index: Index of bit to retrieve.
        :param length: Length of byte.
        :return: The value of the bit.
        """
        right_index = length - index - 1
        return bool((1 << right_index) & byte)

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: ValidBit) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Union[Iterable[ValidBit], str, ByteString]) -> None:
        ...

    def __setitem__(self, index, other):
        """
        Sets a bit or slice of bits.

        >>> bits = Bits('1111 1111 1111')
        >>> bits[4:8] = '0000'
        >>> bits.bin
        '0b1111_0000 0b1111'
        >>> bits[4:8] = 15
        >>> bits.bin
        '0b1111_1111 0b1111'
        >>> bits[-4:] = '0000'
        >>> bits.bin
        '0b1111_1111 0b0000'
        >>> bits[0] = False
        >>> bits.bin
        '0b0111_1111 0b0000'

        :param index: The index or slice to modify.
        :param other: The bit or bits to replace the old bit or bits.
        """
        if isinstance(index, int):
            other = self._validate_bit(other)
            byte_index, bit_index, index = self._byte_bit_indices(index)

            # If the index is in the last (incomplete) byte.
            if byte_index == len(self._bytes):
                self._last_byte = self._set_bit_in_byte(self._last_byte, self._len_last_byte, bit_index, other)

            # If the index is anywhere else.
            else:
                self._bytes[byte_index] = self._set_bit_in_byte(self._bytes[byte_index], 8, bit_index, other)

        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))

            # Cast other to a Bits object
            if isinstance(other, int):
                other_bit = iter(type(self)(other, stop - start))
            else:
                other_bit = iter(type(self)(other))

            try:
                for i in range(start, stop, step):
                    # Recurse into the branch for integers
                    self[i] = next(other_bit)
            except StopIteration:
                pass

        else:
            raise self.SubscriptError(self, index)

    @classmethod
    def _set_bit_in_byte(cls, byte: int, length: int, index: int, value: bool) -> int:
        """
        Modify a bit in a byte, indexed from the left.

        >>> Bits._set_bit_in_byte(0b11011111, 8, 2, True)
        255

        :param byte: Byte in which to modify a bit.
        :param length: Length of the byte.
        :param index: Index of the bit to modify.
        :param value: Value to modify the bit to.
        :return: The Byte with bit modified.
        """
        right_index = length - index - 1
        # If the bit is the same, do nothing.
        if bool((1 << right_index) & byte) == value:
            return byte
        # The bit is different, flip it.
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

        >>> bits = Bits("1000 0000 0000 0100 0001")
        >>> del bits[13]
        >>> bits.bin
        '0b1000_0000 0b0000_0000 0b001'
        >>> bits = Bits("1010 1010 1010 1010 0000")
        >>> del bits[1::2]
        >>> bits.bin
        '0b1111_1111 0b00'
        >>> del bits[8:10]
        >>> bits.bin
        '0b1111_1111'
        >>> del bits[-4:]
        >>> bits.bin
        '0b1111'

        :param index: Index or slice to delete.
        """
        if isinstance(index, int):
            byte_index, bit_index, index = self._byte_bit_indices(index)

            # If the bit deleted in in the last (incomplete) byte.
            if byte_index == len(self._bytes):
                self._last_byte = self._del_bit_from_byte(self._last_byte, self._len_last_byte, bit_index)
                self._decrement_last_byte()

            # All other cases.
            else:
                # Remove the bit from the target byte, then append the first bit from the next byte.
                # Cascade similarly through the list of bytes.
                new_byte = self._del_bit_from_byte(self._bytes[byte_index], 8, bit_index)
                for i in range(byte_index + 1, len(self._bytes)):
                    first_bit = bool(self._bytes[i] & 0b1000_0000)
                    self._bytes[i - 1] = (new_byte << 1) | first_bit
                    new_byte = self._bytes[i] & 0b0111_1111

                # If the last (incomplete) byte is not empty, append the first bit from it.
                if self._len_last_byte:
                    first_bit = bool(self._last_byte & (1 << self._len_last_byte - 1))
                    self._bytes[-1] = (new_byte << 1) | first_bit
                    # Truncate the first bit of the last (incomplete) byte.
                    self._last_byte = self._last_byte & ((1 << self._len_last_byte - 1) - 1)

                # If the last (incomplete) byte is empty, remove the last full byte.
                else:
                    self._bytes.pop()
                    # The former last full byte becomes the last (incomplete) byte with it's first bit removed.
                    self._last_byte = new_byte

                # Decrement the length and last (incomplete) byte length in both cases.
                self._decrement_last_byte()

        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))

            # NOTE: ***VERY inefficient*** Consider refactor.
            # NOTE: Good opportunity to use interval library to remove all deleted bits and concat what remains.
            # Always proceeds in reverse order to not mess up the indexing.
            removal_indices = sorted(list(range(start, stop, step)), reverse=True)
            for i in removal_indices:
                del self[i]

        else:
            raise self.SubscriptError(self, index)

    @staticmethod
    def _del_bit_from_byte(byte: int, length: int, index: int) -> int:
        """
        Remove a bit from a byte, indexed from the left.

        >>> Bits._del_bit_from_byte(0b00010000, 8, 3)
        0

        :param byte: Byte from which to remove a bit.
        :param length: Length of the byte.
        :param index: Index of the bit to remove.
        :return: The Byte with bit removed.
        """
        right_index = length - index
        left_bits = (byte >> right_index) << right_index - 1
        right_bits = byte & ((1 << right_index - 1) - 1)
        return left_bits | right_bits

    def _decrement_last_byte(self) -> None:
        """
        Called when a bit has been removed anywhere in the last (incomplete) byte.

        >>> bits = Bits(0b010001000, bit_length=9)
        >>> bits._len_last_byte
        1
        >>> del bits[0]
        >>> bits._len_last_byte
        0
        """
        self._len_last_byte -= 1
        self._len -= 1
        if self._len_last_byte < 0:
            self._len_last_byte = 7

    def __len__(self) -> int:
        return self._len

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

    def __add__(self, other: ValidBits) -> "Bits":
        """
        This is concatenation, NOT addition.

        >>> (Bits("0110") + Bits("1001")).bin
        '0b0110_1001'
        >>> (Bits("0110") + "1001").bin
        '0b0110_1001'
        >>> (Bits("0110") + (15, 4)).bin
        '0b0110_1111'

        :param other: Other object to be concatenated.
        :return: New Bits object that is a concatenation of the inputs.
        """
        new = self.copy()
        new.extend(*(other if isinstance(other, tuple) else (other,)))
        return new

    def __lshift__(self, index: int) -> "Bits":
        """
        Left shift the bits.

        >>> (Bits("1111") << 4).bin
        '0b1111_0000'

        :param index: Number of places to shift
        :return: Shifted Bits object
        """
        new = self.copy()
        new.extend(type(self)(0, bit_length=index))
        return new

    def __rshift__(self, index: int) -> "Bits":
        """
        Right shift the bits.

        >>> (Bits("11110000") >> 4).bin
        '0b1111'

        :param index: Number of places to shift
        :return: Shifted Bits object
        """
        new = type(self)()
        new.extend(self[:-index])
        return new

    def __and__(self, other: ValidBits) -> "Bits":
        """
        Bitwise and operation.

        >>> (Bits('01111000') & Bits('00011110')).bin
        '0b0001_1000'
        >>> (Bits('0111') & Bits('00011110')).bin
        '0b0001'
        >>> (Bits("1110") & "0b0111").bin
        '0b0110'
        >>> (Bits("1110") & (7, 4)).bin
        '0b0110'

        :param other: Other Bits to 'and' with
        :return: Combined Bits objects
        """
        new = type(self)()
        for self_bit, other_bit in zip(self, Bits(*(other if isinstance(other, tuple) else (other,)))):
            new.append(self_bit & other_bit)
        return new

    def __xor__(self, other: ValidBits) -> "Bits":
        """
        Bitwise xor operation.

        >>> (Bits('01111000') ^ Bits('00011110')).bin
        '0b0110_0110'
        >>> (Bits('01111000') ^ '0b00011110').bin
        '0b0110_0110'
        >>> (Bits("1110") ^ "0111").bin
        '0b1001'

        :param other: Other Bits to 'xor' with
        :return: Combined Bits objects
        """
        new = type(self)()
        for self_bit, other_bit in zip(self, Bits(*(other if isinstance(other, tuple) else (other,)))):
            new.append(self_bit ^ other_bit)
        return new

    def __or__(self, other: ValidBits) -> "Bits":
        """
        Bitwise or operation.

        >>> (Bits('01111000') | Bits('00011110')).bin
        '0b0111_1110'
        >>> (Bits("1100") | "0011").bin
        '0b1111'

        :param other: Other Bits to 'or' with
        :return: Combined Bits objects
        """
        new = type(self)()
        for self_bit, other_bit in zip(self, Bits(*(other if isinstance(other, tuple) else (other,)))):
            new.append(self_bit | other_bit)
        return new

    def __iadd__(self, other: ValidBits) -> "Bits":
        """
        Extend in-place.

        >>> bits = Bits("1111")
        >>> bits += "0000"
        >>> bits.bin
        '0b1111_0000'
        >>> bits += 255, 8
        >>> bits.bin
        '0b1111_0000 0b1111_1111'

        :param other: Bits to extend.
        :return: The Bits object that was modified in place.
        """
        if isinstance(other, tuple):
            self.extend(*other)
            return self
        self.extend(other)
        return self

    def __ilshift__(self, index: int) -> "Bits":
        """
        Left bitshift in-place.

        >>> bits = Bits("1111")
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

        >>> bits = Bits("1111 1111")
        >>> bits >>= 4
        >>> bits.bin
        '0b1111'

        :param index: Number of places to shift.
        :return: The Bits object that was modified in place.
        """
        if index:
            del self[-index:]
        return self

    def __iand__(self, other: ValidBits) -> "Bits":
        """
        Bitwise 'and' with other bits; in-place.

        >>> bits_ = Bits("1110")
        >>> bits_ &= "0111"
        >>> bits_.bin
        '0b0110'

        :param other: The Iterable bits to 'and' with.
        :return: The Bits object that was modified in place.
        """
        index = 0
        for index, bits in enumerate(zip(self, Bits(*(other if isinstance(other, tuple) else (other,))))):
            self[index] = bits[0] & bits[1]
        if len(self) > index + 1:
            del self[-(index + 1) :]
        return self

    def __ixor__(self, other: ValidBits) -> "Bits":
        """
        Bitwise 'xor' with other bits; in-place.

        >>> bits_ = Bits("0110")
        >>> bits_ ^= "0101"
        >>> bits_.bin
        '0b0011'

        :param other: The Iterable bits to 'xor' with.
        :return: The Bits object that was modified in place.
        """
        index = 0
        for index, bits in enumerate(zip(self, Bits(*(other if isinstance(other, tuple) else (other,))))):
            self[index] = bits[0] ^ bits[1]
        if len(self) > index + 1:
            del self[-(index + 1) :]
        return self

    def __ior__(self, other: ValidBits) -> "Bits":
        """
        Bitwise 'or' with other bits; in-place.

        >>> bits_ = Bits("1100")
        >>> bits_ |= "0011"
        >>> bits_.bin
        '0b1111'

        :param other: The Iterable bits to 'or' with.
        :return: The Bits object that was modified in place.
        """
        index = 0
        for index, bits in enumerate(zip(self, Bits(*(other if isinstance(other, tuple) else (other,))))):
            self[index] = bits[0] | bits[1]
        if len(self) > index + 1:
            del self[-(index + 1) :]
        return self

    def __invert__(self) -> "Bits":
        """
        Return a Bits object with each bit inverted.

        >>> (~Bits('01001110')).bin
        '0b1011_0001'

        :return: The Bits object with inverted bits.
        """
        new = type(self)()
        new._bytes = bytearray(byte ^ 0b11111111 for byte in self._bytes)
        new._last_byte = self._last_byte ^ ((1 << self._len_last_byte) - 1)
        new._len = self._len
        return new

    def __int__(self) -> int:
        """
        Represent the sequence of bits as an int.

        >>> int(Bits("0xff"))
        255
        >>> int(Bits("0xfff"))
        4095
        >>> int(Bits("0xffff"))
        65535

        :return: The integer representation.
        """
        return (int.from_bytes(self._bytes, "big") << self._len_last_byte) | self._last_byte

    @property
    def hex(self) -> str:
        """
        Returns a string with hexadecimal representation of each byte.

        >>> Bits("0b1111_1111").hex
        '0xff'
        >>> Bits("0b1111_1111 0b1111").hex
        '0xff 0xf'
        >>> Bits("0b1111_1111 0b1111_1111").hex
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

        >>> Bits(255, bit_length=8).bin
        '0b1111_1111'
        >>> Bits(4095, bit_length=12).bin
        '0b1111_1111 0b1111'
        >>> Bits(65535, bit_length=16).bin
        '0b1111_1111 0b1111_1111'

        >>> Bits("1111 11").bin
        '0b11_1111'

        :return: The string of the bits in binary representation.
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


def chunked(items: Sequence[T], n: int) -> Sequence[Sequence[T]]:
    """
    Yield successive n-sized chunks from lst.
    The last chunk is trunkated as needed.

    :param items: A collection of things to be chunked.
    :param n: The size of each chunk.

     >>> list(chunked([1, 2, 3, 4, 5], 2))
     [[1, 2], [3, 4], [5]]
     >>> list(chunked((1, 1, 1, 1, 1), 2))
     [(1, 1), (1, 1), (1,)]
    """
    # noinspection PyArgumentList
    return type(items)(type(items)(items[i : i + n]) for i in range(0, len(items), n))


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
