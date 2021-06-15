from typing import Iterable, Dict, Tuple, List


def all_string_indices(haystack: str, needle: str) -> Iterable[int]:
    """
    Finds all indices of a given needle in a haystack.
    :return: An iterable of the found indices.

    >>> list(all_string_indices("ABA", "A"))
    [0, 2]
    """
    start = -1
    while True:
        try:
            start = haystack.index(needle, start + 1)
            yield start
        except ValueError:
            break


def find_char(data: Dict[str, str], char: str) -> Iterable[Tuple[str, int]]:
    """
    Finds all indices of a given char in a dictionary of strings.
    :param data: The dictionary to search in.
    :param char: The char to search for.
    :return: An iterable of tuples containing the dictionary key and the indices of the char in the corresponding dictionary values.

    >>> list(find_char({"A": "BC"}, "C"))
    [('A', 2)]
    """

    assert len(char) == 1
    for key, values in data.items():
        for index in all_string_indices(values, char):
            yield key, index + 1


def find_string(
    data: Dict[str, str], string: str
) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """
    Finds all dictionary coordinates of the given string chars.
    :param data: The dictionary to search in.
    :param string: The string containing the chars to search for.
    :return: A list of tuples, each tuple containing the char and the corresponding list of coordinates.

    >>> list(find_string({"A": "BC"}, "CD"))
    [('C', [('A', 2)]), ('D', [])]
    """
    return [(char, list(find_char(data, char))) for char in string]


def swap_values(data: Dict[str, str], a: str, b: str):
    """
    Swaps the values in a dictionary.
    :param data: The dictionary to modify.
    :param a: The first key to swap.
    :param b: The second key to swap.

    >>> data = {"A": "B", "C": "D"}
    >>> swap_values(data, "A", "C")
    >>> data
    {'A': 'D', 'C': 'B'}
    """
    data[a], data[b] = data[b], data[a]


def rotate_column(data: Dict[str, str], column: int, n: int):
    """
    Rotates a column in a dictionary.
    :param data: The dictionary.
    :param column: The column to rotate.
    :param n: The amount of rotation.

    >>> data = {1: "A", 2: "B", 3: "C"}
    >>> rotate_column(data, 0, 1)
    >>> data
    {1: 'C', 2: 'A', 3: 'B'}
    """
    keys = list(data.keys())
    for _ in range(n):
        prev = data[keys[-1]][column]
        for key in keys:
            s = data[key]
            current = s[column]
            data[key] = s[:column] + prev + s[column + 1 :]
            prev = current


def rotate_str_left(string: str, n: int) -> str:
    """
    Rotates a string.
    :param string: The string to rotate.
    :param n: The amount of rotation.
    :return: The rotated string.

    >>> rotate_str_left("hello", 2)
    'llohe'
    """
    return string[n:] + string[:n]


def rotate_str_right(string: str, n: int) -> str:
    return rotate_str_left(string, len(string) - n)


def rotate_value(data: Dict[str, str], key: str, n: int):
    """
    Rotates a value in a dictionary.
    :param data: The dictionary to modify.
    :param key: The key to rotate the value of.
    :param n: The amount of rotation

    >>> data = {"A": "BCDE"}
    >>> rotate_value(data, "A", 1)
    >>> data
    {'A': 'CDEB'}
    """
    data[key] = rotate_str_left(data[key], n)


def swap_columns(data: Dict[str, str], a: int, b: int):
    """
    Swap two columns of a dictionary.
    :param data: The dictionary to modify.
    :param a: The first column.
    :param b: The second column.

    >>> data = {"A": "BC", "D": "EF"}
    >>> swap_columns(data, 0, 1)
    >>> data
    {'A': 'CB', 'D': 'FE'}
    """
    for key, values in data.items():
        ac = values[a]
        bc = values[b]
        s = values
        s = s[:a] + bc + s[a + 1 :]
        s = s[:b] + ac + s[b + 1 :]
        data[key] = s


def char_idx(char: str) -> int:
    """
    The index of an uppercase character.
    :param char: The input character.
    :return: The index.

    >>> char_idx("A"), char_idx("B")
    (1, 2)
    """
    assert len(char) == 1
    return ord(char) - ord("A") + 1


def binary_decode(binary: str) -> str:
    """
    Decodes an 8-bit binary number.
    :param binary: A binary string.
    :return: The character representation.

    >>> binary_decode("01010011")
    'S'
    """
    assert len(binary) == 8 and all(c in ("0", "1") for c in binary)
    return chr(sum(2 ** i for i, c in enumerate(reversed(binary)) if c == "1"))


def binary_decode_multi(words: str) -> str:
    """
    Decodes a string of binary codes.
    :param words: The binary codes.
    :return: The decoded string.

    >>> binary_decode_multi("01010011 01101111")
    'So'
    """
    return "".join(map(binary_decode, words.split()))


def binary_encode(string: str) -> Iterable[str]:
    for c in map(ord, string):
        yield "".join(str((c >> (7 - i)) & 1) for i in range(8))


def insert_spaces(string: str, every: int) -> str:
    """
    Insert spaces at regular intervals.
    :param string: The string to insert the spaces into.
    :param every: The interval to insert spaces at.

    >>> insert_spaces("ABCDEFGH", 3)
    'ABC DEF GH'
    """
    return " ".join(
        string[every * i : every * (i + 1)]
        for i in range((len(string) + every - 1) // every)
    )
