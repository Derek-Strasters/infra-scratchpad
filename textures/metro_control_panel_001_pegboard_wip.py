"""A collection of stuff related to https://stalburg.net/Body_message#Pegs."""
from itertools import chain
from typing import Callable

from biterator import Bits
from termcolor import colored

from textures.metro_control_panel_001_pegboard import (
    PegGroup,
    metro_control_panel_001_pegboard,
)

ColorFunc = Callable[[str], str]


# NOTE: iso8859_10 is for finland


def bits_2_glyphs(bits: Bits, post_op: ColorFunc = lambda x: x) -> str:
    """
    Convert a sequence of bits (booleans) to one ascii character for each byte.

    Non printing characters are represented like "x1f".
    Each character or hex code will have a padded width of 3 characters, and
    a space will be inserted between consecutive characters.

    :param bits: Sequence of booleans as bits to be decoded.
    :param post_op: Optional function to applied to aplhanumerics.
    :return: String containing the decoded bits.

    >>> bits_2_glyphs(Bits([False, True, True, False, False, True, False, True]))
    ' e'
    >>> bits_2_glyphs(Bits([False, False, False, False, True, False, True, False]))
    ' a'
    """
    return " ".join(f"{x:2x}" if not chr(x).isalnum() else f" {post_op(chr(x))}" for x in bytes(bits))


if __name__ == "__main__":

    # Following are some tests applying a nor mask to various combinations of pegs.

    peg_group_names = (("pegs[1,1]", "pegs[1,2]"), ("pegs[3,1]", "pegs[3,2]"), ("pegs[4,1]", "pegs[4,2]"))

    # Flatten the peg groupings for easier handling.
    # pegs[1,1]                 pegs[3,1]    pegs[4,1]
    # pegs[1,2]                 pegs[3,2]    pegs[4,2]

    peg_groups = [
        PegGroup(
            name=peg_group_names[i][j],
            pegs=Bits(peg_str, ones={"P", "M"}),
            markers=Bits(peg_str, ones={"M"}),
        )
        for i, panels in enumerate(metro_control_panel_001_pegboard)
        for j, peg_str in enumerate(panels)
    ]

    zeros = [False] * 24

    def red_text(text: str):
        """Color the given text red when printed."""
        return colored(text, "red")

    # Some labeling to help identify interesting results.
    print()
    print(" " * 26 + (" " * 5).join(f"col {x: <2d}" for x in range(1, 9)))

    def nor_mask(a, b, mask):
        """
        Return a sequence of binary functions.

        Depending on masking_bools[i], each function returns either nor(a,b) if True, or left(a,b):=a otherwise.
        """
        return (mask ^ a) & (mask | a) & ~(mask & b)

    # Iterate through all combinations of the peg groupings.
    # Bits are represented by peg locations.
    # A nor mask is made from the marker locations from one of the peg groupings.
    for i, gp1 in enumerate(peg_groups):

        for j, gp2 in enumerate(peg_groups):
            print(f"{gp1.name} x {gp2.name}", end=" =  ")

            print(bits_2_glyphs(nor_mask(gp1.pegs, gp2.pegs, gp1.markers), red_text), end="   ")  # col 1
            print(bits_2_glyphs(nor_mask(gp1.pegs, gp2.pegs, gp2.markers), red_text), end="   ")  # col 2

            print(bits_2_glyphs(nor_mask(gp1.pegs, ~gp2.pegs, gp1.markers), red_text), end="   ")  # col 3
            print(bits_2_glyphs(nor_mask(gp1.pegs, ~gp2.pegs, gp2.markers), red_text), end="   ")  # col 4

            print(bits_2_glyphs(nor_mask(~gp1.pegs, gp2.pegs, gp1.markers), red_text), end="   ")  # col 5
            print(bits_2_glyphs(nor_mask(~gp1.pegs, gp2.pegs, gp2.markers), red_text), end="   ")  # col 6

            print(bits_2_glyphs(nor_mask(~gp1.pegs, ~gp2.pegs, gp1.markers), red_text), end="   ")  # col 7
            print(bits_2_glyphs(nor_mask(~gp1.pegs, ~gp2.pegs, gp2.markers), red_text), end="   ")  # col 8
            print()
        print()

    top, bot = tuple(tuple(chain(*(group[i] for group in metro_control_panel_001_pegboard))) for i in range(2))

    top_pegs = Bits(top, ones={"P"})
    top_mkrs = Bits(top, ones={"M"})
    bot_pegs = Bits(bot, ones={"P"})
    bot_mkrs = Bits(bot, ones={"M"})

    print(bits_2_glyphs(top_pegs ^ top_mkrs))
    print(bits_2_glyphs(top_pegs ^ bot_mkrs))
    print(bits_2_glyphs(bot_pegs ^ top_mkrs))
    print(bits_2_glyphs(bot_pegs ^ bot_mkrs))
    print(bits_2_glyphs(top_pegs ^ bot_pegs))
    print(bits_2_glyphs(top_mkrs ^ bot_mkrs))
    print()
    print(bits_2_glyphs(top_pegs | top_mkrs))
    print(bits_2_glyphs(top_pegs | bot_mkrs))
    print(bits_2_glyphs(bot_pegs | top_mkrs))
    print(bits_2_glyphs(bot_pegs | bot_mkrs))
    print(bits_2_glyphs(top_pegs | bot_pegs))
    print(bits_2_glyphs(top_mkrs | bot_mkrs))
