import argparse
from os import path, makedirs, listdir
import sys

from PIL import Image


# TODO: make base Steganography class inheriting from image
# TODO: make classes for various steganography techniques
def xtract_lowest_bits_steg(image_: Image) -> Image:
    """
    Create new image by stripping the lowest two bits of each color channel
    on each pixel and then normalize the new image.

    """
    normal_vals = (63, 127, 191, 255)
    return image_.point(lambda i: normal_vals[(i & 0b00000011)])


def xtract_lowest_bits_steg_lazy(image_: Image) -> Image:
    """
    Create new image from only the lowest two bits of each color channel
    of each pixel and then normalize the image by bit shifting.

    """
    return image_.point(lambda i: (i & 0b00000011) << 6)


def xtract_lowest_bit_steg(image_: Image) -> Image:
    """
    Create new image from only the lowest two bits of each color channel
    of each pixel and then normalize the image by bit shifting.

    """
    return image_.point(lambda i: (i & 0b00000001) and 255)


if __name__ == "__main__":
    source_dir = path.realpath("./sources/steganography")
    default_output_dir = path.realpath("./generated/steganography")

    # User may pass arbitrarily many file names of images to be processed,
    # or the contents of ./images/sources/steganography will be processed.
    # (or the clever developer can edit their run configuration to the )
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-o",
                        "--output_dir",
                        help=f"Option output directory, default: {default_output_dir}",
                        nargs="?",
                        default=default_output_dir)
    parser.add_argument("-p", "--prefix",
                        help='Prefix to output files, default: "stega_"',
                        nargs="?",
                        default="stega_")
    parser.add_argument("-f", "--force",
                        help="Overwrite exiting files",
                        action="store_true")
    parser.add_argument("files",
                        help=f"Files to process, default: all files in {source_dir}",
                        nargs="*",
                        default=(path.join(source_dir, name) for name in listdir(source_dir)))
    # fmt: on
    args = parser.parse_args()

    try:
        if not path.exists(args.output_dir):
            makedirs(args.output_dir)
    except OSError:
        sys.stderr.write(f'There was an issue creating the output directory: "{args.output_dir}"\n')

    for image_file_name in args.files:
        image_path = path.realpath(image_file_name)
        out_file_path = path.join(args.output_dir, args.prefix + path.basename(image_file_name))

        if path.exists(out_file_path) and not args.force:
            print(f'Skipping file "{path.basename(out_file_path)}" - file already exists...')
            continue

        try:
            with Image.open(image_path) as image:
                xtract_lowest_bits_steg(image).save(out_file_path, "PNG")
        except OSError as exc:
            # TODO: Add builtin VTF file handling
            sys.stderr.write(f'Could not process steganography for file "{path.basename(image_path)}"\n')
            sys.stderr.write("\t" + str(exc) + "\n")
