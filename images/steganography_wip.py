import argparse
from datetime import datetime, date
from pathlib import Path
import sys
from typing import List

from PIL import Image
from PIL.ExifTags import TAGS

_TAGS_R = {value: key for key, value in TAGS.items()}


# TODO: make base Steganography class inheriting from image
# TODO: make classes for various steganography techniques


def normalized_bit_mask(image_: Image.Image, mask: int) -> Image.Image:
    """
    Apply a bitmask to each channel (color) of each pixel and normalize the results.

    :param image_: PIL Image object to apply bitmask to
    :param mask: Bitmask to be applied
    :return: Image object with applied bitmask
    """
    normalization_factor = 0b1111_1111 // mask
    return image_.point(lambda i: (i & mask) * normalization_factor)


if __name__ == "__main__":
    source_dir = Path("./sources/steganography")
    default_output_dir = Path("./generated/steganography")

    # User may pass arbitrarily many file names of images to be processed,
    # or the contents of ./images/sources/steganography will be processed.
    # (or the clever developer can edit their run configuration to the )
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-o",
                        "--output_dir",
                        help=f"Option output directory, default: {default_output_dir}",
                        default=default_output_dir,
                        type=Path)
    parser.add_argument("-p",
                        "--prefix",
                        help='Prefix to output files, default: "stega_{bitmask}mask_"\n'
                             'The following variable are available for formatting:\n'
                             '{resolution}   Resolution of the image\n'
                             '{bitmask}      The bitmask used (as an integer)\n'
                             '{date}         The ISO date\n'
                             '{time}         The iso time\n'
                             '{entropy}      The numerical entropy of the image',
                        default="stega_{bitmask}mask_",
                        type=str)
    parser.add_argument("-d",
                        "--display",
                        help="Display image windows with the processed images",
                        action="store_true")
    parser.add_argument("-b",
                        "--bitmask",
                        help="The bitmask (as an integer) that will be applied, default: 3 (0b00000011)",
                        default=0b0000_0011,
                        type=int)
    mu_ex_group = parser.add_mutually_exclusive_group()
    mu_ex_group.add_argument("-f",
                             "--force",
                             help="Overwrite exiting files",
                             action="store_true")
    mu_ex_group.add_argument("-n",
                             "--no_save",
                             help="Do not save file (overrides -p and -o)",
                             action="store_true")
    parser.add_argument("files",
                        help=f"Files to process, default: all files in {source_dir}",
                        nargs="*",
                        default=tuple(source_dir.iterdir()),
                        type=Path)
    # fmt: on
    cmd_args = parser.parse_args()

    # Move to regular variables for typing
    output_dir: Path = cmd_args.output_dir
    prefix: str = cmd_args.prefix
    display_results: bool = cmd_args.display
    bitmask: int = cmd_args.bitmask
    force_overwrite: bool = cmd_args.force
    no_save: bool = cmd_args.no_save
    files: List[Path] = cmd_args.files

    if not (0 < bitmask <= 255):
        sys.stderr.write(f"Bitmask must be within 1 byte in size (1-255): got {bitmask}\n")
        exit(1)

    try:
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
    except FileExistsError:
        pass
    except OSError:
        sys.stderr.write(f'There was an issue creating the output directory: "{output_dir}"\n')

    for image_file_path in files:
        image_file_path: Path
        image_basename = image_file_path.name

        try:
            with Image.open(image_file_path) as image:
                processed: Image.Image = normalized_bit_mask(image, bitmask)

                # Get Entropy
                entropy: float = processed.entropy()
                entropy_str = f"Entropy: {entropy}"

                # Get and modify EXIF data
                exif = processed.getexif()
                exif_description_str = f"{image_basename} - {entropy_str}"
                exif[_TAGS_R["ImageDescription"]] = exif_description_str

                if display_results:
                    processed.show()

                if not no_save:
                    # Format the prefix, only the following keys are allowed for the users.
                    # Any keys not found raise an error.
                    user_namespace = {
                        "resolution": f"{processed.size[0]}x{processed.size[1]}",
                        "bitmask": str(bitmask),
                        "date": date.today().isoformat(),
                        "time": datetime.now().time().isoformat().replace(".", ","),
                        "entropy": str(entropy).replace(".", ","),
                    }
                    formatted_prefix = prefix.format(**user_namespace)

                    out_file_path: Path = output_dir / (formatted_prefix + image_basename)

                    if out_file_path.exists() and not force_overwrite:
                        print(f'Skipping file "{out_file_path.name}" - file already exists...')
                        continue
                    processed.save(out_file_path, "PNG", exif=exif)

        except OSError as exc:
            # TODO: Add builtin VTF file handling
            sys.stderr.write(f'Could not process steganography for file "{image_file_path.name}"\n')
            sys.stderr.write("\t" + str(exc) + "\n")
