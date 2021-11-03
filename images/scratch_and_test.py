from pathlib import Path
from time import time

from PIL import Image
from PIL.Image import Image

from images.steganography_wip import normalized_bit_mask
from infra.utils import chunked


def testing_reduction_transform_to_large_format():
    test_image = Image.open(Path("./images/sources/steganography/hidden.png"))

    def make_special(pix_list, shift_):
        # Add each pixel to the nearest "column" that can be found along the x axis
        columns = tuple([] for _ in range(64))
        pix_iter = iter(pix_list)
        for _ in range(204 * 4):
            for j in range(64):
                xi = ((shift_ + j * 153) % 3264) // 51
                columns[xi].append(next(pix_iter))

        flat_list = []
        for pixel in tuple(index for column in zip(*columns) for index in column):
            flat_list.extend(
                [
                    pixel,
                ]
                * 16
            )

        new_img = Image.new("RGBA", (1024, 816))
        new_img.putdata(flat_list)

        # Calculate amount to shift image to prevent "jumping"
        new_x = 16 - shift_ / 3.1875 % 16

        # Copy to new smaller image to prevent letterbox-ing due to moving canvas
        new_new_img = Image.new("RGBA", (1008, 816))
        # Add some light subpixel processing, why not
        new_new_img.paste(
            new_img.transform(
                (1024, 816),
                Image.AFFINE,
                data=(1.0, 0.0, new_x, 0.0, 1.0, 0.0),
                resample=Image.BICUBIC,
            )
        )

        return new_new_img

    for mask in {1, 3, 7, 15, 31, 63, 127, 255}:
        start = time()
        image_data = tuple(normalized_bit_mask(test_image, mask).getdata())
        gif_list = []

        print("-" * 153)
        for shift in range(0, 153):
            gif_list.append(make_special(image_data[shift::153], shift))
            print(".", end="")
        print()
        gif_list[0].save(
            Path(f"./images/generated/steganography/hidden_list_streshdboi_shifty_mask{mask}.gif"),
            save_all=True,
            append_images=gif_list[1:],
            format="GIF",
            optimize=False,
            duration=60,
            loop=0,
        )

        print(time() - start)


def testing_reduction_transform_to_small_format():
    test_image = Image.open(Path("./images/sources/steganography/hidden.png"))

    def make_special(pix_list, shift_):
        # Every pixel is significant in this one
        # Add each pixel to the nearest "column" that can be found along the x axis
        columns = tuple([] for _ in range(64))
        pix_iter = iter(pix_list)
        for _ in range(204 * 4):
            for j in range(64):
                xi = ((shift_ + j * 153) % 3264) // 51
                columns[xi].append(next(pix_iter))
        flat_tuple = tuple(index for column in zip(*columns) for index in column)

        final_list = []
        for four_row in chunked(flat_tuple, 256):
            for i in range(64):
                final_list.append(four_row[i + 64])
                final_list.append(four_row[i + 192])
                final_list.append(four_row[i + 128])
                final_list.append(four_row[i])

        new_img = Image.new("RGBA", (256, 204))
        new_img.putdata(final_list)

        new_x = 4 - shift_ / 12.75 % 4

        new_new_img = Image.new("RGBA", (252, 204))
        new_new_img.paste(
            new_img.transform(
                (256, 204),
                Image.AFFINE,
                data=(1.0, 0.0, new_x, 0.0, 1.0, 0.0),
                resample=Image.BICUBIC,
            )
        )

        return new_new_img

    for mask in {1, 3, 7, 15, 31, 63, 127, 255}:
        start = time()
        image_data = tuple(normalized_bit_mask(test_image, mask).getdata())
        gif_list = []

        print("-" * 153)
        for shift in range(0, 153):
            gif_list.append(make_special(image_data[shift::153], shift))
            print(".", end="")
        print()
        gif_list[0].save(
            Path(f"./images/generated/steganography/hidden_list_smashedboi_mask{mask}.gif"),
            save_all=True,
            append_images=gif_list[1:],
            format="GIF",
            optimize=False,
            duration=60,
            loop=0,
        )

        print(time() - start)


def testing_transposed_reduction():
    test_image = Image.open(Path("./images/sources/steganography/hidden.png"))

    def make_special(pix_list, shift_):
        # Add each pixel to the nearest "column" that can be found along the x axis
        flat_list = []
        for row_of_8 in chunked(pix_list, 48):
            for i in range(16):
                flat_list.append(row_of_8[i + 0])
                flat_list.append(row_of_8[i + 32])
                flat_list.append(row_of_8[i + 16])

        new_list = []
        for item in flat_list:
            new_list.extend((item,) * 16)

        new_img = Image.new("RGBA", (768, 1088))
        new_img.putdata(new_list)

        # Calculate amount to shift image to prevent "jumping"
        new_x = 48 - shift_ / 153 * 48

        new_new_img = new_img.transform(
            (768, 1088), Image.AFFINE, data=(1.0, 0.0, new_x, 0.0, 1.0, 0.0), resample=Image.BICUBIC
        )
        new_new_new_img = Image.new("RGBA", (720, 1088))
        new_new_new_img.paste(new_new_img)

        return new_new_new_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)

    for mask in {1, 3, 7, 15, 31, 63, 127, 255}:
        start = time()
        trans = normalized_bit_mask(test_image, mask).rotate(270, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        image_data = tuple(trans.getdata())
        gif_list = []
        print("-" * 153)
        for shift in range(153):
            gif_list.append(make_special(image_data[shift::153], shift))
            print(".", end="")
        print()
        # gif_list[0].show()
        gif_list[0].save(
            Path(f"./images/generated/steganography/hidden_list_sidewaysboi_shifty_mask{mask}.gif"),
            save_all=True,
            append_images=gif_list[1:],
            format="GIF",
            optimize=False,
            duration=60,
            loop=0,
        )

        print(time() - start)


# DISTANCE FROM TRUE RED

# test_image = Image.open(Path("./images/sources/villa_painting_002_skin3.png"))
#
# for i in range(test_image.width):
#     for j in range(test_image.height):
#         pixel = test_image.getpixel((i, j))
#         hue = get_hue(*pixel)
#         if hue > 180:
#             new = int((360 - hue) / 180 * 256)
#             test_image.putpixel((i, j), (new, new, new, 255))
#         else:
#             new = int(hue / 180 * 256)
#             test_image.putpixel((i, j), (new, new, new, 255))
# test_image.show()
