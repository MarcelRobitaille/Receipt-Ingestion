from pathlib import Path
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont


def find_coeffs(pa, pb):
    '''
    Find coefficients for perspective transformation.
    From http://stackoverflow.com/a/14178717/4414003.
    '''
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def transform(startpoints, endpoints, im):
    '''
    Perform a perspective transformation on an image where startpoints are moved
    to endpoints, and the image is stretched accordingly.
    '''

    # To try to keep the receipt inside the image during perspective
    # transform, output a larger image than the input and shift the QR target
    # towards the bigger image's center. This is effectively the same as
    # making the input image bigger from its center.
    scale = 0.5
    width, height = im.size
    endpoints = np.array(endpoints)
    endpoints[:, 0] += int(width * scale / 2)
    endpoints[:, 1] += int(height * scale / 2)
    coeffs = find_coeffs(endpoints, startpoints)

    # Increase the height slightly in case the rotation makes the receipt go
    # outside of the original image size
    im = im.transform(
        (int(width * (1 + scale)), int(height * (1 + scale))),
        Image.PERSPECTIVE,
        coeffs,
        Image.BICUBIC,
    )
    return im


def perspective_transform_by_qr(filename: Path):
    """
    Straighten input image by the QR code detected in the image. This has two
    huge advantages:
    1. This helps to find the normal vector of the table instead of relying on
       the four corners of the receipt. The latter is super unreliable since
       receipts can be crumpled, not lying flat on the table, or have their
       corners torn when ripping it off of the printer (or some of someone
       else's receipt that got torn in this way)
    2. Instead of developing a website or app to take a picture and sending it
       to the server, I can simply take a photo with my normal camera app. These
       photos are already automatically sent to my Nextcloud, and downloaded to
       my computer. By running this software on my computer and listening for
       new files in this folder, I can automatically analyze any new pictures
       and abort if they don't contain the special "I'm a receipt, analyze me"
       QR code.
    """

    print('perspective_transform_by_qr', filename)
    image = Image.open(filename)
    if image.width > image.height:
        image = image.rotate(-90, Image.NEAREST, expand=True)

    image.save(str(filename).replace('.jpg', '_01_rotated.jpg'))

    code = next(code for code in decode(image) if code.data == b'ingest me\n')

    draw = ImageDraw.Draw(image)
    rect = code.rect
    draw.rectangle(
        (
            (rect.left, rect.top),
            (rect.left + rect.width, rect.top + rect.height)
        ),
        outline='#0080ff',
        width=10,
    )

    draw.polygon(code.polygon, outline='#e945ff', width=10)
    font = ImageFont.truetype('Ubuntu-M.ttf', 200, encoding='unic')
    for i, p in enumerate(code.polygon):
        draw.text(p, str(i), fill='#a00000', font=font)

    image.save(str(filename).replace('.jpg', '_02_qr.jpg'))

    left, top, width, height = code.rect

    transformed = transform(
        startpoints=code.polygon,
        endpoints=(
            (left, top + height),
            (left + width, top + height),
            (left + width, top),
            (left, top),
        ),
        im=image,
    )

    transformed.save(str(filename).replace('.jpg', '_03_transformed.jpg'))

    return transformed
