import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
import math

import numpy as np
from dateutil import parser
import requests
from imutils.perspective import four_point_transform
import pytesseract
import imutils
import cv2

from utils import pairwise, line_intersection
from perspective_transform_by_qr import perspective_transform_by_qr
from constants import env, DEBUG, DEBUG_IMAGE_DIR


# %%


def get_total_advanced(data, text, ocrimg, filename: Path):
    df = data.copy()
    total = df.dropna()[df.dropna().text.apply(lambda x: x.lower()) == 'total']
    # TODO
    assert len(total)
    total = total.iloc[0]
    top = total.top + total.height
    bottom = df[df.top > top].top.min()
    total_im = ocrimg[top:bottom, :]
    if DEBUG:
        cv2.imwrite(
            str(DEBUG_IMAGE_DIR / filename.name.replace('.jpg', '_10_total.jpg')),
            total_im,
        )
    text = pytesseract.image_to_string(total_im, config='--psm 12')
    total = re.search(r'\$?(\d+\.\d+)', text)
    assert total is not None
    return float(total.group(1))


def get_total(text: str, *args, **kwargs):
    for total in re.finditer(
        r'\b(?:total|kreditkarte|total amount|[mh]astercard)'
        r'[\sa-z]*(?:ca[ds]{1,2}|)[\s\$\'§=:]*([0-9\., :]+)',
        text,
        re.IGNORECASE,
    ):
        print(total)
        if re.search('tax', total.group(0), re.IGNORECASE):
            print('Skipping found total with tax', total.group(0))
            continue
        try:
            total = total.group(1)
            total = total.replace(',', '.')
            if len(total) >= 5:
                total = total.replace(' ', '.')
            total = float(re.sub(r'[ :]', '', total))
            total = int(total * 100)
            total = float((total + total % 2) / 100)
            return total
        except ValueError:
            continue

    return get_total_advanced(text=text, *args, **kwargs)


def get_time_posibilities(text: str):
    for sep in (':', r'\s', r'\''):
        for time in re.finditer(rf'\d{{2}}{sep}+\d{{2}}{sep}+\d{{2}}', text):
            print('get_time', time)
            yield re.sub(r'[:\s\']+', ':', time.group(0)), time

    # Sometimes, PM is detected as PH by OCR
    for time in re.finditer(r'\d{2}:\d{2}(am|pm|ph|)', text, re.IGNORECASE):
        print('get_time', time)
        yield time.group(0).replace('ph', 'pm'), time


def get_closest_time(target, text: str):
    """
    Get the time closest in the text to the date
    """

    time, _ = min(
        get_time_posibilities(text=text),
        key=lambda match: abs(match[1].start() - target.start()),
    )

    return time


def get_date(text: str):
    # print(text)
    timezone_offset = -5.0
    now = datetime.now(timezone(timedelta(hours=timezone_offset)))
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
              'OCT', 'NOV', 'DEC', 'marz']

    date_regex = rf'(\d{{2}}[\s-]+(?:{"|".join(months)})[\s-]+\d{{4}})'

    for date in re.finditer(
        date_regex,
        text,
        re.IGNORECASE,
    ):
        # TODO
        print(date)
        time = get_closest_time(date, text=text)
        date = date.group(0).lower().replace('marz', 'march').replace('-', ' ')
        # print(date)
        # date = parser.parse(f'{date} {time}')
        date = parser.parse(f'{date}')
        if date > datetime.now():
            continue
        return date

    for date in re.finditer(
        r'(\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2})',
        text,
    ):
        print(date)
        time = get_closest_time(date, text=text)
        print(time)
        date = date.group(0).replace('/', '-')
        print(date)
        year, month, date = date.split('-')
        if int(month) > 20:
            _, c = month
            month = f'0{c}'
        if int(month) > 12:
            month, date, year = year, month, date
        if len(year) == 2:
            year, date = date, year
            month, date = date, month

        # Sometimes, 0 looks like 6 or 8
            # month[0] = '0'
        # if len(date.split('-')[0]) == 2:
        #     date = '-'.join(date.split('-')[::-1])
        date = parser.parse(f'{year}-{month}-{date}T{time}')
        if date > datetime.now():
            continue
        return date

    for date in re.finditer(r'(\d{2}[-/]\d{2}[-/]\d{2})', text):
        print(date)
        time = get_closest_time(date, text=text)
        print(time)
        date = date.group(0).replace('/', '-')
        print(date)
        year, month, date = date.split('-')
        year = f'20{year}'
        if int(month) > 20:
            _, c = month
            month = f'0{c}'
        if int(month) > 12:
            month, date, year = year, month, date
        if len(year) == 2:
            year, date = date, year
            month, date = date, month

        # Sometimes, 0 looks like 6 or 8
            # month[0] = '0'
        # if len(date.split('-')[0]) == 2:
        #     date = '-'.join(date.split('-')[::-1])
        date = parser.parse(f'{year}-{month}-{date}T{time}')
        if date > datetime.now():
            continue
        return date

    return now.strftime('%Y-%m-%dT%H:%M:%S%z')


def get_paid_by(text: str):
    last_four_digits: str = env('CARD_LAST_FOUR_DIGITS')
    # Eights sometimes come across as B in OCR
    last_four_digits = re.sub('[38]', '[38B]', last_four_digits)
    if re.search(last_four_digits, text):
        return 'Marcel'
    return 'Federica'


def get_store(text: str):
    if re.search(r'(IF YOU DRINK, DON\'?T DRIVE|SI VOUS BUVEZ|'
                 r'NE PRENEZ PAS LE VOLANT)', text):
        return 'LCBO'
    if re.search('GLENBRIAR', text):
        return 'Glenbriar Home Hardware'
    if store := re.search(r'(zehrs)', text, re.IGNORECASE):
        return store.group(0)
    if re.search(r'(shoppers|drug mart)', text, re.IGNORECASE):
        return 'Shoppers Drug Mart'
    if re.search(r'(zehrs|zenrs)', text, re.IGNORECASE):
        return 'Zehrs'
    if re.search(r'Sobeys', text, re.IGNORECASE):
        return 'Sobeys'
    if re.search(r'JINZAKAYA', text, re.IGNORECASE):
        return 'JINZAKAYA'
    if re.search(r'Indigo', text, re.IGNORECASE):
        return 'Indigo'
    if re.search(r'Miniso', text, re.IGNORECASE):
        return 'Miniso'

    return text.split('\n')[0]


def process_image(filename: Path):
    # TODO
    assert str(filename).endswith('.jpg')

    # TODO: It's not really orig anylonger
    orig = perspective_transform_by_qr(filename)
    orig = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    if orig.shape[1] > orig.shape[0]:
        orig = cv2.rotate(orig, cv2.cv2.ROTATE_90_CLOCKWISE)
    image = orig.copy()
    print(image.shape)
    image = imutils.resize(image, width=500)
    ratio = orig.shape[1] / float(image.shape[1])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
    edged = cv2.Canny(blurred, 55, 150)
    if DEBUG:
        cv2.imwrite(
            DEBUG_IMAGE_DIR /
            filename.name.replace('.jpg', '_04_edges.jpg'),
            edged,
        )

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    if DEBUG:
        output = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, c in enumerate(cnts):
            cv2.putText(output, str(i), (c[:, :, 0].min(), c[:, :, 1].min()),
                        font, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
        cv2.imwrite(
            str(DEBUG_IMAGE_DIR /
                filename.name.replace('.jpg', f'_05_all_contours.jpg')),
            output,
        )
    # for i, c in enumerate(cnts):
    #     output = image.copy()
    #     cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
    #     if DEBUG:
    #         cv2.imwrite(
    #             str(DEBUG_IMAGE_DIR /
    #             filename.name.replace('.jpg', f'_05_contour_{i}.jpg')),
    #             output,
    #         )


    receiptCnt = None
    # loop over the contours
    for i, c in enumerate(cnts):
        # approximate the contour
        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.003 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the receipt
        while len(approx) > 4:
            print('Approximated polygon has more than four '
                  f'({len(approx)}) edges. Trying to simplify corners.')
            output = image.copy()
            cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
            if DEBUG:
                cv2.imwrite(
                    str(DEBUG_IMAGE_DIR / filename.name
                    .replace('.jpg', f'_06_too_many_edges_{len(approx)}.jpg')),
                    output,
                )
            lines = approx.reshape(5, 2).tolist()
            lines = list(pairwise(lines + [lines[0]]))
            j = np.argmin([
                math.hypot(x1 - x2, y1 - y2)
                for ((x1, y1), (x2, y2)) in lines
            ])
            shortest = lines[j]
            before = lines[(j - 1) % len(lines)]
            after = lines[(j + 1) % len(lines)]
            approx = np.delete(approx, j, axis=0)
            approx[j, 0, :] = line_intersection(before, after)
            print(approx)

        if len(approx) == 4 or len(approx) == 5:
            receiptCnt = approx
            print('chose', i)
            break
    # if the receipt contour is empty then our script could not find the
    # outline and we should be notified
    if receiptCnt is None:
        raise Exception(
            'Could not find receipt outline. '
            'Try debugging your edge detection and contour steps.')

    output = image.copy()
    cv2.drawContours(output, [receiptCnt], -1, (0, 255, 0), 2)
    if DEBUG:
        cv2.imwrite(
            str(DEBUG_IMAGE_DIR /
                filename.name.replace('.jpg', '_07_contour.jpg')),
            output,
        )
    # apply a four-point perspective transform to the *original* image to
    # obtain a top-down bird's-eye view of the receipt
    receipt = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)

    transformed_filename = DEBUG_IMAGE_DIR / \
        filename.name.replace('.jpg', '_08_transformed.jpg')
    if DEBUG:
        cv2.imwrite(
            str(transformed_filename),
            imutils.resize(receipt, width=500),
        )

    # bw = receipt.copy()
    # %%
    gray = cv2.cvtColor(receipt, cv2.COLOR_BGR2GRAY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 6)
    if DEBUG:
        cv2.imwrite(
            str(DEBUG_IMAGE_DIR /
                filename.name.replace('.jpg', '_09_bw.jpg')),
            bw,
        )

    # %%

    # https://guides.nyu.edu/tesseract/usage

    # apply OCR to the receipt image by assuming column data, ensuring
    # the text is *concatenated across the row* (additionally, for your
    # own images you may need to apply additional processing to cleanup
    # the image, including resizing, thresholding, etc.)
    options = '--psm 11 -l eng'
    ocrimg = bw
    text = pytesseract.image_to_string(ocrimg, config=options)
    # boxes = pytesseract.image_to_boxes(
    #     bw,
    #     config=options)
    data = pytesseract.image_to_data(ocrimg, config=options,
                                     output_type=pytesseract.Output.DATAFRAME)
    print(text)

    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    # %%

    with open('output.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    total = get_total(text=text, data=data, ocrimg=ocrimg, filename=filename)
    date = get_date(text=text)
    paid_by = get_paid_by(text=text)
    store = get_store(text=text)

    print('Store:', store)
    print('Paid by:', paid_by)
    print('Date:', date)

    paid_shares = {
        'Federica': [0, total],
        'Marcel': [total, 0],
    }[paid_by]
    print('Total:', total)

    data = {
            'cost': f'{total:.2f}',
            'currency_code': 'CAD',
            'group_id': '123',
            'users__0__user_id': '123',
            'users__0__owed_share': f'{total / 2:.2f}',
            'users__1__owed_share': f'{total / 2:.2f}',
            'users__1__user_id': '123',
            'users__0__paid_share': f'{paid_shares[0]:.2f}',
            'users__1__paid_share': f'{paid_shares[1]:.2f}',
            'category_id': '18',
            'date': str(date),
            'description': store,
            'creation_method': 'equal',
        }
    print(data)
    # %%

    r = requests.post(
        'https://secure.splitwise.com/api/v3.0/create_expense',
        headers={
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:97.0) '
            'Gecko/20100101 Firefox/97.0',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://secure.splitwise.com/',
        'X-CSRF-Token': env('CSRF_TOKEN'),
        # 'Content-Type': 'application/x-www-form-urlencoded',

        # 'Content-Type': 'multipart/form-data',
        # 'X-Requested-With': 'XMLHttpRequest',
        # 'Origin': 'https://secure.splitwise.com',
        # 'DNT': '1',
        # 'Connection': 'keep-alive',
        'Cookie': env('COOKIE'),
        # 'Sec-Fetch-Dest': 'empty',
        # 'Sec-Fetch-Mode': 'cors',
        # 'Sec-Fetch-Site': 'same-origin',
        # 'Pragma': 'no-cache',
        # 'Cache-Control': 'no-cache',
        },
        data=data,
        files=[('receipt', (
            transformed_filename,
            open(transformed_filename, 'rb'),
            'image/jpeg',
        ))],
    )
    print(r)
