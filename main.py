import re
from datetime import datetime, timezone, timedelta
import math
from pathlib import Path

import numpy as np
from dateutil import parser
import requests
from imutils.perspective import four_point_transform
import pytesseract
import imutils
import cv2
from environs import Env

from utils import pairwise, line_intersection


# %%


env = Env()
env.read_env()

out = Path('./out')
out.mkdir(parents=True, exist_ok=True)

# filename = out / datetime.now().strftime('%Y-%m-%d_%H:%M:%S.jpg')
# filename = Path('/home/marcel/Pictures/Camera roll/IMG_20220310_223121.jpg')
filename = out / './IMG_20220325_232924.jpg'
# filename = Path('/tmp/IMG_20220323_142100.jpg')
# filename = Path('/tmp/instructions_right.jpg')


# %%

orig = cv2.imread(str(filename))
if orig.shape[1] > orig.shape[0]:
    orig = cv2.rotate(orig, cv2.cv2.ROTATE_90_CLOCKWISE)
image = orig.copy()
print(image.shape)
image = imutils.resize(image, width=500)
ratio = orig.shape[1] / float(image.shape[1])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
edged = cv2.Canny(blurred, 50, 100)
cv2.imwrite(str(out / f'{filename.stem}_01_edges.jpg'), edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# for i, c in enumerate(cnts):
#     output = image.copy()
#     cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
#     cv2.imwrite(str(str(out / f'{filename.stem}_02_contour_{i}.jpg')), output)

receiptCnt = None
# loop over the contours
for i, c in enumerate(cnts):
    # approximate the contour
    # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    peri = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    approx = cv2.approxPolyDP(c, 0.003 * peri, True)
    # if our approximated contour has four points, then we can
    # assume we have found the outline of the receipt
    while len(approx) > 4:
        print(f'Approximated polygon has more than four ({len(approx)}) edges. '
              'Trying to simplify corners.')
        output = image.copy()
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
        cv2.imwrite(
            str(out / f'{filename.stem}_02_too_many_edges_{len(approx)}.jpg'),
            output,
        )
        lines = approx.reshape(5, 2).tolist()
        lines = list(pairwise(lines + [lines[0]]))
        j = np.argmin(
            [math.hypot(x1 - x2, y1 - y2) for ((x1, y1), (x2, y2)) in lines])
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
cv2.imwrite(str(str(out / f'{filename.stem}_02_contour.jpg')), output)
# apply a four-point perspective transform to the *original* image to
# obtain a top-down bird's-eye view of the receipt
receipt = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)

transformed_filename = str(out / f'{filename.stem}_03_transformed.jpg')
cv2.imwrite(
    transformed_filename,
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
cv2.imwrite(
    str(out / f'{filename.stem}_04_bw.jpg'),
    # imutils.resize(bw, width=500),
    bw,
)

# %%

# https://guides.nyu.edu/tesseract/usage

# apply OCR to the receipt image by assuming column data, ensuring
# the text is *concatenated across the row* (additionally, for your
# own images you may need to apply additional processing to cleanup
# the image, including resizing, thresholding, etc.)
options = '--psm 4 -l eng'
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


def get_total_advanced(data):
    df = data.copy()
    total = df.dropna()[df.dropna().text.apply(lambda x: x.lower()) == 'total']
    if not len(total):
        return None
    total = total.iloc[0]
    top = total.top + total.height
    bottom = df[df.top > top].top.min()
    total_im = ocrimg[top:bottom, :]
    cv2.imwrite(
        str(out / f'{filename.stem}_06_total.jpg'),
        total_im,
    )
    text = pytesseract.image_to_string(total_im, config='--psm 12')
    total = re.search(r'\$?(\d+\.\d+)', text)
    if not total:
        return None
    return float(total.group(1))


def get_total():
    for total in re.finditer(
        r'\b(?:total|kreditkarte|total amount|[mh]astercard)'
        r'[\sa-z]*(?:ca[ds]{1,2}|)[\s\$\'§=:]*([0-9\., :]+)',
        text,
        re.IGNORECASE,
    ):
        print(total)
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

    return get_total_advanced()


total = get_total()


def get_time_posibilities():
    for sep in (':', r'\s', r'\''):
        for time in re.finditer(rf'\d{{2}}{sep}+\d{{2}}{sep}+\d{{2}}', text):
            print('get_time', time)
            yield re.sub(r'[:\s\']+', ':', time.group(0)), time

    # Sometimes, PM is detected as PH by OCR
    for time in re.finditer(r'\d{2}:\d{2}(am|pm|ph)', text, re.IGNORECASE):
        print('get_time', time)
        yield time.group(0).replace('ph', 'pm'), time


def get_closest_time(target):
    """
    Get the time closest in the text to the date
    """

    time, _ = min(
        get_time_posibilities(),
        key=lambda match: abs(match[1].start() - target.start()),
    )

    return time


def get_date():
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
        time = get_closest_time(date)
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
        print('lalalalalal')
        time = get_closest_time(date)
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
        time = get_closest_time(date)
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


date = get_date()


def get_paid_by():
    last_four_digits: str = env('CARD_LAST_FOUR_DIGITS')
    # Eights sometimes come across as B in OCR
    last_four_digits = last_four_digits.replace('8', '[8B]')
    if re.search(last_four_digits, text):
        return 'Marcel'
    return 'Federica'


paid_by = get_paid_by()


def get_store():
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

    return text.split('\n')[0]


store = get_store()
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
    'User-Agent':
        'Mozilla/5.0 (X11; Linux x86_64; rv:97.0) Gecko/20100101 Firefox/97.0',
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
    files=[(
        'receipt',
        (transformed_filename, open(transformed_filename, 'rb'), 'image/jpeg'),
    )],
)
print(r)
