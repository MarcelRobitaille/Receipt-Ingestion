import re
from itertools import tee
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
import sane


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# %%


out = Path('./out')
out.mkdir(parents=True, exist_ok=True)

# filename = out / datetime.now().strftime('%Y-%m-%d_%H:%M:%S.jpg')
# filename = Path('/home/marcel/Pictures/Camera roll/IMG_20220310_223121.jpg')
filename = out / './IMG_20220325_232924.jpg'
# filename = Path('/tmp/IMG_20220323_142100.jpg')
# filename = Path('/tmp/instructions_right.jpg')
# %%

sane.init()
try:
    dev = sane.open('airscan:e1:Brother MFC-L3750CDW series')
    dev.source = 'ADF'
    dev.br_y = 5000
    print(dev.opt)
    dev.start()
    im = dev.snap()
    im.save(filename)
    print(filename)
finally:
    dev.close()


# %%

orig = cv2.imread(str(filename))
if orig.shape[1] > orig.shape[0]:
    orig = cv2.rotate(orig, cv2.cv2.ROTATE_90_CLOCKWISE)
image = orig.copy()
print(image.shape)
image = imutils.resize(image, width=500)
ratio = orig.shape[1] / float(image.shape[1])

# convert the image to grayscale, blur it slightly, and then apply
# edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
edged = cv2.Canny(blurred, 50, 100)
# check to see if we should show the output of our edge detection
# procedure
# if args["debug"] > 0:
    # cv2.imshow("Input", image)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
cv2.imwrite(str(out / f'{filename.stem}_01_edges.jpg'), edged)

# find contours in the edge map and sort them by size in descending
# order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# c = cnts[0]
# peri = cv2.arcLength(c, True)
# approx = cv2.approxPolyDP(c, 0.02 * peri, True)
# print(approx)
# %%
# receiptCnt = cnts[0]
# initialize a contour that corresponds to the receipt outline
receiptCnt = None
# loop over the contours
for i, c in enumerate(cnts):
    # approximate the contour
    # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    peri = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    approx = cv2.approxPolyDP(c, 0.006 * peri, True)
    # if our approximated contour has four points, then we can
    # assume we have found the outline of the receipt
    while len(approx) > 4:
        print(len(approx), area)
        print(approx)
        print(np.shape(approx))
        lines = approx.reshape(5, 2).tolist()
        lines = list(pairwise(lines + [lines[0]]))
        print([math.hypot(x1 - x2, y1 - y2) for ((x1, y1), (x2, y2)) in lines])
        j = np.argmin([math.hypot(x1 - x2, y1 - y2) for ((x1, y1), (x2, y2)) in lines])
        shortest = lines[j]
        before = lines[(j - 1) % len(lines)]
        after = lines[(j + 1) % len(lines)]
        print(before, after)
        print(line_intersection(before, after))
        print(shortest)
        print(approx)
        approx = np.delete(approx, j, axis=0)
        approx[j, 0, :] = line_intersection(before, after)
        print(approx)
    # np.delete(approx, j)
    # approx.insert(j, line_intersection(before, after))
    # print(approx)
    # shortest = lines.pop(j)
    # print(lines)
    print(j)
    # lines = [x for x in approx.reshape(5, 2)]

    if len(approx) == 4 or len(approx) == 5:
        receiptCnt = approx
        print('chose', i)
        break
# if the receipt contour is empty then our script could not find the
# outline and we should be notified
if receiptCnt is None:
    raise Exception(("Could not find receipt outline. "
        "Try debugging your edge detection and contour steps."))

# check to see if we should draw the contour of the receipt on the
# image and then display it to our screen
# if args["debug"] > 0:
    # output = image.copy()
    # cv2.drawContours(output, [receiptCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Receipt Outline", output)
    # cv2.waitKey(0)
output = image.copy()
cv2.drawContours(output, [receiptCnt], -1, (0, 255, 0), 2)
cv2.imwrite(str(str(out / f'{filename.stem}_02_contour.jpg')), output)
# apply a four-point perspective transform to the *original* image to
# obtain a top-down bird's-eye view of the receipt
receipt = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)
# show transformed image
# cv2.imshow("Receipt Transform", imutils.resize(receipt, width=500))
# cv2.waitKey(0)

transformed_filename = str(out / f'{filename.stem}_03_transformed.jpg')
cv2.imwrite(
    transformed_filename,
    imutils.resize(receipt, width=500),
)

# bw = receipt.copy()
# %%
gray = cv2.cvtColor(receipt, cv2.COLOR_BGR2GRAY)
# thresh, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY, 11, 5)
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
# print(data.to_string())
# height = annotated.shape[0]
# annotated = bw.copy()
# for box in boxes.strip().split('\n'):
#     box = box.split(' ')[1:-1]
#     box = [int(b) for b in box]
#     a, b, c, d = box
#     annotated = cv2.rectangle(
#         annotated,
#         (a, height - b),
#         (c, height - d),
#         (0, 255, 0),
#         1,
#     )
# cv2.imwrite(
#     str(out / f'{filename.stem}_05_ocr.jpg'),
#     annotated,
# )
# show the raw output of the OCR process
print('[INFO] raw output:')
print('==================')
print(text)
print('\n')


with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(text)


# %%

with open('output.txt', 'r', encoding='utf-8') as f:
    text = f.read()


def get_total_advanced():
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
    total = re.search(
        r'\b(?:total|kreditkarte|total amount)[\sa-z]*(?:ca[ds]{1,2}|)[\s\$\'§=:]*([0-9\., :]+)',
        text,
        re.IGNORECASE,
    )
    print(total)
    if total is None:
        return get_total_advanced()
    total = total.group(1)
    total = float(re.sub(r'[ :]', '', total))
    total = int(total * 100)
    total = float((total + total % 2) / 100)
    return total


total = get_total()


def get_time_posibilities():
    for sep in (':', r'\s', r'\''):
        for time in re.finditer(rf'\d{{2}}{sep}+\d{{2}}{sep}+\d{{2}}', text):
            print('get_time', time)
            yield re.sub(r'[:\s\']+', ':', time.group(0)), time

    for time in re.findall(r'\d{2}:\d{2}(am|pm)', text, re.IGNORECASE):
        print('get_time', time)
        yield time.group(0), time


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

    for date in re.findall(
        date_regex,
        text,
        re.IGNORECASE,
    ):
        # TODO
        print(date)
        time = get_closest_time(date)
        date = date.lower().replace('marz', 'march').replace('-', ' ')
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

    return now.strftime('%Y-%m-%dT%H:%M:%S%z')


date = get_date()


def get_paid_by():
    if re.search(r'hihi', text):
        return 'Marcel'
    return 'Federica'


paid_by = get_paid_by()


def get_store():
    if re.search('IF YOU DRINK, DON\'?T DRIVE', text):
        return 'LCBO'
    if re.search('GLENBRIAR', text):
        return 'Glenbriar Home Hardware'
    if store := re.search(r'(zehrs)', text, re.IGNORECASE):
        return store.group(0)

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
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:97.0) Gecko/20100101 Firefox/97.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://secure.splitwise.com/',
    'X-CSRF-Token': 'not so fast',
    # 'Content-Type': 'application/x-www-form-urlencoded',

    # 'Content-Type': 'multipart/form-data',
    # 'X-Requested-With': 'XMLHttpRequest',
    # 'Origin': 'https://secure.splitwise.com',
    # 'DNT': '1',
    # 'Connection': 'keep-alive',
    'Cookie': 'not so fast',
    # 'Sec-Fetch-Dest': 'empty',
    # 'Sec-Fetch-Mode': 'cors',
    # 'Sec-Fetch-Site': 'same-origin',
    # 'Pragma': 'no-cache',
    # 'Cache-Control': 'no-cache',
    },
    data=data,
    files=[('receipt', (transformed_filename, open(transformed_filename, 'rb'), 'image/jpeg'))],
)
print(r)
