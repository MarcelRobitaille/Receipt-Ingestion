# Receipt ingestion

My attempt to automate receipt ingestion and sharing to [splitwise.com](https://secure.splitwise.com)
requiring only that you take a picture of your receipt.
This is very much made for me, and may not work for your situation.
For example, it assumes that there are only two people in the group.
You are welcome to try :).

[Read the accompanying blog post for implementation details.](https://blog.marcelrobitaille.me/receipt-ingestion/)

## Setup

I am using Python 3.10.3, but older versions will probably work as well.

```
git clone https://github.com/MarcelRobitaille/Receipt-Ingestion.git
cd Receipt-Ingestion
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Credentials and environment variables

I am using environment variables and a `.env` file to manage secrets.
You must find your session ID and CSRF token by sniffing the request when sending the splitwise.com form.

Put all the cookies in the variable `COOKIE`.
Put the CSRF token in the variable `CSRF_TOKEN`.
Put the group, user 1, and user 2 IDs (you can also find these in the sniffed request) in the variables (`GROUP_ID`, `USER_0_ID`, and `USER_1_ID`).
Put the last 4 digits of the credit card of the non-default user (it will be assumed that the default user paid if these 4 do not appear in the receipt)
in the variable `CARD_LAST_FOUR_DIGITS`.
You can specify the directory for the `watch` command ([see below](#watch-a-directory)) with the `WATCH_DIR` variable.

## Usage

There are two main commands:

```
$ python main.py
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  process-file
  watch
```

### Process a single file

To try to extract a receipt from a single file, use the following command:
```
python main.py process-file <filename.jpg>
```

Usage:
```
Usage: main.py process-file [OPTIONS] FILENAME

Options:
  --help  Show this message and exit.
```

### Watch a directory

Since my phone's camera roll syncs to a folder on my computer,
I like to just keep this code running in the background to automatically import the receipts from any pictures that show up in this folder.
This means all I have to do to import a receipt is to take a picture of it.

This command will watch the folder specified by the environment variable `WATCH_DIR`.
You can define this in `.env`.

Usage:
```
Usage: main.py watch [OPTIONS]

Options:
  --help  Show this message and exit.
```
