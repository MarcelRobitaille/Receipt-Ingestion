import click

from process_image import process_image


@click.group()
def main():
    pass


@main.command()
@click.argument('filename', required=True, type=click.Path())
def process_file(filename):
    process_image(filename)


if __name__ == '__main__':
    main()
