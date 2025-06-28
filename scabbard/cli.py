"""Console script for scabbard."""
import sys
import click


@click.command()
def main(args=None):
    """
    Console script for scabbard.

    This function serves as the entry point for the scabbard command-line interface.
    It currently provides a placeholder message and directs users to the Click documentation.

    Args:
        args (list, optional): Command-line arguments passed to the script. Defaults to None.

    Returns:
        int: Exit code of the program (0 for success).

    Author: B.G.
    """
    click.echo("Replace this message by putting your code into "
               "scabbard.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
