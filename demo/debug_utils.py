"""
debug_utils.py

Simple debug message utilities that print colored INFO, ERROR, and SUCCESS messages
when DEBUG is enabled.
"""

DEBUG = True

from termcolor import cprint


def debug_info(*args):
    """Prints a yellow `[INFO]` message if debugging is enabled.

    Args:
        *args: Variable length argument list to be converted to strings and
            concatenated into the message.
    """
    if DEBUG:
        cprint(f"[INFO] {' '.join(map(str, args))}", "yellow")


def debug_error(*args):
    """Prints a red `[ERROR]` message if debugging is enabled.

    Args:
        *args: Variable length argument list to be converted to strings and
            concatenated into the message.
    """
    if DEBUG:
        cprint(f"[ERROR] {' '.join(map(str, args))}", "red")


def debug_success(*args):
    """Prints a green `[SUCCESS]` message if debugging is enabled.

    Args:
        *args: Variable length argument list to be converted to strings and
            concatenated into the message.
    """
    if DEBUG:
        cprint(f"[SUCCESS] {' '.join(map(str, args))}", "green")
