"""dummy

This is a dummy module to test the clustlib package.
"""


def fib(num: int) -> int:
    """fib

    Args:
        num (int): number to calculate fibonacci for
    Returns:
        int: fibonacci number
    """
    if num == 0:
        return 0
    if num == 1:
        return 1
    if num > 0:
        return num + fib(num - 1)

    raise ValueError("num must be positive")


def untested() -> int:
    """untested

    Returns:
        int: 0
    """
    print("I do nothing :{")
    return 0
