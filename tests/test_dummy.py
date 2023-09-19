from clustlib.dummy import fib


def test_dummy_0():
    assert fib(0) == 0


def test_dummy_1():
    assert fib(1) == 1


def test_fib_10():
    assert fib(10) == 55
