from clustlib.constraints import ConstraintMatrix

import numpy as np
import unittest as utest


class TestMatrix(utest.TestCase):
    @utest.skip("Not implemented")
    def test_initialization(self):
        test = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).view(ConstraintMatrix)

        assert isinstance(test, np.ndarray)
        assert test.shape == (3, 3)
        assert test.diagonal().tolist() == [0, 0, 0]

    @utest.skip("Not implemented")
    def test_setitem(self):
        test = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).view(ConstraintMatrix)

        test[1, 1] = 1.0
        test[1, 0] = -1.0

        assert test[1, 1] == 1.0
        assert test[1, 0] == -1.0

    @utest.skip("Not implemented")
    def test_new(self):
        test = ConstraintMatrix(shape=(3,), dtype=float, order="F")
        assert test.shape == (3, 3)

        with self.assertRaises(ValueError) as context:
            test = ConstraintMatrix((3, 4))

        message = "shape should be a only index, or two equals dimensions"
        assert message == str(context.exception)

        with self.assertRaises(ValueError) as context:
            test = ConstraintMatrix((3, 1, 2))

        assert "shape should be a only index, or two equals dimensions" == str(
            context.exception
        )

        test = ConstraintMatrix(3)
        assert test.shape == (3, 3)

    @utest.skip("Not implemented")
    def test_set(self):
        test = ConstraintMatrix(3)
        test[0, 1] = 1.0
        test[0, 2] = -1.0

        assert test[2, 0] == -1.0, "The matrix should be symetrical"
        assert test[1, 0] == 1.0, "The matrix should be symetrical"

        test = ConstraintMatrix(3)
        test.fill(1.0)
        test[0, 1] = 1.0

        assert test[0, 1] == 1.0, "The value should be normalized"
        assert test[1, 0] == 1.0, "The value should be normalized"
