from clustlib import metrics
from clustlib.constraints import matrix as mtx

import numpy as np
import unittest as utest


class TestMatrix(utest.TestCase):
    @utest.skip("Not implemented")
    def test_violated_constraints(self):
        constraints = np.array(
            [
                [0, 1, -1, 0, 0, 0],
                [1, 0, 0, -1, -1, 0],
                [-1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 1, 1],
                [0, -1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        ).view(mtx.ConstraintMatrix)
        result = metrics.violates_constraints([0, 2, 3], constraints)

        assert result == 4

    @utest.skip("Not implemented")
    def test_all_must_link(self):
        constraints = np.array(
            [
                [0, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 0],
            ]
        )
        result = metrics.violates_constraints([0, 1, 2, 3, 4], constraints)

        assert result == 5

    @utest.skip("Not implemented")
    def test_all_cant_link(self):
        constraints = np.array(
            [
                [0, -1, -1, -1, -1, -1],
                [-1, 0, -1, -1, -1, -1],
                [-1, -1, 0, -1, -1, -1],
                [-1, -1, -1, 0, -1, -1],
                [-1, -1, -1, -1, 0, -1],
                [-1, -1, -1, -1, -1, 0],
            ]
        )
        result = metrics.violates_constraints([0, 1, 2], constraints)

        assert result == 3

    @utest.skip("Not implemented")
    def test_infeasibility(self):
        # index:                  0   1   2   3   4   5
        constraints = np.array(
            [
                [0, -1, -1, -1, -1, -1],  # 0
                [-1, 0, -1, -1, -1, -1],  # 1
                [-1, -1, 0, -1, -1, -1],  # 2
                [-1, -1, -1, 0, -1, -1],  # 3
                [-1, -1, -1, -1, 0, -1],  # 4
                [-1, -1, -1, -1, -1, 0],
            ]
        )  # 5

        result = metrics.infeasibility([[1, 2], [3, 4], [0, 5]], constraints)
        assert result == 3
