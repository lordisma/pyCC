import pytest as test
from clustlib.fuzzy.lcvqe import LCVQE
import numpy as np

@test.fixture
def define_lcvqe():
    constraints = np.array([
        [ 0,   1,  -1], 
        [ 1,   0,  -1], 
        [-1,  -1,   0]
        ])
    lcvqe = LCVQE(constraints=constraints, n_clusters=2, init="random", max_iter=10)
    yield lcvqe
    del lcvqe

def test_lcvqe_initialization(define_lcvqe):
    lcvqe = define_lcvqe
    assert lcvqe.n_clusters == 2
    assert lcvqe.init == "random"
    assert lcvqe.max_iter == 10
    assert lcvqe.centroids is None

def test_lcvqe_closest_centroid(define_lcvqe):
    lcvqe = define_lcvqe
    lcvqe.centroids = np.array([[0, 1], [2, 0], [-1, -1]])
    instance = np.array([0.5, 0.5])

    closest_centroid, distance = lcvqe._get_closest_centroid(instance)
    assert closest_centroid == 0 
    assert np.isclose(distance, np.linalg.norm(instance - lcvqe.centroids[0]))

def test_lcvqe_get_ml(define_lcvqe):
    lcvqe = define_lcvqe
    ml = lcvqe.get_ml_constraints()
    expected = np.array([
        [0, 1],
        [1, 0],
    ])

    assert np.all(expected == ml), "Must-link cases do not match expected values"

def test_lcvqe_get_cl(define_lcvqe):
    lcvqe = define_lcvqe
    cl = lcvqe.get_cl_constraints()
    expected = np.array([
        [0, 2],
        [1, 2],
        [2, 0],
        [2, 1]
    ])

    assert np.all(expected == cl), "Cannot-link cases do not match expected values"

def test_lcvqe_check_ml(define_lcvqe):
    lcvqe = define_lcvqe
    lcvqe.centroids = np.array([[0, 1], [2, 0], [-1, -1]])
    X = np.array([[0.5, 0.5], [2.5, 0.5], [-1, -1]])

    lcvqe.X = X
    lcvqe._labels = np.array([0, 1, 0])
    lcvqe._check_ml_cases()

    assert lcvqe._labels[0] == lcvqe._labels[1], "Labels for must-link cases do not match"

    lcvqe._labels = np.array([0, 1, 1])
    lcvqe.centroids = np.array([[0, 20], [20, 0], [-1, -1]])
    X = np.array([[0, 20], [20, 0], [-1, -1]])
    lcvqe.X = X

    assert lcvqe.must_link_violations[0][1] == 0, "Must-link violation not detected"
    lcvqe._check_ml_cases()

    assert lcvqe._labels[0] != lcvqe._labels[1], "Labels for must-link cases do not match after violation check"
    assert lcvqe.must_link_violations[0][1] == 1, "Must-link violation detected after check"

def test_lcvqe_check_cl(define_lcvqe):
    lcvqe = define_lcvqe
    lcvqe.centroids = np.array([[0, 1], [2, 0], [-1, -1]])
    X = np.array([[0.5, 0.5], [2.5, 0.5], [-1, -1]])

    lcvqe.X = X
    lcvqe._labels = np.array([0, 1, 0])
    lcvqe._check_ml_cases()

    assert lcvqe._labels[0] == lcvqe._labels[1], "Labels for must-link cases do not match"

    lcvqe._labels = np.array([0, 1, 1])
    lcvqe.centroids = np.array([[0, 20], [20, 0], [-1, -1]])
    X = np.array([[0, 20], [20, 0], [-1, -1]])
    lcvqe.X = X

    assert lcvqe.must_link_violations[0][1] == 0, "Must-link violation not detected"
    lcvqe._check_ml_cases()

    assert lcvqe._labels[0] != lcvqe._labels[1], "Labels for must-link cases do not match after violation check"
    assert lcvqe.must_link_violations[0][1] == 1, "Must-link violation detected after check"

def test_lcvqe_predict(define_lcvqe):
    lcvqe = define_lcvqe
    X = np.array([[0, 1], [1, 0], [-1, -1]])
    lcvqe.fit(X)

    predictions = lcvqe.predict(X)
    assert len(predictions) == len(X)
    assert np.all(np.isin(predictions, [0, 1]))

def test_get_valid_centroids(define_lcvqe):
    lcvqe = define_lcvqe
    X = np.array([[0, 1], [1, 0], [-1, -1]])
    lcvqe.fit(X)

    valid_centroids = lcvqe.get_centroids(0)
    assert len(valid_centroids) == 1