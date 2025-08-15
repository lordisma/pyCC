import pytest as test
from clustlib.kmean.rdpmean import RDPM
import numpy as np

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@test.fixture
def define_rdpm():
    constraints = np.array([
        [0, 1, -1, -1], 
        [1, 0, -1, -1], 
        [-1, -1, 0, 1],
        [-1, -1, 1, 0]
    ])
    rdpm = RDPM(constraints=constraints, n_clusters=3, init="random", limit = 2, max_iter=10)
    yield rdpm
    del rdpm

def test_rdpm_initialization(define_rdpm):
    rdpm = define_rdpm
    assert rdpm.n_clusters == 3
    assert rdpm.init == "random"
    assert rdpm.max_iter == 10
    assert rdpm.centroids is None

def test_rdpm_diff_alliances(define_rdpm):
    rdpm = define_rdpm
    rdpm._labels = np.array([0, 0, 1, 1])

    assert rdpm.diff_alliances(0, 0) == 1
    assert rdpm.diff_alliances(0, 1) == -2

def test_rdpm_predict(define_rdpm):
    rdpm = define_rdpm
    X = np.array([[5, 1], [5, 0], [-5, 0.2], [-5, 0.5]])
    rdpm.fit(X)

    predictions = rdpm.predict(X)
    assert len(predictions) == len(X)

    logging.debug(f"Predictions: {predictions}")
    assert np.all(predictions == rdpm._labels)

def test_rdpm_remove_empty_clusters(define_rdpm):
    rdpm = define_rdpm
    X = np.array([
        [2, 0.2], [2, 0.3], 
        [-2, -0.2], [-2, -0.1]
    ])
    rdpm.X = X
    rdpm.centroids = np.array([[2, 0], [-2, 0], [10, 10]])
    rdpm._labels = np.array([0, 0, 1])
    to_remove = np.array([False, False, True])

    rdpm._delete_centroids(to_remove)
    assert rdpm.n_clusters == 2
    assert np.all(rdpm.centroids != np.array([[10, 10]]))

def test_get_penalties(define_rdpm):
    from clustlib.utils.distance import euclidean_distance
    rdpm = define_rdpm
    X = np.array([
        [0, 1], [1, 0], 
        [-1, 0], [0, -1]
    ])
    rdpm.fit(X)

    centroids = rdpm.centroids

    diff = centroids - np.repeat(X[0, np.newaxis], 2, axis=0)
    distances = euclidean_distance(diff, axis=1).flatten()

    diff_allies = np.array([
       1, -2 
    ])

    penalty = distances - (rdpm.x0 * diff_allies)
    real_penalty = rdpm.get_penalties(0, 0)

    assert np.all(np.abs(penalty - real_penalty) < 1e-2), "Penalties do not match expected values"