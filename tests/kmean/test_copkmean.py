import numpy as np
import pytest as test

from clustlib.kmean.copkmeans import COPKMeans


@test.fixture
def define_copkmeans():
    constraints = np.array([[0, 1, -1], [1, 0, -1], [-1, -1, 0]])
    copkmeans = COPKMeans(constraints=constraints, n_clusters=2, init="random", max_iter=10)
    yield copkmeans
    del copkmeans

def test_copkmeans_initialization(define_copkmeans):
    copkmeans = define_copkmeans
    assert copkmeans.n_clusters == 2
    assert copkmeans.init == "random"
    assert copkmeans.max_iter == 10
    assert copkmeans.centroids is None

def test_copkmeans_fit(define_copkmeans):
    copkmeans = define_copkmeans
    assert not copkmeans.stop_criteria(0)

    X = np.array([[0, 1], [1, 0], [-1, -1]])
    copkmeans.fit(X)

    centroids = np.array([
        np.mean(np.array([[0,1], [1, 0]]), axis=0), 
        [-1, -1]
    ])

    assert np.mean(copkmeans.centroids - centroids) < 1e-2

def test_copkmeans_predict(define_copkmeans):
    copkmeans = define_copkmeans
    X = np.array([[0, 1], [1, 0], [-1, -1]])
    copkmeans.fit(X)

    predictions = copkmeans.predict(X)
    assert len(predictions) == len(X)
    assert np.all(np.isin(predictions, [0, 1]))

def test_get_valid_centroids(define_copkmeans):
    copkmeans = define_copkmeans
    X = np.array([[0, 1], [1, 0], [-1, -1]])
    copkmeans.fit(X)

    valid_centroids = copkmeans.get_centroids(0)
    assert len(valid_centroids) == 1