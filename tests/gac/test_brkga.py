import logging
import sys
from math import sqrt

import numpy as np
import pytest as test
from mock import patch

from clustlib.gac.brkga import BRKGA

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@test.fixture
def define_brkga():
    constraints = np.array([
        [0, 1, -1, -1], 
        [1, 0, -1, -1], 
        [-1, -1, 0, 1],
        [-1, -1, 1, 0]
    ])
    brkga = BRKGA(constraints=constraints, n_clusters=2, init="random", max_iter=10)
    yield brkga
    del brkga

def test_brkga_initialization(define_brkga):
    import math

    brkga = define_brkga
    assert brkga._num_elite == math.ceil(20 * 0.3)
    assert brkga._num_mutants == math.ceil(20 * 0.2)

def test_brkga_convergence(define_brkga):
    brkga = define_brkga
    brkga._delta = 1e-5
    
    assert brkga._convergence()

def test_brkga_create_population(define_brkga):
    brkga = define_brkga
    X = np.array([[2, 1], [3, 0], [-2, 1], [-3, 0]])
    brkga.X = X
    brkga.create_population()

    assert brkga.population.shape[0] == brkga._population_size
    assert brkga.population.shape[1] == brkga.X.shape[0]
    assert brkga._population_fitness.shape[0] == brkga._population_size

    best = brkga._population_fitness[0]
    worst = brkga._population_fitness[-1]

    assert best <= worst, "Best fitness should be less than or equal to worst fitness"

def test_brkga_decode_solution(define_brkga):
    brkga = define_brkga
    brkga.n_clusters = 4
    label = np.array([0, 1, 2, 3])
    code = np.array([.0, .3, .6, .8])
    result = brkga.decode_solution(code)

    assert np.all(result == label), "decoded solution does not match expected labels"

def test_brkga_crossover(define_brkga):
    brkga = define_brkga
    with patch('numpy.random.rand') as RandMock:
        mock = np.array([0.1, 0.2, 0.05, 0.7])
        RandMock.return_value = mock
        to_cross = np.where(mock < brkga._pbt_inherit)[0]

        parent1 = np.array([0, 1, 2, 3])
        parent2 = np.array([3, 2, 1, 0])
        offspring = brkga.crossover(parent1, parent2)

    expected = parent1
    expected[to_cross] = parent2[to_cross]

    assert np.all(offspring == expected), "Crossover did not produce expected offspring"

def test_brkga_predict(define_brkga):
    brkga = define_brkga
    X = np.array([[5, 1], [5, 0], [-5, 0.2], [-5, 0.5]])
    brkga.max_iter = 100 # Set a high max iteration to ensure convergence
    brkga.fit(X)

    predictions = brkga.predict(X)
    assert len(predictions) == len(X)
    assert predictions[0] == predictions[1], "Predictions do not match labels"
    assert predictions[2] == predictions[3], "Predictions do not match labels"
