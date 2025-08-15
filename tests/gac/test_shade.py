import logging
import sys
from math import sqrt

import numpy as np
import pytest as test
from mock import patch

from clustlib.gac.shade import ShadeCC

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@test.fixture
def define_shade():
    constraints = np.array([
        [0, 1, -1, -1], 
        [1, 0, -1, -1], 
        [-1, -1, 0, 1],
        [-1, -1, 1, 0]
    ])
    shade = ShadeCC(constraints=constraints, n_clusters=2, init="random", max_iter=10)
    yield shade
    del shade

def test_shade_initialization(define_shade):
    import math

    shade = define_shade
    assert shade._num_elite == math.ceil(20 * 0.2)

def test_shade_convergence(define_shade):
    shade = define_shade
    shade._delta = 1e-5
    
    assert shade._convergence()

def test_shade_create_population(define_shade):
    shade = define_shade
    X = np.array([[2, 1], [3, 0], [-2, 1], [-3, 0]])
    shade.X = X
    shade.create_population()

    assert shade.population.shape[0] == shade._population_size
    assert shade.population.shape[1] == shade.X.shape[0]
    assert shade._population_fitness.shape[0] == shade._population_size

    best = shade._population_fitness[0]
    worst = shade._population_fitness[-1]

    assert best <= worst, "Best fitness should be less than or equal to worst fitness"

def test_shade_decode_solution(define_shade):
    shade = define_shade
    shade.n_clusters = 4
    label = np.array([0, 1, 2, 3])
    code = np.array([.0, .3, .6, .8])
    result = shade.decode_solution(code)

    assert np.all(result == label), "decoded solution does not match expected labels"

def test_shade_predict(define_shade):
    shade = define_shade
    X = np.array([[5, 1], [5, 0], [-5, 0.2], [-5, 0.5]])
    shade.max_iter = 100 # Set a high max iteration to ensure convergence
    shade.fit(X)

    predictions = shade.predict(X)
    assert len(predictions) == len(X)
    assert predictions[0] == predictions[1], "Predictions do not match labels"
    assert predictions[2] == predictions[3], "Predictions do not match labels"
