import logging
import sys
from math import sqrt

import numpy as np
import pytest as test
from mock import patch

from clustlib.sin.dils import DILS

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@test.fixture
def define_dils():
    constraints = np.array(
        [[0, 1, -1, -1], [1, 0, -1, -1], [-1, -1, 0, 1], [-1, -1, 1, 0]]
    )
    dils = DILS(
        constraints=constraints,
        n_clusters=2,
        init="random",
        max_iter=10,
        mutation_size=2,
    )
    yield dils
    del dils


@patch.object(DILS, "get_single_fitness")
def test_dils_initialization(single_mock, define_dils):
    dils = define_dils
    X = np.array([[5, 1], [5, 0], [-5, 0.2], [-5, 0.5]])

    dils.X = X

    assert dils.best is None
    assert dils.worst is None

    single_mock.side_effect = [0.1, 1.0]

    with patch("numpy.random.randint") as RandMock:
        RandMock.side_effect = [np.array([[0, 0], [1, 1]])]
        dils.initialize()

    assert np.all(dils.best == np.array([0, 0]))
    assert np.all(dils.worst == np.array([1, 1]))


def test_dils_ml_infeasability(define_dils):
    dils = define_dils
    cromosome = np.array([1, 0, 1, 1])
    dils.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

    assert dils.ml_infeasability(cromosome) == 1

    cromosome = np.array([0, 0, 1, 1])
    assert dils.ml_infeasability(cromosome) == 0


def test_dils_cl_infeasability(define_dils):
    dils = define_dils
    cromosome = np.array([1, 0, 1, 1])
    dils.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

    assert dils.cl_infeasability(cromosome) == 2

    cromosome = np.array([0, 0, 1, 1])
    assert dils.cl_infeasability(cromosome) == 0


def test_dils_intra_cluster_distance(define_dils):

    dils = define_dils
    X = np.array([[2, 1], [3, 0], [-2, 1], [-3, 0]])
    dils.X = X

    distance = dils._intra_cluster_distance(np.array([0, 0, 1, 1]))

    assert np.isclose(
        distance, sqrt(2)
    ), "Intra-cluster distance does not match expected value"


def test_dils_get_single_fitness(define_dils):
    dils = define_dils
    X = np.array([[2, 1], [3, 0], [-2, 1], [-3, 0]])
    dils.X = X

    labels = np.array([0, 0, 1, 1])
    fitness = dils.get_single_fitness(labels)

    assert np.isclose(
        fitness, sqrt(2)
    ), "Single fitness value does not match expected value"


def test_dils_cross_over(define_dils):
    dils = define_dils
    X = np.array([[2, 1], [3, 0], [-2, 1], [-3, 0]])
    dils.X = X

    parent1 = np.array([0, 0, 1, 1])
    parent2 = np.array([1, 1, 0, 0])

    with patch("numpy.random.rand") as RandMock:
        dils._probability = 0.5
        RandMock.return_value = np.array([0.6, 0.6, 0.4, 0.4])

        offspring = dils.crossover(parent1, parent2)

    assert np.all(
        offspring == np.array([1, 1, 1, 1])
    ), "Crossover did not produce expected offspring"


def test_dils_mutation(define_dils):
    dils = define_dils
    X = np.array([[2, 1], [3, 0], [-2, 1], [-3, 0]])
    dils.X = X

    chromosome = np.array([0, 0, 1, 1])

    with patch("numpy.random.randint") as RandMock:
        RandMock.side_effect = [3, np.array([0, 1])]
        mutated_chromosome = dils.mutation(chromosome)

    assert np.all(
        mutated_chromosome == np.array([1, 0, 1, 0])
    ), "Mutation did not produce expected chromosome"


def test_dils_predict(define_dils):
    dils = define_dils
    X = np.array([[5, 1], [5, 0], [-5, 0.2], [-5, 0.5]])
    dils.max_iter = 100  # Set a high max iteration to ensure convergence
    dils.fit(X)

    predictions = dils.predict(X)
    assert len(predictions) == len(X)
    assert predictions[0] == predictions[1], "First cluster points not assigned"
    assert predictions[2] == predictions[3], "Second cluster points not assigned"
