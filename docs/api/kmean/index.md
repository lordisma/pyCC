# KMean's module
This module contains K-Mean implementation and derived algorithms from this family.

The K-Mean algorithm is a clustering algorithm that aims to partition n observations into k clusters in which each
observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a
partitioning of the data space into Voronoi cells.

Given the set of observations, the K-Mean algorithm first splits the spaces into K sets, then calculates the centroid of
each set using the following formula:

$$
\\mu_i = \\frac{1}{n_i} \\sum_{x_i \\in S_i} x_i
$$

Following then reassigns each observation to the cluster whose centroid is closest.

The algorithm has convergence when the assignment of instances to clusters no longer changes, or alternativetly, the
limit of iterations is reached.

The output is a set of centroids that represent the multiples divisions of the data space.