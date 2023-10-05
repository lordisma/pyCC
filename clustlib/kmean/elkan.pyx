from cython cimport floating
from libc.stdlib import malloc, free
from libc.string import memset
from libc.math cimport sqrt, INFINITY
from cython.mem cimport PyMem_Malloc, PyMem_Free

cdef floating euclidean_distance(floating[::1] a, floating[::1] b):
    """
    Compute the euclidean distance between two points

    Parameters
    ----------

    a (in): array of floating point values
        First point
    b (in): array of floating point values
        Second point

    Returns
    -------

    distance: floating point value
        Euclidean distance between the two points
    """

    cdef:
        int i
        floating distance = 0.0

    for i in range(len(a)):
        distance += (a[i] - b[i]) ** 2

    return sqrt(distance)

cpdef void elkan_kmean_iteration(
    floating[::1] upper_bounds,
    floating[:, ::1] lower_bounds,
    floating[:, ::1] centroids,
    floating[:, ::1] points,
    int[::1] centroids_to_points,
):
    """
    Single Elkan k-means iteration of the algorithm.

    This method will compute check each point and update the lower and upper bounds for each points

    Parameters
    ----------

    upper_bounds (in/out): array of floating point values
        Upper bounds of the distances between each point and its closest centroid
    lower_bounds (in/out): array of floating point values
        Lower bounds of the distances between each point and the centroids
    centroids (in): array of floating point values
        Centroids of the clusters
    points (in): array of floating point values
        Points to cluster
    """

    cdef:
        int n_points = points.shape[0]
        int n_centroids = centroids.shape[0]
        floating min_distance = INFINITY
        int current_closer = 0
        floating temp_swap = 0.0

        float* clusters_distance = <float *> malloc(n_centroids * n_centroids * sizeof(float))
        int* centroids_closer_cluster = <int *> malloc(n_centroids * sizeof(int))

    for i in range(n_centroids):
        for j in reversed(range(n_centroids)):
            if i != j:
                clusters_distance[i + j] = euclidean_distance(centroids[i], centroids[j])

                if clusters_distance[i + j] < min_distance:
                    min_distance = clusters_distance[i + j]
                    current_closer = j
            else:
                # We don't want to compute the distance between a centroid and itself
                clusters_distance[i + j] = 0.0

        centroids_closer_cluster[i] = current_closer

    for i in range(n_points):
        for j in range(n_centroids):
            if upper_bounds[i] > (0.5 * clusters_distance[j + centroids_closer_cluster[j]]):
                lower_bounds[i, j] = euclidean_distance(points[i], centroids[j])

                if lower_bounds[i, j] < upper_bounds[i]:
                    upper_bounds[i] = euclidean_distance(points[i], centroids[centroids_to_points[i]])

                    if lower_bounds[i, j] < upper_bounds[i]:
                        centroids_to_points[i] = j

                        temp_swap = upper_bounds[i]
                        upper_bounds[i] = lower_bounds[i, j]
                        lower_bounds[i, j] = temp_swap
            else:
                lower_bounds[i, j] = upper_bounds[i]

    free(clusters_distance)
    free(centroids_closer_cluster)



