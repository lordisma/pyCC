from .kmean import KMeans
import numpy as np


class ElkanKMeans(KMeans):
    _lower_bounds: np.ndarray
    _upper_bounds: np.ndarray

    _delta_centroid: np.ndarray
    _distance: np.ndarray

    def _initialize_bounds(self):
        """Initialize lower and upper bounds for each instance.

        This method will calculate the distance to each of the centroids in the cluster.
        After that, it will assign the closest centroid to each instance and apply the constraints
        to make sure that the instances respect the limitations.

        In case of conflict, the instance that is closer to the centroid will be kept, and the other
        will be moved to the next closest centroid.

        FIXME: This method is not efficient and should be refactored.

        NOTE: This method applies the constraints in a soft manner. Which means that the instances
        might be missclassified after the initialization.

        Parameters
        __________
        dataset: numpy.ndarray
            Training instances to cluster.

        Returns
        _______
        numpy.ndarray
            Lower bounds for each instance.
        numpy.ndarray
            Upper bounds for each instance.
        """
        n_clusters = self.centroids.shape[0]
        lower_bounds = np.zeros((self.X.shape[0], n_clusters))
        upper_bounds = np.zeros((self.X.shape[0]))

        self.__update_distance()

        # Initialize lower and upper bounds
        for instance_index, instance in enumerate(self.X):
            # Get the closest centroid to the instance
            self._labels[instance_index] = lower_bounds[instance_index, :].argmin()

            upper_bounds[instance_index] = np.min(
                lower_bounds[instance_index, self._labels[instance_index]]
            )

        # Apply the constraints to the newly created bounds and labels
        for instance_index in range(dataset.shape[0]):
            ml_constraints = self.constraints.get_ml_constraints(instance_index)
            cl_constraints = self.constraints.get_cl_constraints(instance_index)

            for (
                ml_constraint
            ) in ml_constraints:  # Soft ML constraints, it can be violated
                if self._labels[ml_constraint] != self._labels[instance_index]:
                    if upper_bounds[instance_index] > upper_bounds[ml_constraint]:
                        self._labels[instance_index] = self._labels[ml_constraint]
                        upper_bounds[instance_index] = lower_bounds[
                            instance_index, self._labels[ml_constraint]
                        ]
                    else:
                        self._labels[ml_constraint] = self._labels[instance_index]
                        upper_bounds[ml_constraint] = lower_bounds[
                            ml_constraint, self._labels[instance_index]
                        ]

            for cl_constraint in cl_constraints:
                if self._labels[cl_constraint] == self._labels[instance_index]:
                    if upper_bounds[instance_index] > upper_bounds[cl_constraint]:
                        lower_bounds[
                            instance_index, self._labels[instance_index]
                        ] = np.inf
                        new_centroid = lower_bounds[instance_index, :].argmin()
                        self._labels[instance_index] = new_centroid
                        upper_bounds[instance_index] = lower_bounds[
                            instance_index, new_centroid
                        ]
                    else:
                        lower_bounds[
                            cl_constraint, self._labels[cl_constraint]
                        ] = np.inf
                        new_centroid = lower_bounds[cl_constraint, :].argmin()
                        self._labels[cl_constraint] = new_centroid
                        upper_bounds[cl_constraint] = lower_bounds[
                            cl_constraint, new_centroid
                        ]

        return lower_bounds, upper_bounds
