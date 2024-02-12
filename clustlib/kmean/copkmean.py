from .kmeans import BaseKMeans
import numpy as np


class COPKMeans(BaseKMeans):
    def __update_cl_distance(self, index, cl_prototypes):
        """Calculate the distance between an instance and a centroid.

        Parameters
        ----------
        index : int
            The index of the instance.
        cl_prototypes : list[int]
            The index of the centroids.
        """
        self.__lower_bounds[index, cl_prototypes] = np.inf

    def __get_valid_centroid(self, index):
        return super().__get_valid_centroid(index)
