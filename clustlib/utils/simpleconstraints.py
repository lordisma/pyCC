from itertools import repeat

import networkx as net
import numpy as np


class SimpleConstraints:
    __matrix: np.ndarray
    __ml: net.Graph
    __cl: net.Graph

    def __init__(self, matrix: np.ndarray):
        ml = np.copy(matrix)
        ml[np.where(matrix < 0)] = 0

        np.fill_diagonal(ml, 0)
        cl = np.copy(matrix)
        cl[np.where(matrix > 0)] = 0
        cl = np.absolute(cl)
        np.fill_diagonal(cl, 0)

        self.__cl = net.from_numpy_array(cl, create_using=net.Graph)
        self.__ml = net.from_numpy_array(ml, create_using=net.Graph)
        self.__matrix = np.copy(matrix)

    def propagate_ml(self):
        for i, paths in net.shortest_path_length(self.__ml):
            elements = list(filter(lambda j: i != j, paths.keys()))

            self.__matrix[i, elements] = 1
            self.__ml.add_edges_from(list(zip(repeat(i), elements)))
        pass

    def propagate_cl(self):
        for i, paths in net.shortest_path_length(self.__cl):
            elements = list(filter(lambda j: i != j, paths.keys()))

            self.__matrix[i, elements] = -1
            self.__cl.add_edges_from(list(zip(repeat(i), elements)))
        pass

    def __setitem__(self, pos, value):
        self.__matrix[pos] = value

    def __getitem__(self, pos):
        return self.__matrix[pos]

    @property
    def matrix(self):
        return self.__matrix

    @property
    def cl(self):
        return self.__cl

    @property
    def ml(self):
        return self.__ml

    pass
